#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import TOKEN_FOR_MULTIPLE_CHOICE, TOKEN_FOR_LONG_ANSWER, TOKEN_FOR_SHORT_ANSWER, TOKEN_FOR_REPORT_GENERATION

# Import organ token indices for multi-index support
try:
    from stage3_turn.constants import ORGAN_TOKEN_INDICES, ORGAN_INDEX_TO_SLOT
    HAS_ORGAN_TOKENS = True
except ImportError:
    # Fallback for non-stage3_turn usage
    HAS_ORGAN_TOKENS = False
    ORGAN_TOKEN_INDICES = set()
    ORGAN_INDEX_TO_SLOT = {}

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            #self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        """
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()
        """
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = 512
        self.config.mm_context_size= 512
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        """
        把图像嵌入（来自CT-CLIP）转换成语言模型能理解的特征

        大白话：把"图像词汇"翻译成"语言词汇"

        输入: images (batch_size, D, H, W, C) - CT扫描的嵌入向量
              例如: (1, 28, 28, 28, 512)
              或者: list of tensors, each (1, D, H, W, C)
        输出: image_features (batch_size, N, hidden_size) - 图像token序列
              例如: (1, 21952, 4096)
              其中 N = D×H×W = 28×28×28 = 21952

        工作流程：
        1. Flatten空间维度 → 把3D网格展平成1D序列
        2. mm_projector投影 → 把512维视觉特征映射到4096维语言空间

        为什么flatten？
        - 语言模型只认"词序列"，不管3D结构
        - 就像把书页打乱，只要内容都在就行
        """
        #image_features = self.get_model().get_vision_tower()(images)
        # 注：CT-CHAT不用vision_tower，因为已经用CT-CLIP离线编码了

        # Handle list of images: concatenate along batch dimension
        if type(images) is list:
            # Concatenate all images: [img1, img2, ...] -> (N, D, H, W, C)
            images = torch.cat(images, dim=0)

        # If already 2D (N, hidden_size): features are pre-projected by independent/grouped
        # projectors in project_embeddings(). Skip flatten and mm_projector (Identity).
        if images.ndim == 2:
            return images

        # Step 1: Flatten空间维度 (D, H, W) → (D×H×W)
        # images: (batch, D, H, W, C) → (batch, D×H×W, C)
        # 例如: (1, 28, 28, 28, 512) → (1, 21952, 512)
        images = images.flatten(1, 3)  # flatten维度1,2,3，保留0和4

        # Step 2: mm_projector投影到语言空间
        # mm_projector是一个MLP: Linear(512) → GELU → Linear(4096)
        # 作用：把视觉特征映射到LLaMA的hidden_size
        # (1, 21952, 512) → (1, 21952, 4096)
        image_features = self.get_model().mm_projector(images)

        return image_features  # 返回"图像token序列"

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None
    ):
        """
        【核心函数】合并文本和图像，创建多模态输入序列

        大白话：把文字和图片"拼接"成一个序列，让模型能同时看到两者

        比喻：
        - input_ids是一本书的目录，标记了"第3页要插入一张图"
        - images是那张图片
        - 这个函数就是把图片真正插入到第3页

        输入：
        - input_ids: token序列，包含IMAGE_TOKEN_INDEX(-200)作为占位符
          例如: [128000, -200, 13, 15043, ...]
                        ↑
                  这里要插入图像特征
        - images: 图像嵌入 (batch, D, H, W, C)
        - image_sizes: 图像的元素总数 [D×H×W]

        输出：
        - new_input_embeds: 合并后的embedding序列
          [文本embed] + [21952个图像embed] + [文本embed]

        工作流程：
        1. 检查是否需要处理图像（early return优化）
        2. encode_images: 图像 → 图像token序列
        3. 找到所有IMAGE_TOKEN_INDEX(-200)的位置
        4. 在-200位置插入图像tokens
        5. 拼接成完整序列并padding

        关键技巧：
        - num_images == 0 时early return（没有<image>标记）
        - input_ids.shape[1] == 1 时early return（KV-cache优化）
        """
        #vision_tower = self.get_vision_tower()

        # Early return优化1: 没有图像
        # Early return优化2: input_ids只有1个token（auto-regressive生成的后续步骤）
        # 为什么input_ids.shape[1]==1时可以跳过？
        # - 生成时，第一步输入完整prompt（包含图像）
        # - 后续步骤只输入新生成的1个token，复用KV-cache
        # - 所以后续步骤不需要重新处理图像
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            #concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(images)
            split_sizes = [image.shape[0] for image in images]
            # Split image_features back to list if it's a tensor (from concat in encode_images)
            if isinstance(image_features, torch.Tensor):
                image_features = torch.split(image_features, split_sizes, dim=0)
                image_features = list(image_features)
                # Squeeze extra dimensions: (1, 1, hidden_size) -> (1, hidden_size)
                # Each image was (1, 1, 1, 1, 256), after flatten(1,3) -> (1, 1, 256), after projector -> (1, 1, hidden_size)
                image_features = [feat.squeeze(1) if feat.dim() == 3 and feat.shape[1] == 1 else feat for feat in image_features]
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                #image_features = [x.flatten(0, 2) for x in image_features]
                #image_features = [x for x in image_features]
                pass
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        #height = width = self.get_vision_tower().num_patches_per_side
                        #assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        # ============================================================
        # 核心逻辑：自适应多模态槽位对齐
        # ============================================================
        new_input_embeds = []  # 存储合并后的embeddings
        new_labels = []        # 存储对应的labels
        cur_image_idx = 0      # 顺序模式下的计数器
        
        # --- 关键：自适应步长计算 ---
        # 动态计算每个样本在 image_features 列表中占用的长度
        total_images = len(image_features)
        batch_size = len(input_ids)
        assert total_images % batch_size == 0, f"Total images {total_images} must be divisible by batch size {batch_size}"
        images_per_sample = total_images // batch_size 

        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 1. 统计当前样本中的图像 Token 数量
            if HAS_ORGAN_TOKENS:
                valid_image_tokens = ORGAN_TOKEN_INDICES
                num_images = sum((cur_input_ids == token_idx).sum() for token_idx in valid_image_tokens)
            else:
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            # 2. 处理无图像样本
            if num_images == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                # 自适应跳过：根据当前步长跳过对应的 slot
                cur_image_idx += images_per_sample 
                continue

            # 3. 找到图像 Token 物理位置
            image_token_indices = [-1]
            if HAS_ORGAN_TOKENS:
                for pos, token_id in enumerate(cur_input_ids):
                    if token_id.item() in ORGAN_TOKEN_INDICES:
                        image_token_indices.append(pos)
            else:
                image_token_indices += torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            image_token_indices.append(cur_input_ids.shape[0])

            # 4. 切分文本段
            cur_labels = labels[batch_idx]
            cur_input_ids_noim = []
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            # 5. 转换文本 Embeds
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            # 6. 交错合并
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                # 添加文本
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                # 添加图像
                if i < num_images:
                    token_pos = image_token_indices[i + 1]
                    token_id = cur_input_ids[token_pos].item()

                    # --- 自适应定位逻辑 ---
                    # 步长由 images_per_sample 决定
                    sample_start_idx = batch_idx * images_per_sample 
                    
                    if HAS_ORGAN_TOKENS and token_id in ORGAN_INDEX_TO_SLOT:
                        # 如果每样本只有1个embedding，slot强制为0；否则使用映射字典
                        slot = ORGAN_INDEX_TO_SLOT[token_id] if images_per_sample > 1 else 0
                        target_idx = sample_start_idx + slot
                    else:
                        # 兜底：使用自增索引
                        target_idx = cur_image_idx
                    
                    # 每次插入一个图，顺序计数器自增 (供 fallback 使用)
                    cur_image_idx += 1

                    # 插入图像特征 (带越界保护)
                    safe_idx = min(target_idx, total_images - 1)
                    cur_image_features = image_features[safe_idx]
                    cur_new_input_embeds.append(cur_image_features)
                    
                    # 同步插入 Label
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            new_input_embeds.append(torch.cat(cur_new_input_embeds))
            new_labels.append(torch.cat(cur_new_labels))
            


        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]


        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        print("the tokenizer is correctly initialized!")
        tokenizer.add_tokens(TOKEN_FOR_MULTIPLE_CHOICE, special_tokens=True)
        tokenizer.add_tokens(TOKEN_FOR_LONG_ANSWER, special_tokens=True)
        tokenizer.add_tokens(TOKEN_FOR_SHORT_ANSWER, special_tokens=True)
        tokenizer.add_tokens(TOKEN_FOR_REPORT_GENERATION, special_tokens=True)
        
        self.resize_token_embeddings(len(tokenizer))

        
        
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

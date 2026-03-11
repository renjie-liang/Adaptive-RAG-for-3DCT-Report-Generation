# AdaRAG-CT: Adaptive Retrieval-Augmented Generation for 3D CT Report Generation

> **Textual Augmentation Compensates for the Visual Bottleneck in 3D CT Report Generation**
>
> *ECCV 2026 (Medical Computer Vision Workshop)*

3D CT contrastive embeddings encode pathology-discriminative signals but carry as few as **2 effective dimensions out of 512**. This representational poverty — not the generator's capacity — is the fundamental bottleneck: scaling the LLM from 8B to 70B parameters yields no improvement. AdaRAG-CT addresses this by opening a supplementary textual channel through adaptive retrieval-augmented generation.

## Key Results

| Model | Params | Clinical F1 | BLEU-4 | ROUGE-L | LLaMA Score |
|-------|--------|-------------|--------|---------|-------------|
| CT-Agent | — | 0.420 | 0.231 | 0.490 | — |
| Ours (base) | 8B | 0.455 | 0.205 | 0.315 | 7.30 |
| **AdaRAG-CT** | **8B** | **0.480** | **0.242** | **0.354** | **7.75** |
| Ours (base) | 70B | 0.405 | 0.213 | 0.334 | 7.10 |
| **AdaRAG-CT** | **70B** | **0.426** | **0.250** | **0.361** | **7.53** |

## Installation

```bash
conda create -n adaragct python=3.10
conda activate adaragct
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 2.1
- Transformers >= 4.37
- PEFT >= 0.7
- FAISS (faiss-gpu)
- LLaMA-3.1 (8B or 70B)

## Data & Model Checkpoints

Download from [HuggingFace](https://huggingface.co/):

| Resource | Description | Link |
|----------|-------------|------|
| CT-RATE | Dataset (reports + CT scans) | [ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) |
| Embeddings | Pre-computed CT-CLIP + ViSD-Boost embeddings | TBD |
| Sentence DB | Per-organ sentence database + FAISS indices | TBD |
| Base 8B | Base model checkpoint (E29, step 5000) | TBD |
| AdaRAG-CT 8B | Best 8B model (P10, step 2000) | TBD |
| AdaRAG-CT 70B | Best 70B model (P14) | TBD |

## Quick Start: Inference

### Base Model (no retrieval)

```bash
python -m adaragct.inference.inference_rag \
    --checkpoint <base_checkpoint_path> \
    --no-rag \
    --output_jsonl results/base_predictions.jsonl
```

### AdaRAG-CT (with adaptive retrieval)

```bash
python -m adaragct.inference.inference_rag \
    --checkpoint <rag_checkpoint_path> \
    --context-jsonl <precomputed_context_path> \
    --output_jsonl results/rag_predictions.jsonl
```

### Parallel Inference (multi-worker, ~3× speedup)

```bash
python -m adaragct.inference.inference_parallel \
    --checkpoint <rag_checkpoint_path> \
    --context-jsonl <precomputed_context_path> \
    --num_workers 5 \
    --output_jsonl results/rag_predictions.jsonl
```

## Evaluation

```bash
python -m adaragct.eval.cal_metrics \
    --predictions results/rag_predictions.jsonl \
    --output results/metrics.json
```

Metrics: Clinical F1/Precision/Recall, BLEU-1/4, ROUGE-L, METEOR, LLaMA Score.

## Training (AdaRAG-CT)

### 1. Pre-compute Oracle Contexts

```bash
python -m adaragct.train.precompute_perplexity \
    --config configs/P10_rag_token_oracle07_noctx0_8b.yaml

python -m adaragct.train.precompute_oracle \
    --config configs/P10_rag_token_oracle07_noctx0_8b.yaml
```

### 2. RAG Token Training

```bash
python -m adaragct.train.train_rag \
    --config configs/P10_rag_token_oracle07_noctx0_8b.yaml
```

Key hyperparameters (in config):
- `p_oracle`: 0.7 (oracle-mixed ratio)
- `max_rag_per_sample`: 4
- LoRA: r=32, alpha=64, lr=1e-5

## Project Structure

```
AdaRAG-CT/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/                          # Experiment configs
│   ├── P10_rag_token_..._8b.yaml    #   8B AdaRAG-CT
│   ├── P14_rag_token_..._70b.yaml   #   70B AdaRAG-CT
│   └── E29_v2_final.yaml            #   8B base model reference
├── llava/                            # LLaVA model architecture (upstream)
│   ├── model/                        #   Language models, projectors, encoders
│   ├── train/train.py                #   Tokenizer utilities
│   ├── conversation.py
│   └── mm_utils.py
└── adaragct/                         # Main package
    ├── constants.py                  # Organ tokens, special token definitions
    ├── models/
    │   ├── build_model.py            # Base model building (build_model, build_model_from_peft)
    │   └── build_model_rag.py        # RAG inference model loading
    ├── data/
    │   ├── dataset.py                # CT-RATE dataset class
    │   └── oracle_dataset.py         # Oracle-mixed training dataset
    ├── train/
    │   ├── train_rag.py              # RAG token training (main entry)
    │   ├── train_utils.py            # Training utilities (load_config, save_checkpoint)
    │   ├── train_step.py             # Forward step + projector logic
    │   ├── loss.py                   # Context masking loss
    │   ├── precompute_oracle.py      # Oracle context precomputation
    │   └── precompute_perplexity.py  # Per-sentence perplexity scoring
    ├── inference/
    │   ├── inference_rag.py          # AdaRAG-CT inference (single process)
    │   ├── inference_parallel.py     # Parallel multi-worker inference (~3× speedup)
    │   ├── predict_base.py           # Base model inference (no retrieval)
    │   └── evaluate.py               # Inference evaluation wrapper
    ├── eval/
    │   ├── cal_metrics.py            # Unified metric computation
    │   ├── clinical_efficacy.py      # Clinical F1 / finding extraction
    │   ├── text_metrics.py           # BLEU, ROUGE-L, METEOR
    │   └── llama_score.py            # LLaMA-based evaluation
    └── utils/
        ├── logger.py                 # Logging + ETA
        ├── seed.py                   # Reproducibility
        ├── io.py                     # JSONL I/O
        ├── path_utils.py             # Path resolution
        ├── tokenizer_utils.py        # Organ token injection
        └── ...
```

## Citation

```bibtex
@inproceedings{liang2026adaragct,
  title={Textual Augmentation Compensates for the Visual Bottleneck in 3D CT Report Generation},
  author={Liang, Renjie and others},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgements

- [CT-RATE / CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) — dataset and contrastive encoder
- [ViSD-Boost](https://github.com/caohy123/ViSD-Boost) — organ-specific embeddings
- [LLaVA](https://github.com/haotian-liu/LLaVA) — vision-language architecture
- [Self-RAG](https://github.com/AkariAsai/self-rag) — adaptive retrieval paradigm

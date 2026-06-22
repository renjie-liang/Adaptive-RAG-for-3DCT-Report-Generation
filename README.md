# AdaRAG-CT

**Beyond the Embedding Bottleneck: Adaptive Retrieval-Augmented 3D CT Report Generation** (ECCV 2026)

[📄 arXiv](https://arxiv.org/abs/2603.15822) · [💻 GitHub](https://github.com/renjie-liang/Adaptive-RAG-for-3DCT-Report-Generation) · [🤗 Models & Data](https://huggingface.co/LiangRenjie/AdaRAG-CT)

Contrastive 3D CT embeddings concentrate 90% of their variance in just **2 of 512** dimensions, and scaling the LLM from 8B to 70B gives *no* gain — the bottleneck is visual, not generative. **AdaRAG-CT** compensates by retrieving organ-indexed report sentences and adaptively injecting them during generation, lifting Clinical F1 from 0.420 (CT-Agent) to **0.480**.

<p align="center">
  <img src="./figures/figure1.png" width="620">
</p>

## Results (CT-RATE validation)

| Model | Params | Clin-F1 | BLEU-4 | ROUGE-L | LLaMA |
|-------|:------:|:-------:|:------:|:-------:|:-----:|
| CT-CHAT (repro.) | 8B | 0.224 | 0.188 | 0.303 | 6.73 |
| CT-CHAT (repro.) | 70B | 0.161 | 0.182 | 0.321 | 6.02 |
| BTB3D | 8B | 0.258 | 0.213 | — | — |
| CT-Agent | — | 0.420 | 0.231 | 0.490 | — |
| Base (ViSD-Boost + CT-CLIP) | 8B | 0.455 | 0.205 | 0.315 | 7.30 |
| **AdaRAG-CT** | **8B** | **0.480** | **0.242** | **0.354** | **7.75** |
| Base (ViSD-Boost + CT-CLIP) | 70B | 0.405 | 0.213 | 0.334 | 7.10 |
| **AdaRAG-CT** | **70B** | **0.426** | **0.250** | **0.361** | **7.53** |

*CT-CHAT reproduced under our unified evaluation protocol. **Base (ViSD-Boost + CT-CLIP)** is our base model: it takes organ-level [ViSD-Boost](https://github.com/alibaba-damo-academy/ViSD-Boost) embeddings plus whole-volume [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) embeddings as visual input.*

## Install

```bash
conda create -n adaragct python=3.12 && conda activate adaragct
pip install -r requirements.txt
```

## Data & Checkpoints

All weights and retrieval data are on [🤗 HuggingFace](https://huggingface.co/LiangRenjie/AdaRAG-CT). Four checkpoints: **Base 8B/70B** (full) and **AdaRAG-CT 8B/70B** (LoRA adapter + projector, load on top of the matching base).

After downloading, place the `data/` folder and the checkpoint folders under `results/` at the repo root, matching the paths in the commands below — e.g. `results/base_8b/checkpoint`, `results/adaragct_8b/checkpoint_step_2000`. (The released predictions/metrics/logs under `results/` already come with this repo.)

## Evaluate

Score the released predictions directly:
```bash
python -m adaragct.eval.cal_metrics results/adaragct_8b/infer_step_2000.jsonl --output metrics.json
```
Add `--compute-llama-score` for the LLaMA score, `--bootstrap 1000` for 95% confidence intervals.

Or generate predictions, then score:
```bash
# AdaRAG-CT (adaptive retrieval)
python -m adaragct.inference.inference_rag --checkpoint results/adaragct_8b/checkpoint_step_2000 --output pred.jsonl

# Base / no-retrieval (same checkpoint, retrieval disabled)
python -m adaragct.inference.inference_rag --checkpoint results/adaragct_8b/checkpoint_step_2000 --no-rag --output base_pred.jsonl

python -m adaragct.eval.cal_metrics pred.jsonl --output metrics.json
```
Key inference flags: `--no-rag` (disable retrieval), `--text2text` (Text2Text retrieval pipeline), `--oracle` (oracle context), `--top-k`, `--max-retrievals`.

On-the-fly retrieval encodes each generated probe sentence with a fine-tuned text encoder (code + indices ship under `data/retrieval/`; it auto-downloads `microsoft/BiomedVLP-CXR-BERT-specialized` on first run). For a download-free run, use `--oracle` (precomputed contexts). To reproduce the exact paper numbers, score the released predictions in `results/`.

## Train

AdaRAG-CT trains a `[RAG]` trigger token on top of a frozen base model via LoRA, mixing oracle and retrieved contexts. Every required input — base checkpoint, CT embeddings, and the precomputed oracle/retrieval contexts — is provided in the HuggingFace bundle, so training runs directly with **no extra preprocessing**:

```bash
# 8B
python -m adaragct.train.train_rag --config configs/adaragct_8b.yaml
# 70B
python -m adaragct.train.train_rag --config configs/adaragct_70b.yaml
```

Each config points to `data/` (embeddings, `oracle_context_top3.jsonl`, `retrieval_context_top3.jsonl`) and a base checkpoint under `results/`. `base_8b.yaml` is the 8B base-model reference config.

## Citation

```bibtex
@misc{liang2026embeddingbottleneckadaptiveretrievalaugmented,
      title={Beyond the Embedding Bottleneck: Adaptive Retrieval-Augmented 3D CT Report Generation},
      author={Renjie Liang and Yiling Ma and Yang Xing and Zhengkang Fan and Jinqian Pan and Chengkun Sun and Li Li and Kuang Gong and Jie Xu},
      year={2026},
      eprint={2603.15822},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.15822},
}
```

## Acknowledgements

[CT-RATE / CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) · [ViSD-Boost](https://github.com/alibaba-damo-academy/ViSD-Boost) · [LLaVA](https://github.com/haotian-liu/LLaVA) · [Self-RAG](https://github.com/AkariAsai/self-rag)

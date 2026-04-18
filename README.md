# SG-DFN

**Seed-Guided Semantic–Structural Co-Differential Denoising Fusion Network for Semi-Supervised Entity Alignment**

English | [简体中文](./README_zh.md)

---

Official implementation of the paper *Seed-Guided Semantic–Structural Co-Differential Denoising Fusion Network for Semi-Supervised Entity Alignment* (**SG-DFN**).



## Overview

- Entity alignment (EA) aims to match entities referring to the same real-world objects across different knowledge graphs (KGs) with limited seed pair annotations. However, as the critical supervision signal, seed quality is heavily constrained by three intertwined noise sources. Specifically, encoding noise arises from cross-graph semantic and structural divergence, fusion noise is introduced by heterogeneous feature integration, and seed noise stems from unreliable single-view candidate selection. To systematically suppress these noise sources, thereby improving seed quality and alignment performance, we propose SG-DFN, a seed-guided semantic–structural co-differential denoising fusion network for semi-supervised entity alignment. First, we design a triple-to-text semantic enhancement strategy to unify discrete entity relations and attributes into coherent natural language descriptions, and employ relation-aware attention to purify structural representations, jointly suppressing encoding noise.  Then, we design a differential graph attention fusion mechanism that integrates the denoised semantic and structural features by computing the difference between two sets of attention weights, to cancel shared fusion noise and highlight discriminative information. Finally, we propose a multi-view collaborative seed selection strategy that jointly leverages semantic, structural, and fused features with cross-view consistency verification to identify high-confidence seed pairs, suppressing seed noise and improving seed quality for more reliable alignment. Extensive experiments demonstrate that SG-DFN achieves superior performance, confirming the effectiveness of our seed-guided denoising approach for entity alignment.



## Datasets

| Dataset | Subsets |
| :-- | :-- |
| **DBP15K** | ZH-EN, JA-EN, FR-EN |
| **SRPRS**  | EN-FR, EN-DE |



## Requirements

- Python 3.10+
- PyTorch
- NVIDIA GPU (experiments were run on an **A100 40GB**)
- Qwen3-32B (for entity-description generation)
- Qwen3-8B-Embedding (for text encoding)



## Project Structure

```
SG-DFN-CODE/
├── datasets/                # Raw benchmark data (DBP15K / SRPRS)
├── datasets_for_llm/        # LLM-ready triples & prompt JSON (input of LLM pipeline)
│   └── relate_code/         # Script that converts raw KG into LLM input JSON
├── entity_emb/              # LLM-generated semantic embeddings (.pkl, dim=4096)
├── cache_data/              # Preprocessed caches
├── llm_service/             # Thin async clients around OpenAI-compatible APIs
│   ├── llm_serve.py         # Qwen3-32B client for triple-to-text generation
│   └── emb_serve.py         # Qwen3-Embedding-8B client for text encoding
└── src/
    ├── main.py              # Training / evaluation entry point
    ├── base/                # KG loading & preprocessing
    │   ├── base_utils.py            # Device & entity-id helpers
    │   ├── data_kg_loader.py        # KGs: unified loader for triples / seeds / adj
    │   ├── data_att_loader.py       # Attribute parsing for DBP15K / SRPRS
    │   ├── data_llm_emb_loader.py   # Loads ent/rel/att LLM embeddings with caching
    │   └── hign_neighbor.py         # Builds top-k high-order neighbor graph
    ├── llm_data_utils/      # LLM-side data pipeline
    │   ├── data_loader_for_llm.py   # LLMKGProcessor: async batch description gen
    │   ├── data_pro_prompt.py       # EaPrompt: triple-to-text prompt templates
    │   ├── data_emb.py              # EntityEmbeddingProcessor: text → vector
    │   └── entity_alias_pro.py      # Entity name / alias fusion utilities
    ├── model_utils/         # Neural modules of SG-DFN
    │   ├── gnn_model.py             # Encoder_Model — top-level model
    │   ├── sem_layer.py             # Deep_Residual_Network for semantic encoding
    │   ├── gcn_layer.py             # Local_Global_Network: relation-aware GCN
    │   └── diff_gat.py              # DIFF_GraphAttention: differential GAT fusion
    ├── seed_utils/          # Semi-supervised seed mining
    │   ├── multi_seed_select.py     # MVCSS: multi-view cooperative seed selection
    │   └── sinkhorn.py              # Sinkhorn / Gumbel-Sinkhorn matching utilities
    ├── loss_utils/
    │   └── mulit_align_loss.py      # MultiLevelAlignmentLoss (sem / str / fused)
    └── eval_utils/
        └── evals.py                 # CSLS_evaluate: Hits@k / MRR metrics
```



## Data Preparation

1. **Raw KG** — place DBP15K / SRPRS under `datasets/<dataset>/<language>/`;
   each directory must expose `ent_ids_1`, `ent_ids_2`, `rel_ids_*`, `triples_*`,
   `sup_ent_ids`, `ref_ent_ids` and attribute files.
2. **LLM description generation** — run
   `python datasets_for_llm/relate_code/fusion_data_to_json.py` to produce the
   per-entity JSON consumed by `LLMKGProcessor`, then call `LLMKGProcessor` in
   `src/llm_data_utils/data_loader_for_llm.py` to generate entity descriptions
   via Qwen3-32B.
3. **Semantic embedding** — run `EntityEmbeddingProcessor` in
   `src/llm_data_utils/data_emb.py` to encode descriptions with Qwen3-Embedding-8B
   (4096-d). The resulting `*.pkl` files are placed under
   `entity_emb/<dataset>/<language>/`.
4. **Training** — `src/main.py` automatically loads these `.pkl` files through
   `LLMSemEmbeddingLoader` and caches adjacency / high-order matrices into
   `cache_data/` on first run.



## LLM Service

Both `llm_service/llm_serve.py` and `llm_service/emb_serve.py` use an
**OpenAI-compatible** async client. Before running the LLM pipeline, fill in
`api_key` and `base_url` (e.g. a vLLM or SiliconFlow endpoint):

```python
# llm_service/llm_serve.py
VLLM_MODEL_CHOICES = {"Qwen3-32B": "http://<host>:<port>/v1"}

# llm_service/emb_serve.py
self.embed_model_client = openai.AsyncClient(api_key="<your key>", base_url="<embed url>")
```

> **Note on embeddings.** The LLM-generated embedding files
> (`entity_emb/<dataset>/<language>/*_llm_{ent,rel,att}_emb_4096.pkl`, 4096-d)
> are too large to ship through Git, so they are **not** included in this
> repository. After configuring your own LLM / embedding endpoints above, you
> can reproduce them locally by running the following three files in order —
> they will regenerate the entity **relation / attribute descriptions** and the
> corresponding **semantic embeddings**:
>
> 1. `datasets_for_llm/relate_code/fusion_data_to_json.py` — fuse raw KG triples
>    and attributes into the per-entity JSON input.
> 2. `src/llm_data_utils/data_loader_for_llm.py` — call Qwen3-32B through
>    `LLMKGProcessor` to generate natural-language descriptions of each
>    entity's relations and attributes.
> 3. `src/llm_data_utils/data_emb.py` — call Qwen3-Embedding-8B through
>    `EntityEmbeddingProcessor` to encode the descriptions into the
>    `*_llm_{ent,rel,att}_emb_4096.pkl` files that `src/main.py` loads at
>    training time.



## Quick Start

```bash
git clone https://github.com/ShuoShan1/SG-DFN.git
cd SG-DFN
pip install -r requirements.txt

# (Optional) regenerate LLM descriptions & embeddings
python src/llm_data_utils/data_loader_for_llm.py
python src/llm_data_utils/data_emb.py

# Train & evaluate on DBP15K (ZH-EN)
python src/main.py --dataset DBP15K --language zh_en --gpu 0
```


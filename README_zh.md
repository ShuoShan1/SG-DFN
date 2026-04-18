# SG-DFN

**Seed-Guided Semantic–Structural Co-Differential Denoising Fusion Network for Semi-Supervised Entity Alignment**

[English](./README.md) | 简体中文

---

论文《Seed-Guided Semantic–Structural Co-Differential Denoising Fusion Network for Semi-Supervised Entity Alignment》（**SG-DFN**）的代码实现。



## 概述

实体对齐（EA）旨在利用有限的种子对标注，在不同的知识图谱（KG）中匹配指代同一现实世界对象的实体。然而，作为关键的监督信号，种子质量受到三个相互交织的噪声源的严重制约。具体而言，编码噪声源于跨图的语义和结构差异，融合噪声由异构特征融合引入，而种子噪声则源于不可靠的单视图候选选择。为了系统地抑制这些噪声源，从而提高种子质量和对齐性能，我们提出了SG-DFN，一种用于半监督实体对齐的种子引导语义-结构协同差分去噪融合网络。首先，我们设计了一种三元组到文本的语义增强策略，将离散的实体关系和属性统一为连贯的自然语言描述，并采用关系感知注意力机制来净化结构表示，从而共同抑制编码噪声。然后，我们设计了一种差分图注意力融合机制，通过计算两组注意力权重之间的差异来融合去噪后的语义和结构特征，从而消除共享的融合噪声并突出判别性信息。最后，我们提出了一种多视角协同种子选择策略，该策略结合了语义、结构和融合特征，并进行跨视角一致性验证，以识别高置信度的种子对，从而抑制种子噪声并提高种子质量，实现更可靠的对齐。大量实验表明，SG-DFN 取得了优异的性能，验证了我们提出的种子引导去噪方法在实体对齐方面的有效性。



## 数据集

| 数据集 | 子集 |
| :-- | :-- |
| **DBP15K** | ZH-EN、JA-EN、FR-EN |
| **SRPRS**  | EN-FR、EN-DE |



## 环境依赖

- Python 3.10+
- PyTorch
- NVIDIA GPU（实验在 **A100 40GB** 上完成）
- Qwen3-32B（用于生成实体描述）
- Qwen3-8B-Embedding（用于文本编码）



## 项目结构

```
SG-DFN-CODE/
├── datasets/                # 原始基准数据（DBP15K / SRPRS）
├── datasets_for_llm/        # LLM 流水线所需的三元组与 prompt JSON
│   └── relate_code/         # 将原始 KG 转换为 LLM 输入 JSON 的脚本
├── entity_emb/              # LLM 生成的语义嵌入 .pkl（维度 4096）
├── cache_data/              # 预处理缓存
├── llm_service/             # OpenAI 兼容接口的异步客户端封装
│   ├── llm_serve.py         # 调用 Qwen3-32B 生成三元组描述
│   └── emb_serve.py         # 调用 Qwen3-Embedding-8B 进行文本编码
└── src/
    ├── main.py              # 训练 / 评估入口
    ├── base/                # 知识图谱加载与预处理
    │   ├── base_utils.py            # 设备与实体 ID 辅助函数
    │   ├── data_kg_loader.py        # KGs：三元组 / 种子 / 邻接矩阵统一加载
    │   ├── data_att_loader.py       # DBP15K / SRPRS 属性解析
    │   ├── data_llm_emb_loader.py   # 加载并缓存实体/关系/属性嵌入
    │   └── hign_neighbor.py         # 构建 top-k 高阶邻居图
    ├── llm_data_utils/      # LLM 侧数据流水线
    │   ├── data_loader_for_llm.py   # LLMKGProcessor：异步批量生成实体描述
    │   ├── data_pro_prompt.py       # EaPrompt：三元组到文本的提示模板
    │   ├── data_emb.py              # EntityEmbeddingProcessor：文本 → 向量
    │   └── entity_alias_pro.py      # 实体名 / 别名融合工具
    ├── model_utils/         # SG-DFN 神经网络模块
    │   ├── gnn_model.py             # Encoder_Model：主模型类
    │   ├── sem_layer.py             # Deep_Residual_Network：深层语义编码
    │   ├── gcn_layer.py             # Local_Global_Network：关系感知 GCN
    │   └── diff_gat.py              # DIFF_GraphAttention：差分图注意力融合
    ├── seed_utils/          # 半监督种子挖掘
    │   ├── multi_seed_select.py     # MVCSS：多视角协同种子选择
    │   └── sinkhorn.py              # Sinkhorn / Gumbel-Sinkhorn 匹配工具
    ├── loss_utils/
    │   └── mulit_align_loss.py      # MultiLevelAlignmentLoss（语义 / 结构 / 融合）
    └── eval_utils/
        └── evals.py                 # CSLS_evaluate：Hits@k / MRR 评估
```



## 数据准备流程

1. **原始图谱**：将 DBP15K / SRPRS 放置于
   `datasets/<数据集>/<语言子集>/` 下，目录内需包含 `ent_ids_1`、
   `ent_ids_2`、`rel_ids_*`、`triples_*`、`sup_ent_ids`、`ref_ent_ids`
   以及属性文件。
2. **生成实体描述**：运行
   `python datasets_for_llm/relate_code/fusion_data_to_json.py`
   将原始图谱转换为 `LLMKGProcessor` 所需的 JSON，再通过
   `src/llm_data_utils/data_loader_for_llm.py` 中的 `LLMKGProcessor`
   调用 Qwen3-32B 批量生成实体自然语言描述。
3. **文本嵌入**：运行 `src/llm_data_utils/data_emb.py` 中的
   `EntityEmbeddingProcessor`，调用 Qwen3-Embedding-8B 将描述编码为 4096 维
   向量，产物 `.pkl` 写入 `entity_emb/<数据集>/<语言子集>/`。
4. **开始训练**：`src/main.py` 会通过 `LLMSemEmbeddingLoader` 自动加载这些
   `.pkl` 文件，并在首次运行时将邻接矩阵与高阶邻居结果缓存到 `cache_data/`。



## LLM 服务配置

`llm_service/llm_serve.py` 与 `llm_service/emb_serve.py` 均为 OpenAI 兼容的
异步客户端。运行 LLM 流水线前需填写 `api_key` 与 `base_url`
（例如本地 vLLM 或 SiliconFlow 接入点）：

```python
# llm_service/llm_serve.py
VLLM_MODEL_CHOICES = {"Qwen3-32B": "http://<host>:<port>/v1"}

# llm_service/emb_serve.py
self.embed_model_client = openai.AsyncClient(api_key="<你的 key>", base_url="<嵌入接口>")
```

> **关于嵌入文件。** LLM 生成的嵌入文件
> （`entity_emb/<数据集>/<语言子集>/*_llm_{ent,rel,att}_emb_4096.pkl`，维度
> 4096）体积过大，无法随 Git 一同上传，因此本仓库**不包含**这些 `.pkl`
> 文件。在配置好上述 LLM 与嵌入模型接口后，作者可以依次运行下面这三个文件，
> 自行生成实体的**关系 / 属性描述**及对应的**语义嵌入**：
>
> 1. `datasets_for_llm/relate_code/fusion_data_to_json.py`：将原始知识图谱的
>    三元组与属性融合为每个实体的 JSON 输入。
> 2. `src/llm_data_utils/data_loader_for_llm.py`：通过 `LLMKGProcessor`
>    调用 Qwen3-32B，为每个实体生成其关系与属性的自然语言描述。
> 3. `src/llm_data_utils/data_emb.py`：通过 `EntityEmbeddingProcessor`
>    调用 Qwen3-Embedding-8B，将上述描述编码为
>    `*_llm_{ent,rel,att}_emb_4096.pkl` 文件，供 `src/main.py` 在训练时加载。



## 快速开始

```bash
git clone https://github.com/ShuoShan1/SG-DFN.git
cd SG-DFN
pip install -r requirements.txt

# （可选）重新生成 LLM 描述与语义嵌入
python src/llm_data_utils/data_loader_for_llm.py
python src/llm_data_utils/data_emb.py

# 在 DBP15K (ZH-EN) 上训练并评估
python src/main.py --dataset DBP15K --language zh_en --gpu 0
```

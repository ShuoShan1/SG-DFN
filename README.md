# SG-DFN

**Seed-Guided Semantic–Structural Co-Differential Denoising Fusion Network for Semi-Supervised Entity Alignment**

[English](#english) | [简体中文](#简体中文)

---

## English

Official implementation of the paper *Seed-Guided Semantic–Structural Co-Differential Denoising Fusion Network for Semi-Supervised Entity Alignment* (**SG-DFN**).

### Overview

- Entity alignment (EA) aims to match entities referring to the same real-world objects across different knowledge graphs (KGs) with limited seed pair annotations. However, as the critical supervision signal, seed quality is heavily constrained by three intertwined noise sources. Specifically, encoding noise arises from cross-graph semantic and structural divergence, fusion noise is introduced by heterogeneous feature integration, and seed noise stems from unreliable single-view candidate selection. To systematically suppress these noise sources, thereby improving seed quality and alignment performance, we propose SG-DFN, a seed-guided semantic–structural co-differential denoising fusion network for semi-supervised entity alignment. First, we design a triple-to-text semantic enhancement strategy to unify discrete entity relations and attributes into coherent natural language descriptions, and employ relation-aware attention to purify structural representations, jointly suppressing encoding noise.  Then, we design a differential graph attention fusion mechanism that integrates the denoised semantic and structural features by computing the difference between two sets of attention weights, to cancel shared fusion noise and highlight discriminative information. Finally, we propose a multi-view collaborative seed selection strategy that jointly leverages semantic, structural, and fused features with cross-view consistency verification to identify high-confidence seed pairs, suppressing seed noise and improving seed quality for more reliable alignment. Extensive experiments demonstrate that SG-DFN achieves superior performance, confirming the effectiveness of our seed-guided denoising approach for entity alignment.

![image-20260418113123201](C:\Users\18323\AppData\Roaming\Typora\typora-user-images\image-20260418113123201.png)



### Datasets

| Dataset | Subsets |
| :-- | :-- |
| **DBP15K** | ZH-EN, JA-EN, FR-EN |
| **SRPRS**  | EN-FR, EN-DE |

### Requirements

- Python 3.8+
- PyTorch
- NVIDIA GPU (experiments were run on an **A100 40GB**)
- Qwen3-32B (for entity-description generation)
- Qwen3-8B-Embedding (for text encoding)

### Quick Start

```bash
git clone https://github.com/ShuoShan1/SG-DFN.git
cd SG-DFN
pip install -r requirements.txt

# Train & evaluate on DBP15K (ZH-EN)
python main.py
```

### Key Hyperparameters

| Parameter | Value |
| :-- | :-- |
| Embedding dim (entity / relation / attribute) | 100 |
| Deep semantic encoding layers | 2 |
| Local / global relation-aware layers | 2 |
| DGAF layers | 1 |
| MVCSS update interval | 30 epochs |
| MVCSS refinement rounds | 5 |
| Training seed ratio | 30% |
| Learning rate (ZH-EN, SRPRS) | 0.01 |
| Learning rate (JA-EN, FR-EN) | 0.0001 |

### Results

SG-DFN achieves state-of-the-art performance on all five benchmarks. Highlights:

| Dataset | Hits@1 | Hits@10 | MRR |
| :-- | :--: | :--: | :--: |
| DBP15K ZH-EN | **99.38** | **99.97** | **0.996** |
| DBP15K JA-EN | **99.62** | **99.92** | **0.997** |
| DBP15K FR-EN | **99.87** | **99.98** | **0.999** |
| SRPRS EN-DE  | **99.78** | **99.90** | **0.998** |
| SRPRS EN-FR  | **99.76** | **99.90** | **0.998** |



---

## 简体中文

论文《Seed-Guided Semantic–Structural Co-Differential Denoising Fusion Network for Semi-Supervised Entity Alignment》（**SG-DFN**）的官方代码实现。

### 概述

实体对齐（EA）旨在利用有限的种子对标注，在不同的知识图谱（KG）中匹配指代同一现实世界对象的实体。然而，作为关键的监督信号，种子质量受到三个相互交织的噪声源的严重制约。具体而言，编码噪声源于跨图的语义和结构差异，融合噪声由异构特征融合引入，而种子噪声则源于不可靠的单视图候选选择。为了系统地抑制这些噪声源，从而提高种子质量和对齐性能，我们提出了SG-DFN，一种用于半监督实体对齐的种子引导语义-结构协同差分去噪融合网络。首先，我们设计了一种三元组到文本的语义增强策略，将离散的实体关系和属性统一为连贯的自然语言描述，并采用关系感知注意力机制来净化结构表示，从而共同抑制编码噪声。然后，我们设计了一种差分图注意力融合机制，通过计算两组注意力权重之间的差异来融合去噪后的语义和结构特征，从而消除共享的融合噪声并突出判别性信息。最后，我们提出了一种多视角协同种子选择策略，该策略结合了语义、结构和融合特征，并进行跨视角一致性验证，以识别高置信度的种子对，从而抑制种子噪声并提高种子质量，实现更可靠的对齐。大量实验表明，SG-DFN 取得了优异的性能，验证了我们提出的种子引导去噪方法在实体对齐方面的有效性。

![image-20260418113131008](C:\Users\18323\AppData\Roaming\Typora\typora-user-images\image-20260418113131008.png)



### 数据集

| 数据集 | 子集 |
| :-- | :-- |
| **DBP15K** | ZH-EN、JA-EN、FR-EN |
| **SRPRS**  | EN-FR、EN-DE |

### 环境依赖

- Python 3.8+
- PyTorch
- NVIDIA GPU（实验在 **A100 40GB** 上完成）
- Qwen3-32B（用于生成实体描述）
- Qwen3-8B-Embedding（用于文本编码）

### 快速开始

```bash
git clone https://github.com/ShuoShan1/SG-DFN.git
cd SG-DFN
pip install -r requirements.txt

# 在 DBP15K (ZH-EN) 上训练并评估
python main.py
```

### 关键超参数

| 参数 | 取值 |
| :-- | :-- |
| 嵌入维度（实体 / 关系 / 属性） | 100 |
| 深层语义编码层数 | 2 |
| 局部 / 全局关系感知层数 | 2 |
| DGAF 层数 | 1 |
| MVCSS 更新间隔 | 30 epoch |
| MVCSS 优化轮数 | 5 |
| 训练种子比例 | 30% |
| 学习率（ZH-EN、SRPRS） | 0.01 |
| 学习率（JA-EN、FR-EN） | 0.0001 |

### 实验结果

SG-DFN 在五个基准数据集上均取得最优性能，关键指标如下：

| 数据集 | Hits@1 | Hits@10 | MRR |
| :-- | :--: | :--: | :--: |
| DBP15K ZH-EN | **99.38** | **99.97** | **0.996** |
| DBP15K JA-EN | **99.62** | **99.92** | **0.997** |
| DBP15K FR-EN | **99.87** | **99.98** | **0.999** |
| SRPRS EN-DE  | **99.78** | **99.90** | **0.998** |
| SRPRS EN-FR  | **99.76** | **99.90** | **0.998** |

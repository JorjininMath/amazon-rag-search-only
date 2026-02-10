# Amazon Reviews Search-Only Retrieval (No LLM)

一个基于公开 Amazon Reviews 数据集的 **Search-Only 检索系统**，专注于证据级/商品级检索与可评测性，而不依赖任何 LLM API。

当前项目已经实现：

- 使用 Hugging Face 开放数据集（Amazon Reviews）构建本地评论语料
- 使用 `sentence-transformers/all-MiniLM-L6-v2` 构建向量索引（本地缓存，无在线 API 依赖）
- 支持 query → Top-K 证据检索（评论级别，带 score/asin/rating/snippet）
- P0 / P0.5 / P1 三个阶段的可重复评测闭环（含 Hit@K 指标与前后对比）
- 简单意图路由（negative / compare / general）+ rating-aware rerank + compare 专项优化

> 当前状态：已完成 P1，整体 `hit@1` 从 0.50 提升到 0.80，`compare` 场景 top-1 稳定性已修复。

---

## 1. 目录结构（简要）

- `prepare_open_dataset.py`：从 Hugging Face 加载 Amazon Reviews，标准化为本项目 schema
- `build_index.py`：从 JSONL 构建向量索引到 `data/index/`
- `query_evidence.py`：命令行检索脚本（含 intent routing + rerank）
- `run_p0_baseline.py`：跑 P0/P1 基线，生成评测用 evidence
- `eval_retrieval.py`：计算 Hit@1/3/5、avg_best_rank 等检索指标
- `compare_eval.py`：比较 P0.5 与 P1 指标，输出 delta
- `p1_routing.py`：意图识别 + 重排逻辑（negative/compare/general）
- `P0_CHECKLIST.md`：P0 阶段操作清单
- `STAGE_SUMMARY.md`：完整阶段性总结（目标、阶段、指标、下一步计划）

数据相关：

- `data/processed/reviews_open.jsonl`：处理后的 Amazon Reviews 子集（约 5k 条）
- `data/index/`：向量索引目录
- `data/eval/`：
  - `golden_queries.jsonl`：标注的 10 条 golden query
  - `baseline_evidence.jsonl`：各 query 的 Top-K 证据
  - `p0_5_summary_before_p1.json`：P0.5 基线指标快照
  - `p0_5_summary.json`：当前最新指标
  - `p1_vs_p0_5_delta.json`：P1 vs P0.5 指标对比

---

## 2. 环境与安装

建议使用 Python 3.10+，并在虚拟环境中安装依赖：

```bash
cd /Users/zhaojin/Documents/Research/RAG

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` 中主要依赖：

- `datasets`（加载公开 Amazon Reviews 数据）
- `llama-index` + `llama-index-embeddings-huggingface`
- `sentence-transformers`

模型与数据缓存默认写入 `data/cache/`（在各脚本顶部通过环境变量控制）。

---

## 3. 从 0 到可运行（本地无 LLM）

### 3.1 准备开放数据集

```bash
source .venv/bin/activate
cd /Users/zhaojin/Documents/Research/RAG

python prepare_open_dataset.py \
  --dataset mczhu/amazon-reviews-2023 \
  --config All_Beauty \
  --split train \
  --max-samples 5000 \
  --output data/processed/reviews_open.jsonl
```

输出：`data/processed/reviews_open.jsonl`

### 3.2 构建向量索引

```bash
python build_index.py \
  --input data/processed/reviews_open.jsonl \
  --out-dir data/index
```

输出：`data/index/`（本地向量索引）

### 3.3 命令行检索（含 P1 rerank）

```bash
python query_evidence.py --query "what users dislike about this product" --top-k 5
```

输出示例（简要）：

- 显示自动识别的 `intent`（negative / compare / general）
- Top-K 证据行：
  - `rerank_score` / `raw_score`
  - `asin` / `title` / `rating`
  - 文本 snippet

---

## 4. 评测与对比（P0 / P0.5 / P1）

### 4.1 生成基线 evidence

```bash
python run_p0_baseline.py
```

输出：

- `data/eval/baseline_results.csv`
- `data/eval/baseline_evidence.jsonl`

### 4.2 计算检索指标（P0.5 / P1）

```bash
python eval_retrieval.py
```

输出：

- `data/eval/p0_5_metrics.csv`
- `data/eval/p0_5_summary.json`

当前主要指标（P1 后）：

- Overall：
  - `Hit@1 = 0.80`
  - `Hit@3 = 0.90`
  - `Hit@5 = 0.90`
  - `Avg Best Rank ≈ 1.11`
- By intent：
  - negative：`hit@1 = 0.80`
  - general：`hit@1 = 0.75`
  - compare：`hit@1 = 1.00`

### 4.3 P1 vs P0.5 指标对比

首次跑完 P0.5 后，将快照保存为：

- `data/eval/p0_5_summary_before_p1.json`

之后每次调 P1，只需：

```bash
python compare_eval.py
```

输出：

- `data/eval/p1_vs_p0_5_delta.json`
- 终端打印整体 `hit@1/3/5` 与 `avg_best_rank` 的 before/after delta

### 4.4 P1.5：per-query 影响诊断（可选）

如需分析“每条 query 上 rerank 具体做了什么”，可以运行：

```bash
python p1_impact_report.py
```

输出：

- `data/eval/p1_impact.csv`
  - 对每条 golden query 记录：
    - `best_rank_before` / `best_rank_after`
    - `asin_before` / `asin_after`
    - `polarity_before` / `polarity_after`

适合用来在报告/博客中展示 P1 改动对单条查询的具体影响。

---

## 5. 当前阶段定位与下一步

当前系统定位为：

- **Evidence Retrieval System**：专注于检索和证据输出，而非生成最终自然语言答案
- Search-only：不依赖任何 LLM / OpenAI API

已完成的关键点：

- P0：检索闭环 + 手工 yes/no 评估
- P0.5：Hit@1/3/5 与 avg_best_rank 等标准 IR 指标
- P1：意图路由 + rating-aware rerank + compare 场景专项优化（compare hit@1 达到 1.0）

下一步（P1.5 方向，仍不引入 LLM）：

- 增加更细粒度的失败/改进诊断（按 query 输出 rerank 前后 rank 变化）
- 输出商品级聚合结果（Top products + evidence），便于做 demo/展示
- 视需要对不同 intent 的 rerank 权重做参数化配置

更多细节、阶段性总结与设计思路，见 `STAGE_SUMMARY.md`。


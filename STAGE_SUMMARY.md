# Search-Only RAG 阶段总结（当前版本）

## 1. 项目目标（当前阶段）

本项目当前目标是构建一个**不依赖 LLM 的最小检索闭环**：

1. 使用公开 Amazon 评论数据集构建本地数据集  
2. 建立向量索引  
3. 对用户 query 返回 Top-K 证据评论  

也就是说，本阶段只做 `Retrieval`，不做生成式回答（Generation）。

---

## 2. 为什么先做 Search-Only

在项目早期，先把检索层单独跑通有三个好处：

- **可解释**：每条结果都能回到原始评论文本
- **可调试**：问题主要集中在数据、检索、排序，不会被生成模型掩盖
- **低成本**：无 API key 依赖、运行成本和复杂度更低

---

## 3. 数据来源与合规策略

为了避免爬虫和平台条款风险，当前使用的是 **Hugging Face 公共 Amazon Reviews 数据集**，不直接抓取 Amazon/Walmart 网页。

数据准备脚本：

- `prepare_open_dataset.py`

核心做法：

- 使用 `streaming=True` 读取公开数据集  
- 做最小字段标准化并输出 `jsonl`  
- 输出文件：`data/processed/reviews_open.jsonl`

当前样本规模：约 5000 条评论（本地已验证）。

---

## 4. 当前代码与功能（含 P1 已落地部分）

### 4.1 数据准备

- `prepare_open_dataset.py`
  - 从开放数据集读取评论
  - 统一字段为：
    - `asin`
    - `title`
    - `brand`
    - `category`
    - `rating`
    - `text`

### 4.2 索引构建

- `build_index.py`
  - 输入：`data/processed/reviews_open.jsonl`
  - 向量模型：`sentence-transformers/all-MiniLM-L6-v2`
  - 输出索引目录：`data/index`

### 4.3 Top-K 证据检索

- `query_evidence.py`
  - 输入 query
  - 当前流程：先召回候选（`3*top_k`）再做规则化重排
  - 输出字段包含：`raw_score`、`rerank_score`、`asin`、`rating`、文本片段
  - 会显示检测到的 query intent（`negative` / `compare` / `general`）

### 4.4 P1 路由与重排（已实现最小版）

- `p1_routing.py`
  - `detect_intent(query)`：基于关键词做意图识别（negative / compare / general）
  - `rerank_candidates(...)`：
    - negative：对低评分证据加权提升（rating-aware）
    - compare：加入 ASIN 多样性策略，避免结果全部来自同一商品

### 4.5 基线数据生成（已接入 P1 逻辑）

- `run_p0_baseline.py`
  - baseline evidence 生成已与线上检索逻辑对齐（同一套 intent + rerank）
  - 明细输出新增：
    - `detected_intent`
    - `rerank_score`

---

## 5. 已完成验证（含 P0 / P0.5 / P1-最小版）

已完成的最小闭环与基线验证如下：

1. 依赖安装成功（`requirements.txt`）
2. 数据集落盘成功（`reviews_open.jsonl`）
3. 索引构建成功（5000 条文档）
4. Query 可以返回 Top-K 证据
5. P0 基线评测已完成：
   - `data/eval/golden_queries.jsonl`（10 条查询）
   - `data/eval/baseline_results.csv`（含 yes/no 与备注）
   - `data/eval/baseline_evidence.jsonl`（每条 query 的 Top-K 明细）
6. P0.5 指标化评测已完成：
   - `eval_retrieval.py`（自动评测脚本）
   - `data/eval/p0_5_metrics.csv`（逐 query 的 `hit` / `best_rank`）
   - `data/eval/p0_5_summary.json`（聚合指标）
   - 指标包含：`hit@1` / `hit@3` / `hit@5` / `avg_best_rank`
   - 基线快照：`data/eval/p0_5_summary_before_p1.json`
7. P1（最小版 + compare 专项优化）已验证，核心结果如下：
   - P0.5 基线（改动前）：
     - `hit@1 = 0.50`
     - `hit@3 = 0.90`
     - `hit@5 = 0.90`
     - `avg_best_rank = 1.444`
  - P1 当前版本（改动后）：
    - `hit@1 = 0.80`（+0.30）
     - `hit@3 = 0.90`（持平）
     - `hit@5 = 0.90`（持平）
    - `avg_best_rank = 1.111`（更好，越低越好）
  - 分意图表现（改动前 vs 改动后）：
    - negative：
      - 改动前：`hit@1 = 0.40` / `hit@3 = 0.80` / `hit@5 = 0.80` / `avg_best_rank = 1.50`
      - 改动后：`hit@1 = 0.80` / `hit@3 = 0.80` / `hit@5 = 0.80` / `avg_best_rank = 1.00`
    - general：
      - 改动前：`hit@1 = 0.75` / `hit@3 = 1.00` / `hit@5 = 1.00` / `avg_best_rank = 1.25`
      - 改动后：`hit@1 = 0.75` / `hit@3 = 1.00` / `hit@5 = 1.00` / `avg_best_rank = 1.25`
    - compare：
      - 改动前：`hit@1 = 0.00` / `hit@3 = 1.00` / `hit@5 = 1.00` / `avg_best_rank = 2.00`
      - 改动后：`hit@1 = 1.00` / `hit@3 = 1.00` / `hit@5 = 1.00` / `avg_best_rank = 1.00`
      - 结论：`compare` top-1 稳定性已修复
  - P1.5 诊断报告（per-query impact），见 `data/eval/p1_impact.csv`：
    - 对每条 golden query，同时记录：
      - `best_rank_before`：纯相似度排序下命中证据的排名（不经过 rerank）
      - `best_rank_after`：经过 intent-aware rerank 后命中证据的排名
      - `asin_before/asin_after`：命中证据所在商品是否发生变化
      - `polarity_before/polarity_after`：命中证据情感极性是否发生变化
    - 关键观察：
      - q02 / q07 / q10：`best_rank` 从 2 提升到 1，`asin` 与极性保持不变 → rerank 主要是把正确证据提前
      - q08：`best_rank` 仍为 1，但 `asin` 与极性从“中性”切换为“负面” → rerank 选择了更符合负向意图的证据
      - q03：在当前 target 规则下，rerank 前后均未命中 → 可作为后续规则/数据侧误差分析样本
8. 新增自动对比脚本（避免手工抄数）：
   - `compare_eval.py`
   - 默认读取：
     - before：`data/eval/p0_5_summary_before_p1.json`
     - after：`data/eval/p0_5_summary.json`
   - 输出：`data/eval/p1_vs_p0_5_delta.json`
   - 运行命令：`python compare_eval.py`
9. 两个 checkpoint 的当前落地方式（本阶段不新增额外检查脚本）：
   - Checkpoint A（intent 稳定性）：
     - 在当前 10 条 golden query 上，`detected_intent` 与标注 intent 一致（mismatch = 0）
     - 当前以阶段总结文档记录为主，不单独新增检查脚本
   - Checkpoint B（指标对齐）：
     - 已由 `eval_retrieval.py` + `compare_eval.py` 自动输出
     - 对比结果以 `data/eval/p1_vs_p0_5_delta.json` 为准（当前总体 `hit@1` 提升 +0.30）

这说明系统已经从“有数据”进入“可检索”状态。

---

## 6. 当前能力边界（对外要讲清）

系统现在已经支持“用户输入问题并返回证据”，但**还不是完整问答系统**。

当前能做：

- 接收 query 并返回 Top-K 证据评论
- 输出可审计字段（score、asin、rating、文本片段）

当前还不能稳定做到：

- 直接给出“符合用户主观预期”的最终答案
- 自动区分“语义相关”与“意图满足”（例如负评意图）

换句话说：当前是 **evidence retrieval system**，不是 **final answering system**。

---

## 7. 当前发现的真实问题（预期内）

在 query 如 `negative review for this product` 场景中，系统可能会返回包含“negative review”词面的文本，但语义上并不总是最负面。

原因：

- 当前优化目标是**语义相似度**
- 还没有显式建模“负面意图/情感极性”

这不是 bug，而是纯语义检索阶段的典型现象。

---

## 8. 下一阶段计划（进入 P1.5）

P0.5 已完成，P1 的 intent/rerank/compare 优化已验证。下一步建议进入 P1.5（仍不引入 LLM）：

- 对外定位结论：Compare intent has reached stable top-1 performance after structure-aware reranking; further improvements are expected to be marginal without introducing new signals.
- 增加更细粒度的失败诊断（按 query 输出 rerank 前后 rank 变化）
- 做商品级聚合输出（Top products + evidence），提升 demo 展示性
- 视需要再做轻量权重参数化（intent 级别的可配置权重）

---

## 9. 一句话结论

当前项目已经完成了一个可运行、可验证、可迭代的 **Search-Only RAG 基线系统**。  
当前已完成 P1 关键目标：系统在不引入 LLM 的情况下，将总体 `hit@1` 从 `0.50` 提升到 `0.80`，并修复了 compare 场景的 top-1 稳定性。


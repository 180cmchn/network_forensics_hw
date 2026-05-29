# 网络入侵检测与数字取证大作业方法说明

## 任务概览

本次作业包含三套 DNS 域名分析任务：

1. `dns1`：恶意域名家族多分类。训练标签只覆盖约一半恶意域名，需要发现剩余恶意域名并预测其家族编号，提交文件只包含预测为恶意的域名。
2. `dns2`：黑白域名二分类。基于域名字符串、访问行为和解析行为特征，对所有待分类域名输出黑/白标签。
3. `dns3`：域名聚类。基于请求特征和解析特征对域名聚类，簇数不少于 3。

## 特征设计

- 域名字符串特征：长度、层级、TLD、数字/字母/符号比例、括号词数量、字符熵、分段统计、字符 n-gram。
- 访问行为特征：请求量、客户端数、活跃时间跨度、小时分布、请求集中度、突发性、请求量占全网比例。
- DNS 解析特征：解析类型数量、解析结果数量、请求量统计、共享解析目标、解析图度数与连通分量。
- IP/地理/ISP 特征：国家、城市、ISP 多样性，主导 ISP/国家，解析基础设施复用情况。
- 任务专属特征：`dns1` 使用 WHOIS 派生字段；`dns3` 使用 TTL 统计和 TTL 与 IP 变化的交互特征。

## 建模策略

- `dns1`：先用已知恶意样本与未标注样本训练保守的恶意风险模型，再在已知恶意样本上训练家族分类模型；对高置信度未标注恶意域名输出家族编号。
- `dns2`：使用监督二分类模型并通过交叉验证选择阈值。
- `dns3`：将标准化数值特征、解析图特征和 TTL 特征结合，使用聚类模型生成不少于 3 个簇。

## 分数提升过程

最终提交包不是一次性脚本输出后直接提交，而是在基础求解结果上做了受控的在线反馈迭代。核心原则是：一次只改一个候选、提交前固定 `dns2`/`dns3` 哈希、记录线上反馈，再决定是否继续。

| 阶段 | 题目 | 线上分数 | 变体/状态 | 决策 |
| --- | --- | ---: | --- | --- |
| 初始受保护基线 | `dns1` | `80.87` | baseline / rollback candidate | 作为后续 probe 的保护基线。 |
| Probe 1 | `dns1` | `76.96` | `nn_rerank_rows1000_original_classifier`，1000 行 | 低于基线；按人工约束不回滚 Gitee，仅记录失败并准备下一轮。 |
| Probe 2 | `dns1` | `78.48` | `nn_rerank_rows1050_original_classifier`，1050 行 | 比 Probe 1 高 `+1.52`，但仍低于基线；继续寻找更保守的召回扩展方案。 |
| Probe 3 | `dns1` | `82.42` | `conservative_rows1150_original_classifier`，1150 行 | 高于基线 `+1.55`；取消 Probe 4，固定为最终 `dns1` 答案。 |
| 冻结 | `dns2` | `95.51` | 最高分答案 | 不参与 `dns1` probe，始终按哈希冻结。 |
| 冻结 | `dns3` | `15.31` | K=3 聚类答案 | 不参与 `dns1` probe，始终按哈希冻结。 |

`dns1` 提分的关键变化是从更激进的 `nn_rerank` 风险重排，切换到保守的 PU/noisy/NN 风险混合，并把总行数扩展到 1150 行：Probe 3 的 `conservative_rows1150_original_classifier` 在扩大召回的同时降低最大家族占比，线上得分超过基线，因此停止继续 probe。

上表已经把可复查的线上反馈、候选名称、行数和最终决策写入本文档；最终答案文件以 `submission/answers/` 中的 CSV 和 SHA256 为准。

## 复现方式

以下命令默认在仓库根目录运行。`data/` 为课程数据目录，不纳入 Git；`results/` 和 `results_candidate/` 为本地复现/实验输出目录，也不纳入 Git。

### 1. 环境设置

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2. 基础求解复现

基础求解脚本会重新生成三道题的本地输出，保存到 `results/dns1/label.csv`、`results/dns2/label.csv`、`results/dns3/label.csv`：

```bash
.venv/bin/python scripts/solve_dns_homework.py
.venv/bin/python scripts/validate_outputs.py --results results --data data
```

### 3. 最终提交包验证

最终提交包位于 `submission/answers/`，其中包含经过在线反馈和实验流水线确认后的最高分组合。校验提交包格式和当前哈希可运行：

```bash
.venv/bin/python -m py_compile scripts/run_score_experiments.py scripts/score_exp_*.py scripts/render_dns3_cluster_report.py
.venv/bin/python scripts/run_score_experiments.py --help
.venv/bin/python scripts/validate_outputs.py --results submission/answers --data data --answers-layout
shasum -a 256 submission/answers/dns1.csv submission/answers/dns2.csv submission/answers/dns3.csv
```

当前最终提交包对应的线上分数与 SHA256 为：

| 题目 | 线上分数 | 提交文件 | SHA256 |
| --- | ---: | --- | --- |
| `dns1` | `82.42` | `submission/answers/dns1.csv` | `ae5dee85264ef3dbcd25525c12a3be5b8a76419e208cfe34db36eefee5d6620d` |
| `dns2` | `95.51` | `submission/answers/dns2.csv` | `f12ddfbc1296d6876cfaed019227fbb708edd11f3638cbe100f7ba73b02d9c1c` |
| `dns3` | `15.31` | `submission/answers/dns3.csv` | `bbdd3795a83d3322350ac602c6213e2b1fc6be17899e3d7294fb149d727c78e6` |

### 4. 实验流水线命令

`scripts/run_score_experiments.py` 是实验入口，`scripts/score_exp_*.py` 是拆分后的实现模块。该入口统一使用：

```bash
.venv/bin/python scripts/run_score_experiments.py \
  --task {dns1,dns3,all} \
  --mode {baseline,quick-grid,dns1-improvement,dns3-focus,dns3-interpretable,finalize} \
  --run-id <run-id>
```

只读检查命令（不生成候选、不改提交包）：

```bash
.venv/bin/python -m py_compile scripts/run_score_experiments.py scripts/score_exp_*.py scripts/render_dns3_cluster_report.py
.venv/bin/python scripts/run_score_experiments.py --help
.venv/bin/python scripts/validate_outputs.py --results submission/answers --data data --answers-layout
shasum -a 256 submission/answers/dns1.csv submission/answers/dns2.csv submission/answers/dns3.csv
```

下面命令会在本地生成候选输出或刷新报告摘要，适合提分阶段使用，不建议在普通验证或 CI 中反复执行；这些中间产物不作为 GitHub 仓库中的最终交付物。

```bash
# dns3 基线代理指标，会写本地候选/摘要输出
.venv/bin/python scripts/run_score_experiments.py \
  --task dns3 --mode baseline --run-id dns3_baseline_check

# dns1 提分候选：风险混合、总行数、家族分配策略的受控网格
.venv/bin/python scripts/run_score_experiments.py \
  --task dns1 --mode dns1-improvement --run-id dns1_improvement_grid

# dns3 快速网格：不同 K、文本加权矩阵、可选 HDBSCAN 等候选
.venv/bin/python scripts/run_score_experiments.py \
  --task dns3 --mode quick-grid --run-id dns3_quick_grid

# dns3 可解释网格：流量/解析/IP/词汇路线和硬门控
.venv/bin/python scripts/run_score_experiments.py \
  --task dns3 --mode dns3-interpretable --run-id dns3_interpretable_grid

# 最终选择/提交包校验，会检查锁定哈希并可能刷新报告摘要
.venv/bin/python scripts/run_score_experiments.py \
  --task all --mode finalize --run-id final_package_check
```

### 5. `score_exp_*.py` 文件用途

| 文件 | 作用 |
| --- | --- |
| `scripts/run_score_experiments.py` | 实验 CLI 门面，解析 `--task`、`--mode`、`--run-id`，并分发到对应模块。 |
| `scripts/score_exp_common.py` | 通用 I/O 和工具函数：JSON/CSV、SHA256、路径、矩阵形状、安全类型转换。 |
| `scripts/score_exp_config.py` | 实验常量：输出目录、候选网格、随机种子、锁定哈希、风险混合和代理评分公式。 |
| `scripts/score_exp_types.py` | 候选结果、特征 bundle、HDBSCAN 状态等 dataclass。 |
| `scripts/score_exp_metrics.py` | 标签分布、聚类熵、ARI/NMI、轮廓系数、Davies-Bouldin、Calinski-Harabasz 等指标。 |
| `scripts/score_exp_artifacts.py` | 产物管理：输出目录、候选 registry、基线复制、哈希查找、验证状态。 |
| `scripts/score_exp_dns1.py` | `dns1` 专用实验：风险评分、候选生成、家族策略、proxy 排序、`quick-grid` 和 `dns1-improvement`。 |
| `scripts/score_exp_dns3_features.py` | `dns3` 特征矩阵、SVD 降维、文本加权矩阵和可解释路线构建。 |
| `scripts/score_exp_dns3_core.py` | `dns3` 核心评估：标签写入、proxy 分数、聚类诊断、profile lift 和 separation ratio。 |
| `scripts/score_exp_dns3_grid.py` | `dns3` baseline/quick-grid/focus-grid 搜索，包括 KMeans、Birch、GMM、可选 HDBSCAN 等候选。 |
| `scripts/score_exp_dns3_interpretable.py` | `dns3` 可解释候选网格，使用流量/解析/IP/词汇路线和 NMI/ARI/separation/max-cluster 等硬门控。 |
| `scripts/score_exp_finalize.py` | 最终选择与提交包验证：锁定哈希、复制候选、校验 `submission/answers/`、生成 summary。 |

# network_forensics_hw

这个仓库对应“网络入侵检测与数字取证大作业”的三道 DNS 题目：`dns1`、`dns2`、`dns3`。

为了方便查找，下面按“题目 → 文件”整理仓库里的对应关系。

## 题目与文件对应关系

| 题目 | 最终提交文件 | `submission/` 目录中的对应文件 | 备注 |
| --- | --- | --- | --- |
| `dns1` | `submission/answers/dns1.csv` | `submission/dns1/label.csv` | `dns1` 是恶意域名家族多分类任务。 |
| `dns2` | `submission/answers/dns2.csv` | `submission/dns2/label.csv`、`submission/dns2/label2.csv`、`submission/dns2/label2.numbers` | `dns2` 是黑白域名二分类任务。 |
| `dns3` | `submission/answers/dns3.csv` | `submission/dns3/label.csv` | `dns3` 是域名聚类任务。对应可视化报告见 `reports/dns3_cluster_visualization.html`。 |

## 共享脚本

这些脚本不是只服务某一道题，而是整个作业流程共用或和多题相关：

- `scripts/solve_dns_homework.py`：主求解脚本，统一生成三道题的输出。
- `scripts/run_score_experiments.py`：实验驱动 CLI 入口，负责参数解析、模式分发和兼容导出。
- `scripts/score_exp_*.py`：`run_score_experiments.py` 拆分后的实验辅助模块，按配置、类型、指标、候选产物、`dns1`、`dns3` 特征/评估/网格/最终选择等职责组织。
- `scripts/render_dns3_cluster_report.py`：把 `dns3` 的最终聚类结果渲染成 HTML 可视化报告。
- `scripts/validate_outputs.py`：校验 `dns1`、`dns2`、`dns3` 输出格式是否符合要求。

### `run_score_experiments.py` 模块结构

`scripts/run_score_experiments.py` 保持为唯一的命令行入口；下游脚本仍可通过它导入
`build_dns3_feature_bundle()` 和 `build_dns3_interpretable_routes()`。具体实验逻辑拆到以下模块：

| 模块 | 主要职责 |
| --- | --- |
| `scripts/score_exp_common.py` | JSON/CSV、路径、哈希、矩阵形状和安全类型转换等通用工具。 |
| `scripts/score_exp_config.py` | 实验常量、证据目录、候选网格、锁定哈希和阈值。 |
| `scripts/score_exp_types.py` | 候选结果和特征 bundle 的共享 dataclass。 |
| `scripts/score_exp_metrics.py` | 标签分布、聚类熵、ARI/NMI、特征空间指标等共享评分函数。 |
| `scripts/score_exp_artifacts.py` | 输出目录、候选 registry、基线复制和验证状态等产物辅助函数。 |
| `scripts/score_exp_dns1.py` | `dns1` 特征构建、候选生成、评分、registry 和改进模式。 |
| `scripts/score_exp_dns3_features.py` | `dns3` 特征矩阵、特征子集和可解释路线构建。 |
| `scripts/score_exp_dns3_core.py` | `dns3` 标签诊断、候选 gate、profile lift 和核心评估。 |
| `scripts/score_exp_dns3_grid.py` | `dns3` baseline、quick-grid 和 focus-grid 候选搜索。 |
| `scripts/score_exp_dns3_interpretable.py` | `dns3` 可解释候选网格、硬 gate 和候选 registry。 |
| `scripts/score_exp_finalize.py` | 最终选择、锁定哈希检查、提交包验证和 Task 7 汇总。 |

轻量检查命令：

```bash
.venv/bin/python -m py_compile scripts/run_score_experiments.py scripts/score_exp_*.py scripts/render_dns3_cluster_report.py
.venv/bin/python scripts/run_score_experiments.py --help
.venv/bin/python scripts/validate_outputs.py --results submission/answers --data data --answers-layout
```

## 说明与报告文件

- `reports/method_summary.md`：方法说明。
- `reports/run_summary.md`：运行结果摘要。
- `reports/dns3_cluster_visualization.html`：`dns3` 聚类结果的静态 HTML 可视化报告。
- `submission/method_summary.md`：随提交材料整理的说明文件副本。
- `submission/run_summary.md`：随提交材料整理的运行摘要副本。
- `submission/solve_dns_homework.py`：随提交材料整理的求解脚本副本。

## 关于 `results/`

`scripts/solve_dns_homework.py` 运行后会把结果写到本地的：

- `results/dns1/label.csv`
- `results/dns2/label.csv`
- `results/dns3/label.csv`

这些 `results/` 文件在 `.gitignore` 中被排除了，所以不会出现在 GitHub 仓库里。仓库中实际保留的是 `submission/` 和 `reports/` 下的版本。

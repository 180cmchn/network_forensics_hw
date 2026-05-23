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
- `scripts/run_score_experiments.py`：实验驱动脚本，用于候选方案、评分改进和结果选择。
- `scripts/render_dns3_cluster_report.py`：把 `dns3` 的最终聚类结果渲染成 HTML 可视化报告。
- `scripts/validate_outputs.py`：校验 `dns1`、`dns2`、`dns3` 输出格式是否符合要求。

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

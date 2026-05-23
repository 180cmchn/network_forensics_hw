# DNS Homework Solver Run Summary

Updated by Task 4 reporting correction. This summary reflects the current final local package after dns1 rollback and deterministic dns3 focus selection; no new dns3 experiments were run for this correction.

## Final Task 4 deterministic package
- dns1: rollback baseline because online feedback dropped from `80.87` to `78.51`; final hash `e697c1c86681c38b385fc955370895c03761b9a9015f8d664ebd94ec23c0fd54`; rows `1000`.
- dns2: frozen baseline; final hash `f12ddfbc1296d6876cfaed019227fbb708edd11f3638cbe100f7ba73b02d9c1c`; rows `17468`.
- dns3: selected focus candidate `traffic_dns_only_k12`; final hash `4507b2fbebc005ec63879df6068ef5a8c6e8791c84ab38c7cf0165064b976724`; rows `17468`.

## dns1 rollback details
- Decision: rollback to known better baseline, not the old dns1 primary candidate.
- Reason: online dns1 score dropped `80.87 -> 78.51` after the failed candidate, while the rollback baseline hash is `e697c1c86681c38b385fc955370895c03761b9a9015f8d664ebd94ec23c0fd54`.
- Failed dns1 candidate hash kept absent from final staged answers: `b3d60d8d4dd6363760ea7774c446aef864a59b58f8c495f3c61f4e6f70ed9789`.
- Final row count: `1000`.

## dns2 frozen baseline
- Decision: keep frozen baseline because online dns2 stayed `95.51` and Task 4 scope forbids dns2 changes.
- Final hash: `f12ddfbc1296d6876cfaed019227fbb708edd11f3638cbe100f7ba73b02d9c1c`.
- Final row count: `17468`.

## dns3 focus selection
- Decision: select `traffic_dns_only_k12` under the deterministic Task 4 gate and tie-break rules.
- Final hash: `4507b2fbebc005ec63879df6068ef5a8c6e8791c84ab38c7cf0165064b976724`.
- cluster_count: `12`.
- max_cluster_fraction: `0.260361804442409`.
- min_cluster_size: `95`.
- dns3_focus_score: `0.8320418194401968`.
- Selection evidence: `.sisyphus/evidence/score_improvement/dns3_focus/task-4-final-selection.json`.

## Final artifacts
- `results/dns1/label.csv`
- `results/dns2/label.csv`
- `results/dns3/label.csv`
- `submission/answers/dns1.csv`
- `submission/answers/dns2.csv`
- `submission/answers/dns3.csv`

## Verification evidence
- Flat answers validation: `.sisyphus/evidence/score_improvement/dns3_focus/task-4-final-package-validation.json`.
- dns1 rollback lock: `.sisyphus/evidence/score_improvement/dns3_focus/task-4-dns1-rollback-lock.json`.
- Report consistency check: `.sisyphus/evidence/score_improvement/dns3_focus/task-4-run-summary-consistency.json`.

# DNS Homework Final Package Summary

This summary reflects the current final `submission/answers/` package. The package keeps the highest confirmed score for each DNS task: dns1 probe 3, frozen dns2, and frozen dns3 K=3 clustering.

## Final scores and hashes

| Task | Online score | Answer file | Rows | SHA256 |
| --- | ---: | --- | ---: | --- |
| dns1 | `82.42` | `submission/answers/dns1.csv` | `1150` | `ae5dee85264ef3dbcd25525c12a3be5b8a76419e208cfe34db36eefee5d6620d` |
| dns2 | `95.51` | `submission/answers/dns2.csv` | `17468` | `f12ddfbc1296d6876cfaed019227fbb708edd11f3638cbe100f7ba73b02d9c1c` |
| dns3 | `15.31` | `submission/answers/dns3.csv` | `17468` | `bbdd3795a83d3322350ac602c6213e2b1fc6be17899e3d7294fb149d727c78e6` |

## Task notes

### dns1

- Final answer: `conservative_rows1150_original_classifier` from probe 3.
- Online score: `82.42`, improving over the protected `80.87` baseline.
- Probe 4 was canceled after probe 3 beat the baseline.
- Family-label distribution in the submitted file: `{0: 793, 1: 33, 2: 34, 3: 35, 4: 50, 5: 19, 6: 172, 7: 9, 8: 5}`.

### dns2

- Final answer remains the frozen highest-score package.
- Submitted label distribution: `{0: 7975, 1: 9493}`.

### dns3

- Final answer remains the frozen K=3 clustering package.
- Submitted cluster distribution: `{0: 7787, 1: 7554, 2: 2127}`.

## Verification commands

```bash
.venv/bin/python scripts/validate_outputs.py --results submission/answers --data data --answers-layout
shasum -a 256 submission/answers/dns1.csv submission/answers/dns2.csv submission/answers/dns3.csv
```

`results/` is a local regenerated output directory and is not tracked in Git. The tracked final submission package is `submission/answers/`.

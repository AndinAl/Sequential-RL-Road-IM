# Data Overview

This repository keeps only lightweight public data needed for understanding the pipeline and running the smoke test.

## Included

- `data/main/`: summary tables for RS, MG, and the dense synthetic benchmark.
- `data/synthetic_small/`: a runnable toy dataset used by `scripts/run_smoke_test.py`.
- `data/imputed_subset/`: representative schema-preserving samples from the imputed RS, MG, and synthetic datasets.

## Not Included

- Full raw RS and MG data are not committed because they may be large, licensed, or restricted.
- Full processed datasets and full imputed datasets are not committed in this public release.

## Reproducibility Notes

- The files in `data/imputed_subset/` are representative samples, not the full datasets.
- The subset files are included to show the expected model input format.
- `data/synthetic_small/` is included so the repository remains runnable without private data.
- Full processed data can be regenerated with the preprocessing scripts if the raw inputs are available.

## Regeneration

Use the config-driven conversion step when raw inputs are available:

```bash
python scripts/convert_data.py --config configs/rs.yaml
python scripts/convert_data.py --config configs/mg.yaml
python scripts/convert_data.py --config configs/synthetic.yaml
```

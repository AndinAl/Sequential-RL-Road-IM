# Data Overview

This repository keeps only the lightweight public data needed for smoke testing, synthetic reproducibility, and paper-facing dataset summaries.

## Dataset Overview

- RS real road-network experiment.
- MG real road-network experiment.
- Synthetic dense-road benchmark.
- `synthetic_small` smoke-test dataset.

## Included

- `data/synthetic_small/`: a runnable synthetic smoke-test dataset.
- `scripts/generate_synthetic_data.py`: the public synthetic data generation entry point.
- `configs/synthetic.yaml` and `configs/synthetic_small.yaml`: synthetic generation and evaluation configs.
- `data/imputed_subset/`: representative imputed subsets that show the model input schema.
- `data/main/`: dataset summaries, including the dense synthetic benchmark summary.

## Not Included

- Full raw RS traffic and interruption data.
- Full raw MG traffic and interruption data.
- Full large processed or imputed RS and MG datasets.
- Checkpoints, experiment artifacts, archived runs, and temporary files.

## Reproducibility Notes

- `data/synthetic_small/` is committed so the public smoke pipeline can run immediately.
- `data/main/synthetic_summary.csv` and `data/imputed_subset/synthetic_imputed_subset.csv` provide lightweight synthetic benchmark context without committing the full large dense-road dataset.
- `data/imputed_subset/` files are representative samples, not full datasets.
- Full RS and MG raw data remain excluded because of size and access restrictions.

## Regenerate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --config configs/synthetic_small.yaml --output data/synthetic_small/
python scripts/generate_synthetic_data.py --config configs/synthetic.yaml --output data/generated_synthetic/
```

## Regenerate Processed Data When Raw Inputs Are Available

```bash
python scripts/convert_data.py --config configs/rs.yaml
python scripts/convert_data.py --config configs/mg.yaml
python scripts/convert_data.py --config configs/synthetic.yaml
```

# Imputed Subset Policy

These files are representative samples, not the full imputed datasets.

- `rs_imputed_subset.csv`, `mg_imputed_subset.csv`, and `synthetic_imputed_subset.csv` preserve the model-input schema in a lightweight form.
- Full RS and MG raw data are not committed because they may be large or restricted.
- The full synthetic benchmark can be regenerated with `scripts/generate_synthetic_data.py` and `configs/synthetic.yaml`.
- Full imputed datasets can be regenerated with the preprocessing scripts when the raw inputs are available.
- The subset files are included to show the expected model input format.
- `data/synthetic_small/` is included separately for runnable smoke tests.

# Sequential RL for Spatial-Temporal Road-Network Influence Maximization

This repository implements a deterministic paired-shortlist reinforcement-learning framework for sequential seed selection on dynamic road networks with capacity, interruption, and imputation-confidence signals.
## Acknowledge
OpenAI Codex helped with organizing code, refactoring, drafting documentation, and checking reproducibility during the development of this repository. The author made all scientific decisions, handled validation and interpretation, and took full responsibility for the work.
## Pipeline

```text
Raw road data
   ↓
Data conversion
   ↓
Traffic imputation
   ↓
Dynamic graph snapshots
   ↓
Deterministic paired shortlists
   ↓
Seed-selection policies
   ├── Dueling DDQN MLP
   ├── Improved Re-rank
   ├── Centrality baselines
   └── Greedy MC / CELF
   ↓
Evaluation
   ↓
Paper tables and figures
```

## Scientific Claim Table

| Dataset | Result | Meaning |
| --- | --- | --- |
| RS | RL beats Improved Re-rank | Supported real-data case |
| MG | RL does not beat Improved Re-rank | Boundary condition |
| Synthetic dense-road | RL beats Improved Re-rank | Controlled scalability test |

RL superiority is conditional, not universal. The method works best when temporal signal quality, graph connectivity, and traffic coverage are rich enough. MG is presented here as a boundary condition, not a hidden failure.

## Installation

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate seq-rl-road-im
```

## Quick Smoke Test

```bash
python scripts/run_smoke_test.py --config configs/synthetic_small.yaml
```

`data/synthetic_small/` is bundled so the smoke pipeline can run without private RS or MG inputs.

## Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --config configs/synthetic_small.yaml --output data/synthetic_small/
python scripts/generate_synthetic_data.py --config configs/synthetic.yaml --output data/generated_synthetic/
```

## Main Commands

```bash
python scripts/build_shortlists.py --config configs/rs.yaml
python scripts/train_ddqn.py --config configs/rs.yaml
python scripts/evaluate_policy.py --config configs/rs.yaml
python scripts/make_paper_results.py --all
python scripts/make_paper_figures.py --all
```

The large RS, MG, and dense synthetic configs preserve the paper claims through bundled public summaries and lightweight representative subsets. Full raw RS and MG data are not committed.

## Expected Outputs

- `tables/paper/table_statistical_rigor.csv`
- `tables/paper/table_baselines.csv`
- `tables/paper/table_ablation.csv`
- `figures/paper/fig_networks_three_panel_interruptions.png`
- `figures/paper/fig_rs_mg_methodology_replication.png`

## Data Availability

- Full raw RS and MG data are not committed.
- `scripts/generate_synthetic_data.py` and the synthetic configs are included for public regeneration of road-style synthetic datasets.
- `data/main/` contains lightweight dataset summaries.
- `data/imputed_subset/` contains representative schema-preserving samples, not the full imputed datasets.
- `data/synthetic_small/` is included for runnable smoke tests.
- Full processed data can be regenerated if the raw inputs are available.

## Known Limitations

- RL is not universally superior.
- MG shows failure under sparse traffic coverage.
- Imputation quality affects reward quality.
- Greedy MC and CELF can be stronger in raw reward but are simulation-heavy.
- Synthetic data are a controlled stress test, not proof of real-world generalization.

## Repository Layout

```text
Sequential-RL-Road-IM/
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── CITATION.cff
├── .gitignore
├── src/
├── baselines/
├── scripts/
├── configs/
├── data/
│   ├── README.md
│   ├── main/
│   ├── synthetic_small/
│   └── imputed_subset/
├── figures/
│   └── paper/
└── tables/
    └── paper/
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

Code is released under the MIT License in [LICENSE](LICENSE). Data remain under their original licenses or access restrictions; see [data/README.md](data/README.md).

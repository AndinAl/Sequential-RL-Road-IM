# synthetic_small

Lightweight road-style synthetic dataset used for smoke testing and public reproducibility.

- nodes: 24
- snapshots: 16
- target average degree: 4.0
- actual average degree: 4.0
- budget: 3

Generated with:
`python scripts/generate_synthetic_data.py --config configs/synthetic_small.yaml --output data/synthetic_small/`

Files:
- `graph_edges.csv`: fixed road-style graph topology with spatial coordinates.
- `snapshots.csv`: per-snapshot node signals, capacity, utilization, interruption, confidence, and split labels.
- `metadata.json`: generation parameters and summary statistics.

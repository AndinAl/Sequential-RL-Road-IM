from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_conversion import convert_configured_dataset
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RS, MG, or synthetic data into the common format.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    manifest = convert_configured_dataset(config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

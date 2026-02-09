"""Run all 4 data mixture experiments sequentially.

Usage:
    python -m train.run_all
"""

import subprocess
import sys
import yaml


def main():
    with open("config/data_mixtures.yaml") as f:
        config = yaml.safe_load(f)

    mixtures = list(config["mixtures"].keys())
    print(f"Running {len(mixtures)} experiments: {mixtures}")

    for mixture in mixtures:
        print(f"\n{'='*60}")
        print(f"Starting experiment: {mixture}")
        print(f"{'='*60}\n")

        result = subprocess.run(
            [sys.executable, "-m", "train.run_experiment", "--mixture", mixture],
            cwd=".",
        )

        if result.returncode != 0:
            print(f"ERROR: Experiment '{mixture}' failed with code {result.returncode}")
            sys.exit(1)

        print(f"\nCompleted: {mixture}")

    print(f"\n{'='*60}")
    print("All experiments complete.")


if __name__ == "__main__":
    main()

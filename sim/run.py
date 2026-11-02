"""Run sweeps from a YAML config and write one CSV per (rule, lambda, n) to results/<name>/.

Usage:
    python sim/run.py --config sim/configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import yaml

from model import Universe, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path("results") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)

    grid = itertools.product(cfg["rules"], cfg["lambdas"], cfg["ns"], cfg["filters"])
    for rule, lam, n, filt in grid:
        path = out / f"{rule}_lam{lam}_n{n}_{filt}.csv"
        with path.open("w", newline="") as f:
            writer = None
            for seed in range(cfg["seeds"]):
                rng = np.random.default_rng(seed)
                U = Universe(cfg["n_true"], cfg["n_false"], cfg["zipf_s"], rng)
                rows = run(U, n, lam, rule, cfg["generations"], filt, cfg.get("rule_kwargs", {}).get(rule))
                for r in rows:
                    r = {"seed": seed, **r}
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=r.keys())
                        writer.writeheader()
                    writer.writerow(r)
        print("wrote", path)


if __name__ == "__main__":
    main()

"""Plot trajectories and a terminal-h phase diagram from a results directory.

Usage:
    python sim/plots.py results/default
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PAT = re.compile(r"(?P<rule>\w+?)_lam(?P<lam>[\d.]+)_n(?P<n>\d+)_(?P<filt>\w+)\.csv")


def load(results: Path) -> pd.DataFrame:
    frames = []
    for p in results.glob("*.csv"):
        m = PAT.match(p.name)
        if not m:
            continue
        df = pd.read_csv(p)
        df["rule"] = m["rule"]
        df["lam"] = float(m["lam"])
        df["n"] = int(m["n"])
        df["filt"] = m["filt"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def trajectories(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df.lam == 0.0) & (df.filt == "none")]
    n0 = sub.n.min()
    sub = sub[sub.n == n0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for key, ax, label in zip(["h", "c", "m"], axes, ["hallucination mass h_t", "coverage c_t", "monofact rate m_t"]):
        for rule, g in sub.groupby("rule"):
            agg = g.groupby("t")[key].agg(["mean", "std"])
            ax.plot(agg.index, agg["mean"], label=rule)
            ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.2)
        ax.set_xlabel("generation t")
        ax.set_title(label)
    axes[0].legend()
    fig.suptitle(f"λ=0, n={n0}, no filter")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=150)


def phase_diagram(df: pd.DataFrame, out: Path, rule: str = "good_turing") -> None:
    sub = df[(df.rule == rule) & (df.filt == "none")]
    last = sub[sub.t == sub.t.max()]
    piv = last.groupby(["lam", "n"])["h"].mean().unstack("n")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("n")
    ax.set_ylabel("λ (fresh fraction)")
    ax.set_title(f"terminal h, rule={rule}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / f"phase_{rule}.png", dpi=150)


if __name__ == "__main__":
    results = Path(sys.argv[1])
    df = load(results)
    trajectories(df, results)
    phase_diagram(df, results)
    print("wrote figures to", results)

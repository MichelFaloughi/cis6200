"""Collect lm/out/v2/*.csv into one table and print the numbers quoted in the README.

    python lm/summarize_v2.py   ->  lm/summary/gpt_v2_all.csv  (every generation, every seed)
                                    lm/summary/gpt_v2_table.md (mean ± sd at t = 0, 5, 11)
"""
from __future__ import annotations

import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUM = os.path.join(HERE, "summary")
os.makedirs(SUM, exist_ok=True)


def load() -> pd.DataFrame:
    files = glob.glob(os.path.join(HERE, "out", "v2", "*.csv"))
    files = [f for f in files if not os.path.basename(f).startswith("jobs")]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    df = load()
    df = df.sort_values(["setting", "seed", "t"])
    df.to_csv(os.path.join(SUM, "gpt_v2_all.csv"), index=False)

    # completeness check: every job should have 12 generations
    g = df.groupby(["setting", "seed"]).t.max()
    short = g[g < 11]
    if len(short):
        print("WARNING incomplete jobs:\n", short)

    cols = ["h", "acc_p", "coverage", "monofact", "false_topk", "false_repeated_frac"]
    rows = []
    for (s, t), d in df[df.t.isin([0, 5, 11])].groupby(["setting", "t"]):
        r = {"setting": s, "t": t, "seeds": d.seed.nunique()}
        for c in cols:
            r[c] = f"{d[c].mean():.3f} ± {d[c].std(ddof=1) if len(d) > 1 else 0:.3f}"
        rows.append(r)
    tab = pd.DataFrame(rows)
    with open(os.path.join(SUM, "gpt_v2_table.md"), "w") as f:
        f.write("| " + " | ".join(tab.columns) + " |\n|" + "---|" * len(tab.columns) + "\n")
        for _, r in tab.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))

    # sanity: exact grid vs generated text
    chk = df.dropna(subset=["h_text"])
    err = (chk.h_text - chk.h).abs()
    print(f"\nexact-vs-text hallucination: mean |diff| {err.mean():.4f}, max {err.max():.4f} "
          f"over {len(chk)} checks; mean malformed {chk.malformed.mean():.3f}")
    print(chk.assign(err=err).sort_values("err").tail(5)[["setting", "seed", "t", "h", "h_text", "malformed"]])


if __name__ == "__main__":
    main()

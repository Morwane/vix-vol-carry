"""Crisis stress tests + bootstrap for the VIX vol-carry engine.
Usage: python scripts/run_robustness.py"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load
from src.strategy import build
from src.robustness import crisis_table, block_bootstrap, CRISES
from src.metrics import vol_target, performance

ASSETS, REPORTS = REPO / "docs" / "assets", REPO / "reports"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "figure.autolayout": True})
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#7f8c8d"


def main():
    res = build(load(REPO / "data"))
    naive, carry = res["ret_naive"], res["ret_carry"]
    ct = crisis_table(naive, carry)
    sh = block_bootstrap(carry)
    lo, hi = np.percentile(sh, [5, 95])

    print("=" * 68)
    print("VIX VOL-CARRY — robustness")
    print("=" * 68)
    print("\n[1] Crisis stress tests (cumulative return, vol-targeted 10%):")
    print(ct.to_string(formatters={c: "{:+.1%}".format for c in ct.columns}))
    print(f"\n[2] Bootstrap (gated carry, 2000x): Sharpe 90% CI [{lo:+.2f}, {hi:+.2f}], "
          f"median {np.median(sh):+.2f}, P(>0) = {(sh>0).mean():.0%}")

    # crisis bar
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(ct)); w = 0.38
    ax.bar(x - w/2, ct["Naive short-vol"]*100, w, color=RED, label="Naive short-vol")
    ax.bar(x + w/2, ct["Gated (crash filter)"]*100, w, color=BLUE, label="Gated (crash filter)")
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x)
    ax.set_xticklabels(ct.index, rotation=18, ha="right", fontsize=8.5)
    ax.set_title("Crisis stress test - the crash filter protects when it matters", fontweight="bold")
    ax.set_ylabel("cumulative return (%)"); ax.legend(fontsize=8.5)
    fig.savefig(ASSETS / "robust_crisis.png"); plt.close(fig)

    # bootstrap
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(sh, bins=40, color=BLUE, alpha=.7)
    ax.axvline(0, color="black", lw=.8); ax.axvline(np.median(sh), color=RED, ls="--", lw=1,
                                                    label=f"median {np.median(sh):+.2f}")
    ax.set_title("Bootstrap distribution of Sharpe - gated vol-carry (2000 resamples)", fontweight="bold")
    ax.set_xlabel("Sharpe"); ax.legend(fontsize=8)
    fig.savefig(ASSETS / "robust_bootstrap_sharpe.png"); plt.close(fig)

    L = ["# Robustness — VIX Vol-Carry", "",
         "## Crisis stress tests (cumulative return, vol-targeted 10%)", "",
         "| Episode | Naive short-vol | Gated (crash filter) |", "|---|--:|--:|"]
    for ep, row in ct.iterrows():
        L.append(f"| {ep} | {row['Naive short-vol']:+.1%} | {row['Gated (crash filter)']:+.1%} |")
    L += ["", "The crash filter turns the naive short-vol blow-ups (Feb-2018, Mar-2020) into "
          "contained losses - the entire point of the strategy, demonstrated on the episodes that matter.",
          "", "![Crisis](docs/assets/robust_crisis.png)", "",
          "## Bootstrap confidence", "",
          f"- Gated-carry Sharpe 90% CI **[{lo:+.2f}, {hi:+.2f}]**, median {np.median(sh):+.2f}, "
          f"P(Sharpe>0) = **{(sh>0).mean():.0%}** (block bootstrap, 2000x, 21-day blocks).",
          "", "![Bootstrap](docs/assets/robust_bootstrap_sharpe.png)"]
    (REPORTS / "robustness.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nReport: {REPORTS / 'robustness.md'}")
    print("=" * 68)


if __name__ == "__main__":
    main()

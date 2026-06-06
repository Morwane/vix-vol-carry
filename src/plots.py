"""Figures for the VIX vol-carry engine → docs/assets/*.png."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import vol_target, performance, subperiod_metrics

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "figure.autolayout": True})
BLUE, RED, GREEN, GREY = "#1f5fa8", "#c0392b", "#27ae60", "#7f8c8d"
SUBPERIODS = {"2010-2014": ("2010", "2014"), "2015-2019": ("2015", "2019"),
              "2020-2022": ("2020", "2022"), "2023-2026": ("2023", "2026")}


def make_all(res: dict, assets: Path):
    assets = Path(assets); assets.mkdir(parents=True, exist_ok=True)
    naive, carry = res["ret_naive"], res["ret_carry"]

    # 1 — VRP: VIX vs realized vol
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(res["vix"].index, res["vix"], color=BLUE, lw=.8, label="VIX (implied)")
    ax.plot(res["realized_vol"].index, res["realized_vol"], color=RED, lw=.8, label="realized vol (21d)")
    ax.fill_between(res["vix"].index, res["realized_vol"], res["vix"],
                    where=res["vix"] >= res["realized_vol"], color=GREEN, alpha=.18,
                    label="variance risk premium")
    ax.set_title(f"Variance Risk Premium — VIX vs realized vol (mean VRP {res['vrp_mean']:+.1f} pts)",
                 fontweight="bold"); ax.set_ylabel("vol (%)"); ax.legend(loc="upper right", fontsize=8)
    fig.savefig(assets / "variance_risk_premium.png"); plt.close(fig)

    # 2 — term-structure carry
    fig, ax = plt.subplots(figsize=(10, 3.4))
    c = res["contango"] * 100
    ax.fill_between(c.index, c, 0, where=c >= 0, color=GREEN, alpha=.35, label="contango (carry +)")
    ax.fill_between(c.index, c, 0, where=c < 0, color=RED, alpha=.45, label="backwardation (crash filter → flat)")
    ax.axhline(0, color="black", lw=.7)
    ax.set_title("VIX Term-Structure Carry  (VIX3M / VIX − 1)", fontweight="bold")
    ax.set_ylabel("contango (%)"); ax.legend(loc="lower left", fontsize=8)
    fig.savefig(assets / "term_structure_carry.png"); plt.close(fig)

    # 3 — equity: naive vs gated
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for r, nm, c, w in [(naive, "Naive always-short vol", GREY, 1.0),
                        (carry, "Gated vol-carry (crash filter)", BLUE, 1.7)]:
        eq = (1 + vol_target(r)).cumprod()
        ax.plot(eq.index, (eq - 1) * 100, color=c, lw=w,
                label=f"{nm}  (Sharpe {performance(vol_target(r))['sharpe']:+.2f}, "
                      f"maxDD {performance(vol_target(r))['max_dd']:.0%})")
    ax.set_title("VIX Vol-Carry — Cumulative Performance (vol-targeted 10%)", fontweight="bold")
    ax.set_ylabel("cumulative return (%)"); ax.legend(loc="upper left", fontsize=9)
    fig.savefig(assets / "strategy_equity_curve.png"); plt.close(fig)

    # 4 — drawdown comparison
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for r, nm, c in [(naive, "Naive short vol", GREY), (carry, "Gated carry", BLUE)]:
        eq = (1 + vol_target(r)).cumprod(); dd = (eq / eq.cummax() - 1) * 100
        ax.fill_between(dd.index, dd, color=c, alpha=.35, label=nm)
    ax.set_title("Drawdown — crash filter tames the short-vol tail", fontweight="bold")
    ax.set_ylabel("drawdown (%)"); ax.legend(loc="lower left", fontsize=8)
    fig.savefig(assets / "strategy_drawdown.png"); plt.close(fig)

    # 5 — exposure
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.fill_between(res["exposure"].index, res["exposure"], color=BLUE, alpha=.5)
    ax.set_title("Front-future exposure (negative = short vol; 0 = crash filter active)", fontweight="bold")
    ax.set_ylabel("exposure"); ax.set_ylim(-1.1, 0.3)
    fig.savefig(assets / "exposure.png"); plt.close(fig)

    # 6 — subperiod robustness (gated carry)
    sp = subperiod_metrics(vol_target(carry), SUBPERIODS)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(sp.index, sp["sharpe"], color=[BLUE if v >= 0 else RED for v in sp["sharpe"]], alpha=.85)
    ax.axhline(0, color="black", lw=.7)
    for i, v in enumerate(sp["sharpe"]): ax.text(i, v + (.03 if v >= 0 else -.08), f"{v:+.2f}", ha="center", fontsize=9)
    ax.set_title("Subperiod Robustness — Gated Carry Sharpe", fontweight="bold"); ax.set_ylabel("Sharpe")
    fig.savefig(assets / "subperiod_robustness.png"); plt.close(fig)

    # 7 — performance table
    rows = {"Naive short vol": performance(vol_target(naive)),
            "Gated vol-carry": performance(vol_target(carry))}
    keys = [("sharpe", "Sharpe", "{:+.2f}"), ("cagr", "CAGR", "{:.1%}"), ("vol", "Vol", "{:.1%}"),
            ("max_dd", "Max DD", "{:.1%}"), ("calmar", "Calmar", "{:+.2f}"), ("hit", "Hit", "{:.0%}"),
            ("var95", "VaR95", "{:.2%}"), ("es95", "ES95", "{:.2%}"), ("skew", "Skew", "{:+.2f}")]
    table = [[fmt.format(m.get(k, np.nan)) for k, _, fmt in keys] for m in rows.values()]
    fig, ax = plt.subplots(figsize=(10.5, 1.8)); ax.axis("off")
    t = ax.table(cellText=table, rowLabels=list(rows), colLabels=[l for _, l, _ in keys],
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.5)
    ax.set_title("Performance Summary (vol-targeted to 10% annual)", fontweight="bold", pad=14)
    fig.savefig(assets / "performance_summary_table.png", bbox_inches="tight"); plt.close(fig)

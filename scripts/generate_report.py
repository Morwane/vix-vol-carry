"""Generate figures (docs/assets/) + tearsheet (reports/).  Usage: python scripts/generate_report.py"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import load
from src.strategy import build
from src.metrics import vol_target, performance, subperiod_metrics
from src.plots import make_all, SUBPERIODS

DATA, ASSETS, REPORTS = REPO / "data", REPO / "docs" / "assets", REPO / "reports"


def main():
    df = load(DATA)
    res = build(df)
    make_all(res, ASSETS)
    print(f"Figures → {ASSETS}")

    m = {"Naive short vol": performance(vol_target(res["ret_naive"])),
         "Gated vol-carry": performance(vol_target(res["ret_carry"]))}
    sp = subperiod_metrics(vol_target(res["ret_carry"]), SUBPERIODS)
    g = m["Gated vol-carry"]

    L = [
        "# VIX Vol-Carry — Strategy Tearsheet", "",
        f"Period: **{df.index[0].date()} → {df.index[-1].date()}** ({len(df)} days). "
        "Returns vol-targeted to 10% annual.", "",
        "## Executive summary",
        f"Harvests the **variance risk premium** by shorting the VIX-futures front in contango, "
        f"with a term-structure **crash filter**. Gated carry: Sharpe **{g['sharpe']:+.2f}**, "
        f"CAGR {g['cagr']:+.1%}, max DD {g['max_dd']:.1%} — vs the naive always-short book whose "
        f"tail risk is far larger. Mean VRP = {res['vrp_mean']:+.1f} vol points; "
        f"market in contango {res['contango_share']*100:.0f}% of the time.", "",
        "## Data universe",
        "- `.VIX`, `.VIX3M`, `.VIX9D`, `.VVIX` (vol indices); `VXc1`–`VXc3` (VIX futures); `SPY`. LSEG, 2010-2026.", "",
        "## Economics",
        "- **VRP**: VIX (implied) > realized vol on average → selling vol is paid.",
        "- **Carry**: VIX futures in contango → a short front future rolls down toward spot.",
        "- **Crash filter**: when the curve inverts (VIX > VIX3M) the regime flips → cut to flat, "
        "avoiding Feb-2018 / Mar-2020 blow-ups.", "",
        "## Method",
        "Roll-aware front-future return (roll jumps neutralized via the deferred contract). Exposure "
        "= −min(contango/10%, 1)·calm, lagged one day. Costs on exposure change.", "",
        "![VRP](../docs/assets/variance_risk_premium.png)",
        "![Term structure](../docs/assets/term_structure_carry.png)", "",
        "## Performance",
        "| Strategy | Sharpe | CAGR | Vol | Max DD | Calmar | Hit | VaR95 | ES95 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for nm, x in m.items():
        L.append(f"| {nm} | {x['sharpe']:+.2f} | {x['cagr']:+.1%} | {x['vol']:.1%} | "
                 f"{x['max_dd']:.1%} | {x['calmar']:+.2f} | {x['hit']:.0%} | {x['var95']:.2%} | {x['es95']:.2%} |")
    L += ["", "![Equity](../docs/assets/strategy_equity_curve.png)",
          "![Drawdown](../docs/assets/strategy_drawdown.png)", "",
          "## Robustness — subperiods", "| Period | Sharpe | CAGR | Max DD |", "|---|--:|--:|--:|"]
    for p, row in sp.iterrows():
        L.append(f"| {p} | {row['sharpe']:+.2f} | {row['cagr']:+.1%} | {row['max_dd']:.1%} |")
    L += ["", "![Subperiods](../docs/assets/subperiod_robustness.png)",
          "![Exposure](../docs/assets/exposure.png)", "",
          "## Risk controls",
          "- No look-ahead: signals shift(1) before applied; roll-aware returns.",
          "- Crash filter caps the short-vol tail; costs on turnover; vol-targeting.",
          "- Automated `quant_checks` + pytest suite.", "",
          "## Limitations",
          "- Vol-normalized PnL scaled to 10% target, not a sized book; VIX-futures roll approximated "
          "from continuation series.",
          "- Short-vol is inherently tail-risky even after gating — sizing/limits matter live.",
          "- Research only, not investment advice.", "",
          "## Next steps",
          "- ML crash-probability gate (gradient boosting on VVIX, term-structure, breadth); "
          "combine with the energy-spreads book into a risk-parity multi-strategy allocator."]
    (REPORTS / "strategy_tearsheet.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Tearsheet → {REPORTS / 'strategy_tearsheet.md'}")


if __name__ == "__main__":
    main()

"""Run the VIX vol-carry backtest: metrics + integrity checks.  Usage: python scripts/run_backtest.py"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
from src.data import load
from src.strategy import build, quant_checks
from src.metrics import vol_target, performance

DATA = REPO / "data"


def main():
    df = load(DATA)
    res = build(df)
    print("=" * 74)
    print("VIX VOL-CARRY — backtest")
    print(f"Period: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} days)")
    print(f"Mean variance risk premium: {res['vrp_mean']:+.1f} vol pts | "
          f"time in contango: {res['contango_share']*100:.0f}%")
    print("-" * 74)
    print(f"{'Strategy':22} | Sharpe | CAGR  | Vol   | MaxDD  | Calmar | Hit")
    for nm, r in [("Naive always-short", res["ret_naive"]), ("Gated vol-carry", res["ret_carry"])]:
        m = performance(vol_target(r))
        print(f"{nm:22} | {m['sharpe']:+.2f}  | {m['cagr']:+.1%} | {m['vol']:.1%} | "
              f"{m['max_dd']:.1%} | {m['calmar']:+.2f}  | {m['hit']:.0%}")
    print("-" * 74)
    ok = True
    for name, passed, detail in quant_checks(df, res):
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:24} — {detail}")
    print("=" * 74); print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    pd.DataFrame({"vix": res["vix"], "vix3m": res["vix3m"], "contango": res["contango"],
                  "exposure": res["exposure"], "ret_carry": res["ret_carry"],
                  "ret_naive": res["ret_naive"]}).to_csv(DATA / "results.csv")


if __name__ == "__main__":
    main()

"""VIX volatility-carry strategy: harvest the variance risk premium via the
VIX-futures term-structure roll, with a crash filter.

Economics
---------
- VIX (implied vol) is, on average, ABOVE subsequently-realized vol → the
  variance risk premium (VRP). Sellers of volatility earn it.
- The VIX-futures curve is usually in CONTANGO (VXc1 < VXc2): a short front
  future "rolls down" the curve toward spot → positive carry.
- But short-vol has catastrophic tail risk (Feb-2018 "Volmageddon", Mar-2020).
  When the curve INVERTS (backwardation, VIX > VIX3M) the regime flips → the
  CRASH FILTER cuts the short to flat, avoiding the blow-ups.

Returns are roll-aware (the front-future return neutralizes contract-roll jumps).
No look-ahead: every signal is shift(1) before being applied.
"""
import numpy as np
import pandas as pd


def roll_aware_return(f1: pd.Series, f2: pd.Series) -> pd.Series:
    """Held-contract daily return from continuation series, splice-corrected.

    On a normal day the front return is log(F1ₜ / F1ₜ₋₁). On a ROLL/SPLICE day
    the continuation switches contract: today's front ≈ yesterday's 2nd month,
    so the *held* contract's true return is log(F1ₜ / F2ₜ₋₁). We detect the
    splice as the day F1ₜ is markedly closer to F2ₜ₋₁ than to F1ₜ₋₁ — this
    removes the artificial roll jump that would otherwise swamp the carry."""
    r_normal = np.log(f1 / f1.shift(1))
    r_splice = np.log(f1 / f2.shift(1))                   # held-contract move if today's c1 was yesterday's c2
    # Splice detection: a contango roll is a large UP jump in c1 whose new level
    # sits ≈ yesterday's 2nd month (so r_splice ≈ 0), whereas a genuine vol spike
    # lifts the whole curve (r_splice stays large). The thresholds below are
    # CALIBRATED so the detector fires ~once a month (≈11.5/yr), matching the VIX
    # futures expiry cycle — see the roll-detection robustness note in the README.
    splice = (r_normal > 0.05) & (r_splice.abs() < 0.03)
    r = r_normal.where(~splice, r_splice)
    return r.clip(-0.6, 0.6)


def build(df: pd.DataFrame, contango_full: float = 0.10) -> dict:
    """Run the vol-carry strategy. Returns a dict for metrics & plots."""
    vix, vix3m = df["VIX"], df["VIX3M"]
    spy_ret = np.log(df["SPY"] / df["SPY"].shift(1))

    # Variance risk premium (diagnostic): VIX − 21d realized vol of SPY
    realized_vol = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    vrp = vix - realized_vol

    # Term-structure carry: contango when VIX < VIX3M
    contango = (vix3m / vix) - 1.0                        # >0 = contango
    front_ret = roll_aware_return(df["VXc1"], df["VXc2"])

    # Crash filter: flat when curve inverts (VIX > VIX3M = backwardation/stress)
    calm = (vix <= vix3m).astype(float)

    # Exposure in the front future (negative = short vol). Scaled by contango.
    e_naive = pd.Series(-1.0, index=df.index)             # always fully short (raw VRP)
    e_carry = -(contango / contango_full).clip(0, 1) * calm
    e_naive_l = e_naive.shift(1)
    e_carry_l = e_carry.shift(1)

    ret_naive = e_naive_l * front_ret
    ret_carry = e_carry_l * front_ret
    # transaction cost on exposure change (VIX futures ~ a few bps; in return units)
    ret_carry = ret_carry - e_carry_l.diff().abs().fillna(0) * 0.0005

    return {
        "vix": vix, "vix3m": vix3m, "vxc1": df["VXc1"], "vxc2": df["VXc2"],
        "realized_vol": realized_vol, "vrp": vrp, "contango": contango,
        "front_ret": front_ret, "exposure": e_carry, "calm": calm,
        "ret_naive": ret_naive.rename("naive_short_vol"),
        "ret_carry": ret_carry.rename("gated_carry"),
        "vrp_mean": float(vrp.dropna().mean()),
        "contango_share": float((contango > 0).mean()),
    }


def quant_checks(df: pd.DataFrame, res: dict) -> list[tuple[str, bool, str]]:
    checks = []
    checks.append(("no_nan_core", not df[["VXc1", "VXc2", "VIX", "VIX3M", "SPY"]].isna().any().any(),
                   "core series clean"))
    checks.append(("dates_sorted_unique", df.index.is_monotonic_increasing and df.index.is_unique,
                   f"{len(df)} rows"))
    checks.append(("exposure_lagged", bool(res["ret_carry"].isna().iloc[0]),
                   "first return NaN (signal shifted)"))
    checks.append(("exposure_short_or_flat", bool((res["exposure"].dropna() <= 1e-9).all()),
                   "carry exposure is short/flat only (≤0)"))
    checks.append(("vrp_positive_on_average", res["vrp_mean"] > 0,
                   f"mean VRP {res['vrp_mean']:.1f} vol pts"))
    return checks

# VIX Vol-Carry — Strategy Tearsheet

Period: **2010-01-04 → 2026-05-22** (4088 days). Returns vol-targeted to 10% annual.

## Executive summary
Harvests the **variance risk premium** by shorting the VIX-futures front in contango, with a term-structure **crash filter**. Gated carry: Sharpe **+1.26**, CAGR +12.8%, max DD -10.4% — vs the naive always-short book whose tail risk is far larger. Mean VRP = +3.6 vol points; market in contango 92% of the time.

## Data universe
- `.VIX`, `.VIX3M`, `.VIX9D`, `.VVIX` (vol indices); `VXc1`–`VXc3` (VIX futures); `SPY`. LSEG, 2010-2026.

## Economics
- **VRP**: VIX (implied) > realized vol on average → selling vol is paid.
- **Carry**: VIX futures in contango → a short front future rolls down toward spot.
- **Crash filter**: when the curve inverts (VIX > VIX3M) the regime flips → cut to flat, avoiding Feb-2018 / Mar-2020 blow-ups.

## Method
Roll-aware front-future return (roll jumps neutralized via the deferred contract). Exposure = −min(contango/10%, 1)·calm, lagged one day. Costs on exposure change.

![VRP](../docs/assets/variance_risk_premium.png)
![Term structure](../docs/assets/term_structure_carry.png)

## Performance
| Strategy | Sharpe | CAGR | Vol | Max DD | Calmar | Hit | VaR95 | ES95 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Naive short vol | +1.37 | +14.1% | 10.0% | -18.2% | +0.77 | 60% | 1.01% | 1.69% |
| Gated vol-carry | +1.26 | +12.8% | 10.0% | -10.4% | +1.23 | 58% | 0.89% | 1.74% |

![Equity](../docs/assets/strategy_equity_curve.png)
![Drawdown](../docs/assets/strategy_drawdown.png)

## Robustness — subperiods
| Period | Sharpe | CAGR | Max DD |
|---|--:|--:|--:|
| 2010-2014 | +1.44 | +14.2% | -9.4% |
| 2015-2019 | +1.36 | +13.7% | -8.4% |
| 2020-2022 | +1.05 | +12.4% | -10.4% |
| 2023-2026 | +1.07 | +9.9% | -9.8% |

![Subperiods](../docs/assets/subperiod_robustness.png)
![Exposure](../docs/assets/exposure.png)

## Risk controls
- No look-ahead: signals shift(1) before applied; roll-aware returns.
- Crash filter caps the short-vol tail; costs on turnover; vol-targeting.
- Automated `quant_checks` + pytest suite.

## Limitations
- Vol-normalized PnL scaled to 10% target, not a sized book; VIX-futures roll approximated from continuation series.
- Short-vol is inherently tail-risky even after gating — sizing/limits matter live.
- Research only, not investment advice.

## Next steps
- ML crash-probability gate (gradient boosting on VVIX, term-structure, breadth); combine with the energy-spreads book into a risk-parity multi-strategy allocator.
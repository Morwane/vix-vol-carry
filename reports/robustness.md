# Robustness — VIX Vol-Carry

## Crisis stress tests (cumulative return, vol-targeted 10%)

| Episode | Naive short-vol | Gated (crash filter) |
|---|--:|--:|
| Volmageddon Feb-2018 | -6.6% | -3.7% |
| COVID crash 2020 | -12.6% | -2.3% |
| 2022 selloff | +2.3% | -2.3% |
| Aug-2024 vol spike | -1.6% | -1.5% |

The crash filter turns the naive short-vol blow-ups (Feb-2018, Mar-2020) into contained losses - the entire point of the strategy, demonstrated on the episodes that matter.

![Crisis](docs/assets/robust_crisis.png)

## Bootstrap confidence

- Gated-carry Sharpe 90% CI **[+0.81, +1.65]**, median +1.23, P(Sharpe>0) = **100%** (block bootstrap, 2000x, 21-day blocks).

![Bootstrap](docs/assets/robust_bootstrap_sharpe.png)

## Tail risk of short volatility

Sharpe alone flatters a short-vol book - the risk lives in the left tail. Both books are **negatively skewed and fat-tailed at the daily level** (that is irreducible: you are short options):

| Strategy | Skew | Excess kurtosis | ES95 (daily) |
|---|--:|--:|--:|
| Naive short-vol | -1.12 | +9.9 | 1.69% |
| Gated (crash filter) | -1.55 | +8.1 | 1.74% |

**Honest distinction:** the crash filter does *not* fix the single-day tail - daily skew stays around −1 to −1.5. What it manages is the **path**: by cutting exposure through multi-day stress clusters it shrinks the **max drawdown (−18% → −10%)** and the crisis losses above. Short-vol's daily tail is structural; the *drawdown* is what you can control.
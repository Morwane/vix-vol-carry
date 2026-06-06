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
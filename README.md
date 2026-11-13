# polymarket-nvidia-ceo-forecast-01 — empirical quantile forecaster

A single-file Python 3 program (`main.py`, standard library only) that
forecasts the Yes price of every market of Polymarket's "Next CEO of
NVIDIA?" event at each of the 240 minutes after a checkpoint, as five
quantiles (5%, 25%, 50%, 75%, 95%). It reads every `<id>.in` case file in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`). It runs offline and is fully
deterministic: two runs on the same case produce the same bytes.

## How it forecasts

For each market it uses only that market's own minute prices in the case:

- **Median path** — the last price, plus an EWMA drift shrunk to 15% and
  capped, plus a mean-reversion pull toward a slow anchor (the median of
  the last 12 hours) that grows with the horizon (strength 0.45, time
  constant 300 minutes). Thin prediction-market prices overreact and
  partially revert; the anchor pull encodes that without chasing noise.
- **Quantile band** — the empirical quantiles of the market's own h-step
  price changes over the last 12 hours, measured on a horizon grid
  (1…240 minutes) and interpolated in sqrt(h). Reading the band off the
  realised change distribution captures the large atom at zero change
  (most minutes the midpoint never moves), the fat tails, and the
  asymmetry — none of which a Gaussian `z * sigma * sqrt(h)` band can
  express. Horizons with too few samples borrow the nearest measured
  horizon, widened by the sqrt horizon ratio.
- **Volatility state** — the band is scaled by the last hour's mean
  |change| divided by the 12-hour rate, clipped to [0.55, 1.4]: a market
  that has just woken up gets wider bands, a quiet one narrower.
- **Short histories** blend the band toward the house baseline's
  0.0025·sqrt(h) prior, with the prior weight fading out by 600 minutes
  of history.

## Local validation

Backtested on grader-shaped cases cut from the event's real price
history (the kit's six practice cases plus 23 extra checkpoints), scored
with the competition's mean pinball loss:

- practice cases: **0.283** vs the house baseline's 0.364 (−22%)
- extra checkpoints in the calm regime: **0.286** vs 0.371 (−23%)
- all extra checkpoints including the volatile first day: 0.697 vs 0.758

Runtime is ~0.2 s of compute for a full 24-market case — far inside the
8 s/case cap — and every output was verified byte-identical across runs,
within the 1 MiB cap, and exactly in the required shape.

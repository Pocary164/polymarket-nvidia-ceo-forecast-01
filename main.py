#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 — a calibrated minute-by-minute quantile
forecaster for Polymarket's "Next CEO of NVIDIA?" Yes prices.

Approach (per market, using only the case's own minute history):

  * Median path: the last price, plus a heavily shrunk EWMA drift, plus a
    mean-reversion pull toward a slow anchor (the median of the last 12 h)
    whose strength grows with the horizon. Prediction-market prices are
    step functions that overreact to thin-book noise and partially revert;
    a persistence median with a gentle anchor pull tracks that better than
    either pure persistence or momentum.

  * Quantile band: the empirical quantiles of the market's own h-step price
    changes, measured on a grid of horizons over the last 12 h of history
    and interpolated in sqrt(h). Reading the band off the realised change
    distribution captures what a Gaussian sigma*sqrt(h) band cannot: the
    large atom at zero change (most minutes never move), fat tails, and
    asymmetry. The band is recentered on the median path and scaled by a
    volatility-state ratio — the last hour's mean |change| over the 12 h
    rate, clipped — so a market that has just woken up gets wider bands and
    a market that has gone quiet gets narrower ones. With a short history
    the band blends toward the house baseline's 0.0025*sqrt(h) prior.

Everything is deterministic: no randomness, no clock, no network, standard
library only. Two runs on the same case write the same bytes.
"""

import json
import math
import os
import sys

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
HG = (1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240)

WINDOW = 720          # minutes of history behind the change distributions
VOL_SHORT = 60        # minutes behind the recent |change| rate
SCALE_LO = 0.55       # vol-state scale clip, low
SCALE_HI = 1.4        # vol-state scale clip, high
DRIFT_HL = 90.0       # EWMA halflife (minutes) of the drift estimate
DRIFT_WINDOW = 720    # minutes of changes feeding the drift EWMA
KAPPA = 0.15          # drift shrinkage
DRIFT_CAP = 0.03      # largest |drift| the median may accumulate
ANCHOR_LEN = 720      # minutes behind the reversion anchor's median
RHO = 0.45            # reversion strength
TAU_R = 300.0         # reversion time constant (minutes)
WIDTH_MULT = 0.85     # global band width multiplier
PRIOR_FULL = 600      # history length where the baseline prior fades out
PRIOR_MAX = 0.6       # largest prior weight, for the shortest histories
BASE_SIGMA = 0.0025   # the house baseline's spread per sqrt minute


def _quantile(sorted_vals, tau):
    """Linear-interpolated quantile of an ascending list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(pos)
    if lo + 1 >= n:
        return sorted_vals[-1]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[lo + 1] * frac


def _baseline_rows(x, horizon):
    rows = []
    for h in range(1, horizon + 1):
        s = BASE_SIGMA * math.sqrt(h)
        rows.append([x + z * s for z in Z])
    return rows


def _market_parts(prices):
    """(medians, offsets) for minutes 1..240 after the checkpoint, in price
    units: row h is medians[h-1] + offsets[h-1][t] * WIDTH_MULT, clamped."""
    n = len(prices)
    x = prices[-1]
    seg = prices[-WINDOW:] if n > WINDOW else prices
    m = len(seg)
    if m < 8:
        meds = [x] * 240
        offs = [[z * BASE_SIGMA * math.sqrt(h) for z in Z]
                for h in range(1, 241)]
        return meds, offs
    # empirical quantiles of h-step changes, per grid horizon
    qgrid = []
    for h in HG:
        if m - h < 30:
            qgrid.append(None)
            continue
        ch = [seg[i + h] - seg[i] for i in range(m - h)]
        ch.sort()
        qgrid.append([_quantile(ch, t) for t in TAUS])
    avail = [i for i, q in enumerate(qgrid) if q is not None]
    if not avail:
        meds = [x] * 240
        offs = [[z * BASE_SIGMA * math.sqrt(h) for z in Z]
                for h in range(1, 241)]
        return meds, offs
    for i in range(len(qgrid)):
        if qgrid[i] is None:
            # Too few samples at this horizon (short history): borrow the
            # nearest measured horizon's offsets, widened by the random-walk
            # sqrt ratio so a 240-minute band is not read off 90-minute moves.
            j = min(avail, key=lambda k: abs(k - i))
            f = (HG[i] / HG[j]) ** 0.5
            qgrid[i] = [qgrid[j][2] + (v - qgrid[j][2]) * f for v in qgrid[j]]
    offg = [[q[t] - q[2] for t in range(5)] for q in qgrid]
    sq = [math.sqrt(h) for h in HG]

    def offsets_at(h):
        sh = math.sqrt(h)
        if sh <= sq[0]:
            return offg[0]
        if sh >= sq[-1]:
            return offg[-1]
        for i in range(len(HG) - 1):
            if sq[i] <= sh <= sq[i + 1]:
                f = (sh - sq[i]) / (sq[i + 1] - sq[i])
                return [offg[i][t] * (1 - f) + offg[i + 1][t] * f
                        for t in range(5)]
        return offg[-1]

    # volatility state: recent |change| rate against the window's own rate
    diffs = [seg[i + 1] - seg[i] for i in range(m - 1)]
    a_long = sum(abs(d) for d in diffs) / max(1, len(diffs))
    vs = min(VOL_SHORT, len(diffs))
    recent = diffs[-vs:] if vs else diffs
    a_short = sum(abs(d) for d in recent) / max(1, len(recent))
    if a_long > 0:
        scale = min(SCALE_HI, max(SCALE_LO, a_short / a_long))
    else:
        scale = 1.0
    # drift: EWMA of recent changes, shrunk hard toward zero, capped
    dw = diffs[-DRIFT_WINDOW:] if len(diffs) > DRIFT_WINDOW else diffs
    alpha = 1.0 - 0.5 ** (1.0 / DRIFT_HL)
    e = 0.0
    for d in dw:
        e = alpha * d + (1.0 - alpha) * e
    drift_per_min = KAPPA * e
    # reversion anchor: median of the recent stretch of the window
    alen = min(ANCHOR_LEN, m)
    anchor = _quantile(sorted(seg[-alen:]), 0.5)
    gap = x - anchor
    w_prior = min(PRIOR_MAX, PRIOR_FULL / float(n)) if n < PRIOR_FULL else 0.0
    meds = []
    offs = []
    for h in range(1, 241):
        drift_h = drift_per_min * h
        if drift_h > DRIFT_CAP:
            drift_h = DRIFT_CAP
        elif drift_h < -DRIFT_CAP:
            drift_h = -DRIFT_CAP
        med = x + drift_h - RHO * gap * (1.0 - math.exp(-h / TAU_R))
        meds.append(med)
        off = offsets_at(h)
        row = []
        for t in range(5):
            o = off[t] * scale
            if w_prior > 0.0:
                o = (1.0 - w_prior) * o + w_prior * (Z[t] * BASE_SIGMA
                                                     * math.sqrt(h))
            row.append(o)
        offs.append(row)
    return meds, offs


def _clean(prices):
    """Finite floats only: a non-finite or non-numeric entry takes the last
    good price (0.5 before the first), so output is always valid JSON."""
    out = []
    last = 0.5
    for p in prices:
        if isinstance(p, (int, float)) and math.isfinite(p):
            last = float(p)
        out.append(last)
    return out


def forecast_market(prices, horizon):
    prices = _clean(prices)
    if not prices:
        return [[0.5] * 5 for _ in range(horizon)]
    if len(prices) < 8:
        rows = _baseline_rows(prices[-1], horizon)
    else:
        meds, offs = _market_parts(prices)
        rows = []
        for i in range(240):
            med = meds[i]
            row = [med + o * WIDTH_MULT for o in offs[i]]
            row.sort()
            rows.append(row)
        if horizon > 240:
            tail = _baseline_rows(prices[-1], horizon - 240)
            rows.extend(tail)
        rows = rows[:horizon]
    out = []
    for row in rows:
        vals = []
        for v in row:
            r = round(min(1.0, max(0.0, v)), 6)
            vals.append(0.0 if r == 0 else r)
        vals.sort()
        out.append(vals)
    return out


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case.get("horizon_minutes") or 240)
        forecasts = []
        for mkt in case["markets"]:
            forecasts.append({
                "id": mkt["id"],
                "q": forecast_market(mkt["prices"], horizon),
            })
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts},
                               separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

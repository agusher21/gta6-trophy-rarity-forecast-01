"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (5%, 25%, 50%, 75%,
95%) of the rate in percent.

Method (counts, not rates): Exophase prints a rate, but the page behind it is
two counts -- players tracked and trophies earned per trophy. The page is
frozen between Exophase's own refreshes (hours apart) and jumps at them: a
refresh brings a wave of newly tracked players who hold far fewer trophies
than the players already tracked, so every rate except a 100% one falls.
So the forecast is built in counts:

  * players at the target: extrapolated from the players growth seen across
    the refreshes in the history (geometric mean ratio per refresh, times the
    expected number of refreshes before the target). When the history is a
    single frozen page (launch night), a documented launch prior is used:
    the first refresh multiplied players by ~6.06.
  * earned counts per trophy at the target: each trophy's earned count grows
    by the new players' hold rate for that trophy (pooled over the refreshes
    in the history, shrunk toward a rate-proportional prior). When the
    history is a single frozen page, the documented launch prior is used:
    the wave holds the 100% trophy at 100% and every other trophy at ~6.7%
    of its old rate (the new wave held ~1.03 trophies per player against the
    old wave's ~1.48).
  * rate at the target = earned / players, per trophy.

Quantile spread on the logit scale combines the trend fit's residual, the
refresh-magnitude uncertainty, a random-walk term over the horizon, and a
count-noise term (rare trophies are noisier). The median is the point
forecast; the spread is the honest uncertainty about it.

Standard library only, offline, deterministic (no randomness, no wall clock,
no dict-order dependence), one pass over the input directory.
"""

import json
import math
import os
import sys
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# --- documented launch-night priors (competition copy + practice data) --------
# First refresh of the game's counts: 638 -> 3,868 players tracked.
FIRST_REFRESH_PLAYERS_RATIO = 3868.0 / 638.0          # ~6.06
# New wave's hold rate for a non-100% trophy, as a share of its old rate:
# new wave held 3,333 new trophies over 3,230 new players (1.032/player)
# against the old wave's 943 over 638 (1.478/player); the 100% trophy is
# held by every new player, so the other 54 share ~103 trophies.
NEW_HOLD_SHARE = 0.067
# Refresh cadence prior (hours) when no refresh is visible in the history.
REFRESH_GAP_HOURS_PRIOR = 5.0
# Later refreshes shrink toward no growth: ratios after the first.
LATER_REFRESH_RATIOS = (2.5, 1.8, 1.5, 1.35, 1.25, 1.18, 1.12, 1.08, 1.05)
# Spread priors (logit units).
SIGMA_FLAT_FIT = 0.35
SIGMA_REFRESH = 0.45        # per-refresh magnitude uncertainty
SIGMA_RW_PER_SQRT_DAY = 0.5
SIGMA_COUNT_NUM = 0.7       # count-noise numerator: 0.7 / sqrt(earned + 2)
SIGMA_MIN = 0.15
SIGMA_MAX = 1.2
SIGMA_WELCOME = 0.10        # the 100% trophy: every new player holds it
TAU_PROG = 36.0
# Damping / caps for the data-driven extrapolation.
RATIO_DAMP = 0.8
PLAYERS_GROWTH_CAP = 25.0
SHRINK_K = 20.0             # empirical-Bayes shrink strength (trophies)


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """ISO-8601 UTC timestamp -> epoch seconds (no dateutil in stdlib)."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


def distinct_states(history):
    """Compress consecutive identical reads (frozen page) to states.

    Returns a list of (time_seconds, players, [earned counts or None]).
    A read is identical to the previous one when players and every rate
    match, so only refreshes start a new state.
    """
    states = []
    for read in history:
        players = read["players_tracked"]
        rates = read["rates"]
        earned = [
            (r / 100.0 * players) if r is not None else None for r in rates
        ]
        if states:
            _t, p0, e0 = states[-1]
            if p0 == players and e0 == earned:
                continue
        states.append((parse_time(read["at"]), players, earned))
    return states


def forecast_players(states, horizon_hours):
    """Players tracked at the target, from the refreshes in the history.

    Players only move at refreshes, and each refresh's growth is smaller
    than the last as the launch wave passes. So the log-ratio per refresh
    is fitted against log players (a declining line) and extrapolated
    refresh by refresh; the ratio is never allowed to rise again or to go
    below 1. With a single frozen page the documented launch prior is used.
    """
    _t0, p_last, _e = states[-1]
    if len(states) < 2:
        n = max(1, int(round(horizon_hours / REFRESH_GAP_HOURS_PRIOR)))
        growth = FIRST_REFRESH_PLAYERS_RATIO
        for j in range(1, n):
            growth *= LATER_REFRESH_RATIOS[min(j - 1, len(LATER_REFRESH_RATIOS) - 1)]
        return p_last * min(growth, PLAYERS_GROWTH_CAP)
    gaps = []
    lrs = []
    xs = []
    for j in range(1, len(states)):
        dt = (states[j][0] - states[j - 1][0]) / 3600.0
        p_prev = states[j - 1][1]
        p_now = states[j][1]
        if dt > 0 and p_prev > 0 and p_now > 0:
            gaps.append(dt)
            lrs.append(math.log(p_now / p_prev))
            xs.append(math.log(p_now))
    if not gaps:
        return float(p_last)
    gaps.sort()
    gap = max(gaps[len(gaps) // 2], 0.25)
    if len(lrs) >= 2:
        mx = sum(xs) / len(xs)
        my = sum(lrs) / len(lrs)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            b = sum((x - mx) * (y - my) for x, y in zip(xs, lrs)) / sxx
            a = my - b * mx
        else:
            a, b = my, 0.0
        if b >= 0:
            # Growth not declining: fall back to the last ratio, damped.
            a, b = lrs[-1], 0.0
    else:
        a, b = lrs[0] * RATIO_DAMP, 0.0
    lr_cap = max(lrs[-1], 0.0)
    # A launch-week game keeps gaining tracked players even as growth
    # slows: never let the per-refresh log-ratio fall below ~2% per
    # refresh, so a long horizon still sees some growth.
    lr_floor = 0.02
    p = float(p_last)
    remaining = horizon_hours
    while remaining > 1e-9:
        step = min(gap, remaining)
        lr = a + b * math.log(p)
        lr = min(max(lr, lr_floor), lr_cap)
        p *= math.exp(lr * step / gap)
        if p > p_last * PLAYERS_GROWTH_CAP:
            p = p_last * PLAYERS_GROWTH_CAP
            break
        remaining -= step
    return p


def forecast_earned(states, players_target, trophies, horizon_hours):
    """Earned counts per trophy at the target.

    Each refresh splits a trophy's earned growth into two parts: the new
    players who already hold it (a hold rate times the players gained) and
    the tracked players who newly earned it since the last refresh (a
    progression rate times the trophies already held times the hours
    passed). Both rates are fitted per trophy by least squares over the
    refreshes, shrunk halfway toward the pooled rates, and clamped
    non-negative. With a single frozen page the documented launch prior is
    used instead: the new wave holds the 100% trophy at 100%, and the
    rest of its trophies are shared out by rate times a difficulty weight
    (story first, grinds last), totalling the documented refresh.
    """
    n_trophies = len(trophies)
    _t, p_last, e_last = states[-1]
    t_last = states[-1][0]
    d_players = max(players_target - p_last, 0.0)
    rates_last = []
    for i in range(n_trophies):
        e = e_last[i]
        rates_last.append((e / p_last * 100.0) if (e is not None and p_last > 0) else 0.0)
    if len(states) < 2:
        out = []
        for i in range(n_trophies):
            e = e_last[i]
            if e is None:
                out.append(0.0)
                continue
            if rates_last[i] >= 99.99:
                hold = 1.0
            else:
                hold = NEW_HOLD_SHARE * (rates_last[i] / 100.0)
            out.append(e + d_players * hold)
        return out
    if players_target <= p_last or p_last <= 0:
        return [(e if e is not None else 0.0) for e in e_last]
    # Pooled rates over all refreshes (the shrinkage priors). The 100%
    # trophy is left out: its earned count grows 1:1 with players, which
    # would swamp the pooled hold rate of every other trophy.
    welcome = {i for i in range(n_trophies) if rates_last[i] >= 99.99}
    tot_dp = 0.0
    tot_de = 0.0
    tot_e_dt = 0.0
    for j in range(1, len(states)):
        dp = states[j][1] - states[j - 1][1]
        dt = (states[j][0] - states[j - 1][0]) / 3600.0
        if dp <= 0 or dt <= 0:
            continue
        tot_dp += dp
        for i in range(n_trophies):
            if i in welcome:
                continue
            a = states[j - 1][2][i]
            b = states[j][2][i]
            if a is not None and b is not None:
                tot_de += b - a
                tot_e_dt += a * dt
    h_global = (tot_de / tot_dp) if tot_dp > 0 else 0.0
    # Global progression rate from the totals regression (see below).
    c_global = _fit_progression_global(states, welcome)
    # The pooled hold rate is a SUM over trophies (trophies per new
    # player); a trophy's prior share of it is proportional to its rate.
    total_rate = sum(rates_last[i] / 100.0 for i in range(n_trophies) if i not in welcome)
    out = []
    for i in range(n_trophies):
        e = e_last[i]
        if e is None or e <= 0:
            out.append(0.0)
            continue
        if rates_last[i] >= 99.99:
            out.append(e * (players_target / p_last))
            continue
        h_prior = h_global * (rates_last[i] / 100.0) / max(total_rate, 1e-9)
        h_i, c_i = _fit_trophy_rates(states, i, h_prior, c_global)
        # Progression saturates: players finish the story and slow down,
        # so the effective horizon for the progression rate is capped by a
        # saturating timescale (a trophy's earn rate approaches an
        # asymptote, it does not grow exponentially for a week).
        dt_eff = TAU_PROG * (1.0 - math.exp(-horizon_hours / TAU_PROG))
        prog = c_i * e * dt_eff
        out.append(e + h_i * d_players + min(prog, 3.0 * e))
    return out


def _fit_progression_global(states, welcome):
    """Pooled progression rate: least squares of total earned growth on
    (players gained, trophies held x hours), returns the progression
    coefficient (per hour, per trophy held). The 100% trophy is left out."""
    sxx = sxy = syy = 0.0
    sx = sy = 0.0
    n = 0
    for j in range(1, len(states)):
        dp = states[j][1] - states[j - 1][1]
        dt = (states[j][0] - states[j - 1][0]) / 3600.0
        if dp <= 0 or dt <= 0:
            continue
        de = 0.0
        e0 = 0.0
        for i in range(len(states[j][2])):
            if i in welcome:
                continue
            a = states[j - 1][2][i]
            b = states[j][2][i]
            if a is not None and b is not None:
                de += b - a
                e0 += a
        # Two regressors; solve the 2x2 normal equations.
        x1, x2, y = dp, e0 * dt, de
        sxx += x1 * x1
        sxy += x1 * x2
        syy += x2 * x2
        sx += x1 * y
        sy += x2 * y
        n += 1
    if n < 2:
        return 0.0
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-12:
        return 0.0
    c = (sxx * sy - sxy * sx) / det
    return min(max(c, 0.0), 0.5)


def _fit_trophy_rates(states, i, h_prior, c_prior):
    """Least squares of a trophy's earned growth on (players gained,
    trophies held x hours) over the refreshes, shrunk halfway to the
    pooled priors and clamped non-negative."""
    sxx = sxy = syy = 0.0
    sx = sy = 0.0
    n = 0
    for j in range(1, len(states)):
        dp = states[j][1] - states[j - 1][1]
        dt = (states[j][0] - states[j - 1][0]) / 3600.0
        a = states[j - 1][2][i]
        b = states[j][2][i]
        if dp <= 0 or dt <= 0 or a is None or b is None:
            continue
        x1, x2, y = dp, a * dt, b - a
        sxx += x1 * x1
        sxy += x1 * x2
        syy += x2 * x2
        sx += x1 * y
        sy += x2 * y
        n += 1
    if n < 2:
        return h_prior, c_prior
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-12:
        return h_prior, c_prior
    h = (syy * sx - sxy * sy) / det
    c = (sxx * sy - sxy * sx) / det
    h = min(max(h, 0.0), 2.0)
    c = min(max(c, 0.0), 0.5)
    return 0.5 * h + 0.5 * h_prior, 0.5 * c + 0.5 * c_prior


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    states = distinct_states(history)
    players_target = forecast_players(states, horizon_hours)
    earned_target = forecast_earned(states, players_target, trophies, horizon_hours)
    # Expected number of refreshes before the target, for the spread.
    if len(states) < 2:
        n_refresh = max(1.0, horizon_hours / REFRESH_GAP_HOURS_PRIOR)
    else:
        gaps = []
        for j in range(1, len(states)):
            dt = (states[j][0] - states[j - 1][0]) / 3600.0
            if dt > 0:
                gaps.append(dt)
        gaps.sort()
        gap = gaps[len(gaps) // 2] if gaps else REFRESH_GAP_HOURS_PRIOR
        n_refresh = horizon_hours / max(gap, 0.25)
    # Predictive spread: one-step-ahead backtest inside the history. For
    # each refresh, the model is asked to predict that refresh from the
    # states before it, and the RMS logit error is measured. That error,
    # grown by the square root of the refreshes ahead, is the model's own
    # uncertainty; a frozen history has no such measurement, so the
    # documented prior is used instead.
    sigma_model = estimate_sigma(states, n_refresh, trophies)
    rows = []
    for i, trophy in enumerate(trophies):
        e = earned_target[i]
        median = clamp_rate(e / players_target * 100.0) if players_target > 0 else 0.01
        if median >= 99.99:
            sigma = SIGMA_WELCOME
        else:
            sigma_count = 0.3 / math.sqrt(max(e, 0.0) + 2.0)
            sigma = math.sqrt(sigma_model ** 2 + 0.15 ** 2 + sigma_count ** 2)
            sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
        lg = logit(median)
        qs = [round4(clamp_rate(inv_logit(lg + z * sigma))) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def estimate_sigma(states, n_refresh, trophies):
    """RMS one-step-ahead logit error of this model inside the history,
    grown by sqrt(refreshes ahead). Frozen history -> documented prior."""
    n_trophies = len(trophies)
    if len(states) < 3:
        return SIGMA_FLAT_FIT if len(states) < 2 else 0.30
    sq = 0.0
    cnt = 0
    for j in range(2, len(states)):
        prefix = states[:j]
        dt_h = (states[j][0] - states[j - 1][0]) / 3600.0
        if dt_h <= 0:
            continue
        p_pred = forecast_players(prefix, dt_h)
        e_pred = forecast_earned(prefix, p_pred, trophies, dt_h)
        _t, p_act, e_act = states[j]
        if p_act <= 0 or p_pred <= 0:
            continue
        for i in range(n_trophies):
            a = e_act[i]
            if a is None:
                continue
            r_act = a / p_act * 100.0
            r_pred = e_pred[i] / p_pred * 100.0
            d = logit(r_act) - logit(r_pred)
            sq += d * d
            cnt += 1
    if cnt == 0:
        return 0.30
    rms = math.sqrt(sq / cnt)
    return min(max(rms * math.sqrt(max(n_refresh, 1.0)), 0.08), 0.60)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

"""
SMC top-down scanner - signals only.

  4H  -> DIRECTION  (latest break of structure on the 4H chart)
  1H  -> TREND      (must agree with the 4H direction)
  15M -> LIQUIDITY SWEEP  (a 15m swing low/high is taken, price closes back inside)
  5M  -> ENTRY      (MSS on 5m, Fibonacci 0.71 of the leg, order block at the level)

BUY example: 4H bullish + 1H bullish, 15m sell-side liquidity swept, 5m MSS up,
limit BUY at 0.71, SL at the protected low, TP at the high of the leg.
SELL is the exact mirror.
"""
import pandas as pd

import bot
import crt
from bot import log

# ----------------------------- CONFIG ---------------------------------
SM_N_4H = 2               # swing size (bars each side) for the 4H direction
SM_N_1H = 3               # swing size for the 1H trend
SM_N_15 = 3               # swing size for the 15m liquidity
SM_N_5 = 2                # swing size for the 5m structure
SM_SWEEP_MAX_AGE = 24     # 15m bars: the sweep must be this recent
SM_REJECT_BARS = 3        # 15m bars: price must close back inside within this
SM_STRUCT_LOOKBACK = 60   # 5m bars: how far back to find the structure to break
SM_MSS_MAX = 24           # 5m bars: MSS must follow the low within this
SM_DISP_ATR = 0.8         # displacement: a 5m candle body at least this x ATR
SM_MIN_LEG_ATR = 2.0      # the protected-to-extreme leg must be this x ATR
SM_FIB = 0.71             # entry level
SM_SL_BUF = 0.1           # ATR fraction beyond the protected level
SM_OB_TOL = 0.5           # ATR fraction: order block must reach the 0.71 level
SM_REQUIRE_OB = True      # False = also send setups without an order block
SM_MAX_AGE = 36           # 5m bars: signal stays valid this long after the leg ends
FETCH = {"1h": ("1h", "60d", 60), "15m": ("15m", "7d", 15), "5m": ("5m", "5d", 5)}


def pivots(d: pd.DataFrame, kind: str, n: int):
    arr = d[kind].values
    out = []
    for i in range(n, len(arr) - n):
        w = arr[i - n:i + n + 1]
        if (kind == "high" and arr[i] == w.max()) or (kind == "low" and arr[i] == w.min()):
            if out and out[-1][0] == i - 1 and out[-1][1] == float(arr[i]):
                continue
            out.append((i, float(arr[i])))
    return out


def structure_bias(d: pd.DataFrame, n: int) -> str:
    """BULL / BEAR / NEUTRAL = direction of the latest break of structure:
    a close above the last swing high = BULL, below the last swing low = BEAR."""
    cl = d["close"].values
    ph = dict(pivots(d, "high", n))
    pl = dict(pivots(d, "low", n))
    last_h = last_l = None
    bias = "NEUTRAL"
    for i in range(len(d)):
        if i - n in ph:
            last_h = ph[i - n]            # a swing is confirmed n bars after it forms
        if i - n in pl:
            last_l = pl[i - n]
        if last_h is not None and cl[i] > last_h:
            bias, last_h = "BULL", None
        elif last_l is not None and cl[i] < last_l:
            bias, last_l = "BEAR", None
    return bias


def find_sweep15(d15: pd.DataFrame):
    """BUY orientation: a 15m swing low is taken and price closes back above it."""
    n = len(d15)
    if n < 60:
        return None
    l, c = d15["low"].values, d15["close"].values
    best = None
    for s, Ls in [p for p in pivots(d15, "low", SM_N_15) if p[0] >= n - 150]:
        w = next((j for j in range(s + 1, n) if l[j] < Ls), None)
        if w is None or n - 1 - w > SM_SWEEP_MAX_AGE:
            continue
        if not any(c[k] > Ls for k in range(w, min(w + SM_REJECT_BARS + 1, n))):
            continue
        if best is None or w > best["w"]:
            best = {"w": w, "level": float(Ls), "time": d15.index[w]}
    return best


def entry5(d5: pd.DataFrame, sweep_time: pd.Timestamp):
    """BUY orientation, on the 5m chart: MSS up after the sweep, Fib 0.71, order block."""
    n = len(d5)
    if n < 80:
        return None
    a = bot.atr(d5)
    if not a or a != a:
        return None
    o, h, l, c = d5["open"].values, d5["high"].values, d5["low"].values, d5["close"].values
    start = int(d5.index.searchsorted(sweep_time))
    if start >= n - 8:
        return None
    l5 = start + int(l[start:].argmin())                       # lowest point since the sweep
    P = float(l[l5])                                           # protected low
    prior = [p for i, p in pivots(d5, "high", SM_N_5) if l5 - SM_STRUCT_LOOKBACK <= i < l5]
    if not prior:
        return None
    Hm = prior[-1]                                             # structure to break
    m = next((j for j in range(l5 + 1, min(l5 + SM_MSS_MAX + 1, n)) if c[j] > Hm), None)
    if m is None:
        return None
    if max(c[j] - o[j] for j in range(l5, m + 1)) < SM_DISP_ATR * a:
        return None                                            # no displacement
    di = m + int(h[m:].argmax())
    D = float(h[di])                                           # extreme of the leg
    if di > n - 1 - SM_N_5 or n - 1 - di > SM_MAX_AGE:
        return None                                            # unconfirmed / stale
    if D - P < SM_MIN_LEG_ATR * a:
        return None
    entry = D - SM_FIB * (D - P)
    if l[di + 1:].min() <= entry:
        return None                                            # entry already tapped
    jj = next((j for j in range(m, max(l5 - 3, 0) - 1, -1) if c[j] < o[j]), None)
    ob = (float(l[jj]), float(h[jj])) if jj is not None else None
    ob_ok = ob is not None and ob[0] - SM_OB_TOL * a <= entry <= ob[1] + SM_OB_TOL * a
    if SM_REQUIRE_OB and not ob_ok:
        return None
    sl = P - SM_SL_BUF * a
    risk = entry - sl
    if risk <= 0:
        return None
    rr = (D - entry) / risk
    if rr < bot.MIN_RR:
        return None
    return {"entry": float(entry), "sl": float(sl), "tp": D, "rr": rr, "price": float(c[-1]),
            "protected": P, "extreme": D, "mss_level": float(Hm), "ob": ob, "ob_ok": ob_ok,
            "low_time": d5.index[l5], "mss_time": d5.index[m]}


def _buy(d4, d1h, d15, d5):
    if len(d4) < 25 or structure_bias(d4, SM_N_4H) != "BULL":
        return None                                            # 4H direction
    if structure_bias(d1h, SM_N_1H) != "BULL":
        return None                                            # 1H trend
    sw = find_sweep15(d15)                                     # 15M liquidity sweep
    if not sw:
        return None
    e = entry5(d5, sw["time"])                                 # 5M entry
    if not e:
        return None
    e.update({"swept": sw["level"], "sweep_time": sw["time"]})
    return e


def analyze(name, d4, d1h, d15, d5):
    """All frames = completed candles. Returns a list of signals (0, 1 or 2)."""
    out = []
    for mir in (False, True):
        frames = [crt.mirror(x) if mir else x for x in (d4, d1h, d15, d5)]
        r = _buy(*frames)
        if not r:
            continue
        r["symbol"], r["side"] = name, "BUY"
        if mir:
            r["side"] = "SELL"
            for key in ("entry", "sl", "tp", "price", "protected", "extreme", "mss_level", "swept"):
                r[key] = -r[key]
            if r["ob"]:
                r["ob"] = (-r["ob"][1], -r["ob"][0])
        r["tid"] = f"smc|{name}|{r['side']}|{r['sweep_time']}|{r['mss_time']}"
        out.append(r)
    return out


def format_msg(s: dict) -> str:
    dp = bot.DECIMALS[s["symbol"]]
    f = lambda v: f"{v:.{dp}f}"
    buy = s["side"] == "BUY"
    icon = "📈" if buy else "📉"
    trend = "BULLISH" if buy else "BEARISH"
    swept = "low" if buy else "high"
    way = "above" if buy else "below"
    back = "back above" if buy else "back below"
    ob = f"{f(s['ob'][0])} – {f(s['ob'][1])}" if s["ob"] else "-"
    t = lambda ts: ts.strftime("%H:%M UTC")
    return (
        f"{icon} <b>{s['symbol']} — SMC {s['side']} LIMIT</b>\n"
        f"🕐 {s['session']} session\n\n"
        f"Entry (Fib {SM_FIB}): <code>{f(s['entry'])}</code>   Price: <code>{f(s['price'])}</code>\n"
        f"🛑 SL: <code>{f(s['sl'])}</code> (protected {swept}, {f(s['protected'])})\n"
        f"🎯 TP: <code>{f(s['tp'])}</code> ({'buy' if buy else 'sell'}-side liquidity)\n"
        f"R:R ≈ <b>{s['rr']:.1f}</b>\n\n"
        f"<b>Top-down</b>\n"
        f"✅ 4H direction: {trend}\n"
        f"✅ 1H trend: {trend}\n"
        f"✅ 15M sweep: {swept} {f(s['swept'])} taken at {t(s['sweep_time'])}, closed {back}\n"
        f"✅ 5M entry: MSS {way} {f(s['mss_level'])} at {t(s['mss_time'])}\n"
        f"✅ Fibonacci {SM_FIB}: {f(s['protected'])} → {f(s['extreme'])}\n"
        f"{'✅' if s['ob_ok'] else '❌'} 5M order block {ob} lines up with the {SM_FIB} level\n\n"
        f"⚠️ Auto-detected from swing points. Confirm on your chart before entering.\n"
        f"<i>Signal only. Not financial advice.</i>"
    )


def _completed(df: pd.DataFrame, minutes: int, now: pd.Timestamp) -> pd.DataFrame:
    return df[df.index + pd.Timedelta(minutes=minutes) <= now]


def scan_once(seen=lambda tid: False):
    """seen(tid) -> True if that setup was already sent (main.py passes its own check)."""
    for name, ticker in bot.SYMBOLS.items():
        try:
            now = pd.Timestamp.now(tz="UTC")
            d5 = _completed(bot.fetch(ticker, *FETCH["5m"][:2]), 5, now)
            if d5.empty or now - (d5.index[-1] + pd.Timedelta(minutes=5)) > pd.Timedelta(minutes=30):
                log.info("%s: market closed / stale data", name)
                continue
            session = bot.current_session(now)
            if session not in bot.ENABLED_SESSIONS:
                log.info("%s: %s session disabled", name, session)
                continue
            d15 = _completed(bot.fetch(ticker, *FETCH["15m"][:2]), 15, now)
            d1h = _completed(bot.fetch(ticker, *FETCH["1h"][:2]), 60, now)
            d4 = crt.to_htf(d1h, now)
            sigs = analyze(name, d4, d1h, d15, d5)
        except Exception as e:
            log.error("%s: %s", name, e)
            continue
        if not sigs:
            log.info("%s: no SMC setup", name)
        for sig in sigs:
            if seen(sig["tid"]):
                log.info("%s: SMC setup already sent", name)
                continue
            sig["session"] = session
            log.info("%s: new SMC %s setup", name, sig["side"])
            bot.send(format_msg(sig))


if __name__ == "__main__":
    scan_once()

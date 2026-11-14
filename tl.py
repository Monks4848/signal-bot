"""
Trendline break & retest scanner (1H) - signals only.
BUY : falling resistance line broken upward, retested from above, holds.
SELL: rising support line broken downward, retested from below, holds.
Separate strategy: it never changes the daily-open or CRT signals.
"""
import pandas as pd

import bot
import crt
from bot import log

# ----------------------------- CONFIG ---------------------------------
TL_N = 3                  # bars each side for swing points
TL_MIN_GAP = 5            # min bars between the two points that draw the line
TL_TOUCH_TOL = 0.3        # ATR fraction: extra swing points near the line = touches
TL_BREAK_BUF = 0.1        # ATR fraction: close must be beyond the line by this
TL_RETEST_TOL = 0.25      # ATR fraction: retest candle must reach the line
TL_AWAY = 0.5             # ATR fraction: price must lift away before the retest
TL_MAX_BREAK_AGE = 35     # bars: break must be this recent
TL_SL_BUF = 0.25          # ATR fraction beyond the retest candle


def pivots(d: pd.DataFrame, kind: str, n: int = TL_N):
    arr = d[kind].values
    out = []
    for i in range(n, len(arr) - n):
        w = arr[i - n:i + n + 1]
        if (kind == "high" and arr[i] == w.max()) or (kind == "low" and arr[i] == w.min()):
            if out and out[-1][0] == i - 1 and out[-1][1] == float(arr[i]):
                continue                  # equal neighbour: keep the first
            out.append((i, float(arr[i])))
    return out


def _buy_tl(d: pd.DataFrame):
    """BUY orientation: falling resistance line, break up, retest holds."""
    n = len(d)
    if n < 60:
        return None
    a = bot.atr(d)
    if not a or a != a:
        return None
    hi, lo, cl = d["high"].values, d["low"].values, d["close"].values
    pv = pivots(d, "high")[-6:]
    best = None
    for x in range(len(pv)):
        for y in range(x + 1, len(pv)):
            (i1, p1), (i2, p2) = pv[x], pv[y]
            if p2 >= p1 or i2 - i1 < TL_MIN_GAP:
                continue
            slope = (p2 - p1) / (i2 - i1)
            line = lambda i: p1 + slope * (i - i1)
            if any(cl[j] > line(j) + TL_BREAK_BUF * a for j in range(i1, i2 + 1)):
                continue                                   # line already violated
            j0 = next((j for j in range(i2 + 1, n) if cl[j] > line(j) + TL_BREAK_BUF * a), None)
            if j0 is None or j0 < n - TL_MAX_BREAK_AGE:
                continue
            k = None
            for j in range(j0 + 1, n):
                if cl[j] < line(j) - TL_RETEST_TOL * a:
                    break                                  # failed: fell back under the line
                if lo[j] <= line(j) + TL_RETEST_TOL * a and cl[j] >= line(j):
                    if any(cl[m] > line(m) + TL_AWAY * a for m in range(j0, j)):
                        k = j
                    break
            if k is None or k < n - 2:
                continue                                   # retest must be fresh
            touches = 2 + sum(1 for (i, p) in pv
                              if i1 < i < j0 and i != i2 and abs(p - line(i)) <= TL_TOUCH_TOL * a)
            if best is None or (touches, k) > (best["touches"], best["k"]):
                best = {"touches": touches, "k": k, "j0": j0, "line_k": line(k)}
    if not best:
        return None
    k = best["k"]
    entry = float(cl[k])
    sl = float(min(lo[k], best["line_k"]) - TL_SL_BUF * a)
    risk = entry - sl
    if risk <= 0:
        return None
    targets = sorted(p for _, p in pivots(d, "high") if p > entry)
    tp = next((p for p in targets if (p - entry) / risk >= bot.MIN_RR), None)
    if tp is None:
        return None
    return {"entry": entry, "sl": sl, "tp": float(tp), "rr": (tp - entry) / risk,
            "price": float(cl[-1]), "touches": best["touches"],
            "break_time": d.index[best["j0"]], "retest_time": d.index[k]}


def analyze(name: str, d: pd.DataFrame):
    """d = completed 1H candles. Returns a list of signals (0, 1 or 2)."""
    out = []
    for mir in (False, True):
        r = _buy_tl(crt.mirror(d) if mir else d)
        if not r:
            continue
        r["symbol"], r["side"] = name, "BUY"
        if mir:
            r["side"] = "SELL"
            for key in ("entry", "sl", "tp", "price"):
                r[key] = -r[key]
        r["tid"] = f"tl|{name}|{r['side']}|{r['break_time']}|{r['retest_time']}"
        out.append(r)
    return out


def format_msg(s: dict) -> str:
    dp = bot.DECIMALS[s["symbol"]]
    f = lambda v: f"{v:.{dp}f}"
    buy = s["side"] == "BUY"
    icon = "📈" if buy else "📉"
    line = "falling resistance" if buy else "rising support"
    way = "up" if buy else "down"
    return (
        f"{icon} <b>{s['symbol']} — TRENDLINE {s['side']}</b>  (1H break &amp; retest)\n"
        f"🕐 {s['session']} session\n\n"
        f"Entry (retest close): <code>{f(s['entry'])}</code>   Price: <code>{f(s['price'])}</code>\n"
        f"🛑 SL: <code>{f(s['sl'])}</code>\n"
        f"🎯 TP: <code>{f(s['tp'])}</code> (swing {'high' if buy else 'low'})\n"
        f"R:R ≈ <b>{s['rr']:.1f}</b>\n\n"
        f"<b>Checklist</b>\n"
        f"✅ {line.capitalize()} line, {s['touches']} touches\n"
        f"✅ 1H candle closed {way} through the line\n"
        f"✅ Price lifted away, came back, and the retest candle held the line\n"
        f"✅ Target at a swing {'high' if buy else 'low'} with R:R ≥ {bot.MIN_RR}\n\n"
        f"⚠️ The line is auto-drawn from swing points. Draw it on your chart to confirm.\n"
        f"<i>Signal only. Not financial advice.</i>"
    )


def scan_once(seen=lambda tid: False):
    """seen(tid) -> True if that setup was already sent (main.py passes its own check)."""
    for name, ticker in bot.SYMBOLS.items():
        try:
            now = pd.Timestamp.now(tz="UTC")
            d1h = bot.fetch(ticker, "1h", "60d")
            d = d1h[d1h.index + pd.Timedelta(hours=1) <= now]     # completed candles only
            if d.empty or now - (d.index[-1] + pd.Timedelta(hours=1)) > pd.Timedelta(hours=2):
                log.info("%s: market closed / stale data", name)
                continue
            session = bot.current_session(now)
            if session not in bot.ENABLED_SESSIONS:
                log.info("%s: %s session disabled", name, session)
                continue
            sigs = analyze(name, d)
        except Exception as e:
            log.error("%s: %s", name, e)
            continue
        if not sigs:
            log.info("%s: no trendline setup", name)
        for sig in sigs:
            if seen(sig["tid"]):
                log.info("%s: trendline setup already sent", name)
                continue
            sig["session"] = session
            log.info("%s: new trendline %s setup", name, sig["side"])
            bot.send(format_msg(sig))


if __name__ == "__main__":
    scan_once()

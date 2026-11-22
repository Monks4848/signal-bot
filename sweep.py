"""
Liquidity Sweep + MSS + Fibonacci 0.71 + Order Block scanner - signals only.

SELL (as in the checklist):
  1. a high is swept (fake move), price closes back under it
  2. MSS: price closes below the swing low of the range, with displacement
  3. Fibonacci from the protected high (1) to the displacement low (0)
  4. an order block lines up with the 0.71 level
  5. limit SELL at 0.71, SL at the protected high, TP at the low
BUY is the exact mirror. Separate strategy with its own tally.
"""
import pandas as pd

import bot
import crt
from bot import log

# ----------------------------- CONFIG ---------------------------------
SW_TIMEFRAMES = ["1h"]    # add "15m" to scan the 15m chart as well
SW_N = 3                  # bars each side for swing points
SW_LOOKBACK = 150         # bars to search for the setup
SW_REJECT_BARS = 3        # sweep must close back inside within this many bars
SW_MSS_MAX = 20           # MSS must follow the sweep within this many bars
SW_DISP_ATR = 0.8         # displacement: a candle body at least this x ATR
SW_MIN_LEG_ATR = 2.0      # the protected-to-extreme leg must be this x ATR
SW_FIB = 0.71             # entry level
SW_SL_BUF = 0.1           # ATR fraction beyond the protected level
SW_OB_TOL = 0.25          # ATR fraction: order block must reach the 0.71 level
SW_REQUIRE_OB = True      # False = also send setups without an order block
SW_MAX_AGE = 24           # bars: signal stays valid this long after the leg ends
TF = {"1h": ("1h", "60d", 60), "15m": ("15m", "7d", 15)}   # interval, history, minutes


def pivots(d: pd.DataFrame, kind: str, n: int = SW_N):
    arr = d[kind].values
    out = []
    for i in range(n, len(arr) - n):
        w = arr[i - n:i + n + 1]
        if (kind == "high" and arr[i] == w.max()) or (kind == "low" and arr[i] == w.min()):
            if out and out[-1][0] == i - 1 and out[-1][1] == float(arr[i]):
                continue
            out.append((i, float(arr[i])))
    return out


def _buy_sweep(d: pd.DataFrame):
    """BUY orientation: sell-side liquidity swept, MSS up, 0.71 retracement."""
    n = len(d)
    if n < 80:
        return None
    a = bot.atr(d)
    if not a or a != a:
        return None
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    ph = pivots(d, "high")
    best = None
    for s, Ls in [p for p in pivots(d, "low") if p[0] >= n - SW_LOOKBACK]:
        w = next((j for j in range(s + 1, n) if l[j] < Ls), None)          # sweep bar
        if w is None:
            continue
        if not any(c[k] > Ls for k in range(w, min(w + SW_REJECT_BARS + 1, n))):
            continue                                                       # no rejection
        inner = [p for i, p in ph if s < i < w]
        if not inner:
            continue
        Hm = max(inner)                                                    # structure to break
        m = next((j for j in range(w, min(w + SW_MSS_MAX + 1, n)) if c[j] > Hm), None)
        if m is None:
            continue
        if max(c[j] - o[j] for j in range(w, m + 1)) < SW_DISP_ATR * a:
            continue                                                       # no displacement
        P = float(l[w:m + 1].min())                                        # protected low
        di = m + int(h[m:].argmax())
        D = float(h[di])                                                   # leg extreme
        if di > n - 1 - SW_N or n - 1 - di > SW_MAX_AGE:
            continue                                                       # unconfirmed / stale
        if D - P < SW_MIN_LEG_ATR * a:
            continue
        if m + 1 < n and l[m + 1:].min() < P:
            continue                                                       # protected low broken
        entry = D - SW_FIB * (D - P)
        if l[di + 1:].min() <= entry:
            continue                                                       # already tapped
        jj = next((j for j in range(m, max(w - 5, 0) - 1, -1) if c[j] < o[j]), None)
        ob = (float(l[jj]), float(h[jj])) if jj is not None else None
        ob_ok = ob is not None and ob[0] - SW_OB_TOL * a <= entry <= ob[1] + SW_OB_TOL * a
        if SW_REQUIRE_OB and not ob_ok:
            continue
        sl = P - SW_SL_BUF * a
        risk = entry - sl
        if risk <= 0:
            continue
        rr = (D - entry) / risk
        if rr < bot.MIN_RR:
            continue
        cand = {"entry": float(entry), "sl": float(sl), "tp": D, "rr": rr, "price": float(c[-1]),
                "swept": Ls, "mss_level": float(Hm), "protected": P, "extreme": D,
                "ob": ob, "ob_ok": ob_ok, "di": di,
                "sweep_time": d.index[w], "mss_time": d.index[m]}
        if best is None or di > best["di"]:
            best = cand
    return best


def analyze(name: str, d: pd.DataFrame, tf: str):
    """d = completed candles. Returns a list of signals (0, 1 or 2)."""
    out = []
    for mir in (False, True):
        r = _buy_sweep(crt.mirror(d) if mir else d)
        if not r:
            continue
        r["symbol"], r["side"], r["tf"] = name, "BUY", tf
        if mir:
            r["side"] = "SELL"
            for key in ("entry", "sl", "tp", "price", "swept", "mss_level", "protected", "extreme"):
                r[key] = -r[key]
            if r["ob"]:
                r["ob"] = (-r["ob"][1], -r["ob"][0])
        r["tid"] = f"sweep|{name}|{r['side']}|{tf}|{r['sweep_time']}|{r['mss_time']}"
        out.append(r)
    return out


def format_msg(s: dict) -> str:
    dp = bot.DECIMALS[s["symbol"]]
    f = lambda v: f"{v:.{dp}f}"
    buy = s["side"] == "BUY"
    icon = "📈" if buy else "📉"
    swept = "low" if buy else "high"
    way = "above" if buy else "below"
    back = "back above" if buy else "back below"
    ob = f"{f(s['ob'][0])} – {f(s['ob'][1])}" if s["ob"] else "-"
    return (
        f"{icon} <b>{s['symbol']} — SWEEP + MSS {s['side']} LIMIT</b>  ({s['tf']})\n"
        f"🕐 {s['session']} session\n\n"
        f"Entry (Fib {SW_FIB}): <code>{f(s['entry'])}</code>   Price: <code>{f(s['price'])}</code>\n"
        f"🛑 SL: <code>{f(s['sl'])}</code> (protected {swept}, {f(s['protected'])})\n"
        f"🎯 TP: <code>{f(s['tp'])}</code> ({'buy' if buy else 'sell'}-side liquidity)\n"
        f"R:R ≈ <b>{s['rr']:.1f}</b>\n\n"
        f"<b>Checklist</b>\n"
        f"✅ Liquidity sweep: {swept} {f(s['swept'])} taken, closed {back}\n"
        f"✅ MSS: closed {way} {f(s['mss_level'])} with displacement\n"
        f"✅ Fibonacci {SW_FIB}: {f(s['protected'])} → {f(s['extreme'])}\n"
        f"{'✅' if s['ob_ok'] else '❌'} Order block {ob} lines up with the {SW_FIB} level\n"
        f"✅ Limit set: entry {SW_FIB}, SL at the protected {swept}, TP at the extreme\n\n"
        f"⚠️ Auto-detected from swing points. Confirm the sweep, MSS and order block on your chart.\n"
        f"<i>Signal only. Not financial advice.</i>"
    )


def scan_once(seen=lambda tid: False):
    """seen(tid) -> True if that setup was already sent (main.py passes its own check)."""
    for name, ticker in bot.SYMBOLS.items():
        for tf in SW_TIMEFRAMES:
            interval, period, minutes = TF[tf]
            try:
                now = pd.Timestamp.now(tz="UTC")
                df = bot.fetch(ticker, interval, period)
                d = df[df.index + pd.Timedelta(minutes=minutes) <= now]   # completed candles only
                limit = pd.Timedelta(minutes=max(2 * minutes, 45))
                if d.empty or now - (d.index[-1] + pd.Timedelta(minutes=minutes)) > limit:
                    log.info("%s %s: market closed / stale data", name, tf)
                    continue
                session = bot.current_session(now)
                if session not in bot.ENABLED_SESSIONS:
                    log.info("%s: %s session disabled", name, session)
                    continue
                sigs = analyze(name, d, tf)
            except Exception as e:
                log.error("%s %s: %s", name, tf, e)
                continue
            if not sigs:
                log.info("%s %s: no sweep/MSS setup", name, tf)
            for sig in sigs:
                if seen(sig["tid"]):
                    log.info("%s: sweep setup already sent", name)
                    continue
                sig["session"] = session
                log.info("%s: new sweep+MSS %s setup", name, sig["side"])
                bot.send(format_msg(sig))


if __name__ == "__main__":
    scan_once()

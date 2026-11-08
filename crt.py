"""
CRT (Candle Range Theory) signal scanner - signals only.
Uses bot.py for data + Telegram. Run:  python crt.py --once
"""
import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import pandas as pd

import bot
from bot import log

# ----------------------------- CONFIG ---------------------------------
CRT_HTF_HOURS = 4            # CRT candle size (4H)
CRT_VALID_HOURS = 4          # signal window after the sweep candle closes
CISD_MAX_AGE_BARS = 16       # CISD must be this recent (15m bars)
STRUCT_ALLOW_NEUTRAL = True  # accept 1H swing "NEUTRAL" as aligned
REQUIRE_ALL = True           # send only if every auto-checked rule passes
STATE_FILE = Path(__file__).with_name("sent_crt.json")


# ----------------------------- HELPERS --------------------------------
def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Flip prices so SELL setups can reuse the BUY logic."""
    return pd.DataFrame({"open": -df["open"], "high": -df["low"],
                         "low": -df["high"], "close": -df["close"]}, index=df.index)


def to_htf(df1h: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    o = df1h.resample(f"{CRT_HTF_HOURS}h", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if len(o) and o.index[-1] + pd.Timedelta(hours=CRT_HTF_HOURS) > now:
        o = o.iloc[:-1]          # drop the still-forming candle
    return o


def bull_fvgs(d: pd.DataFrame):
    h, l = d["high"].values, d["low"].values
    return [(h[i - 2], l[i]) for i in range(2, len(d)) if h[i - 2] < l[i]]


def bear_fvgs(d: pd.DataFrame):
    h, l = d["high"].values, d["low"].values
    return [(h[i], l[i - 2]) for i in range(2, len(d)) if l[i - 2] > h[i]]


def find_cisd(d: pd.DataFrame, start: pd.Timestamp):
    """Buy-side CISD: after the sweep low, price closes above the open of the
    down-move (bearish run) that led into that low."""
    x = d[d.index >= start]
    if len(x) < 4:
        return None
    o, c, lo = x["open"].values, x["close"].values, x["low"].values
    li = int(lo.argmin())
    j = li
    if c[j] >= o[j]:
        j -= 1
    if j < 0 or c[j] >= o[j]:
        return None
    while j - 1 >= 0 and c[j - 1] < o[j - 1]:
        j -= 1
    level = float(o[j])
    for k in range(li + 1, len(x)):
        if c[k] > level:
            return {"level": level, "low": float(lo[li]), "k": k, "li": li, "x": x}
    return None


# ----------------------------- CORE (BUY orientation) -----------------
def _buy(d15, d4, d1h, now):
    if len(d4) < 20:
        return None
    c1, c2 = d4.iloc[-2], d4.iloc[-1]
    c2_start = d4.index[-1]
    if not (c2["low"] < c1["low"] and c1["low"] < c2["close"] <= c1["high"]):
        return None
    if now - (c2_start + pd.Timedelta(hours=CRT_HTF_HOURS)) > pd.Timedelta(hours=CRT_VALID_HOURS):
        return None
    a4, a15 = bot.atr(d4), bot.atr(d15)
    if not a4 or not a15 or a4 != a4 or a15 != a15:
        return None
    price = float(d15["close"].iloc[-1])

    # --- LTF confirmation: CISD (mandatory) ---
    cis = find_cisd(d15, c2_start)
    if not cis:
        return None
    x, k, li = cis["x"], cis["k"], cis["li"]
    entry, L = cis["level"], cis["low"]
    if len(x) - 1 - k > CISD_MAX_AGE_BARS:
        return None
    if k + 1 < len(x) and x["low"].iloc[k + 1:].min() <= entry:
        return None                      # entry already tapped
    if price <= entry or price - entry > bot.MAX_ENTRY_DIST_ATR * a15:
        return None
    sl = L - bot.SL_BUFFER_ATR * a15
    risk = entry - sl
    if risk <= 0:
        return None

    # --- HTF key level mitigated by the sweep ---
    ref = d4.iloc[-42:-1]
    kinds = []
    if any(hi >= c2["low"] and lo <= c1["low"] + 0.25 * a4 for lo, hi in bull_fvgs(ref)):
        kinds.append("4H FVG")
    d_open, pdh, pdl = bot.daily_levels(d15)
    for lvl, label in ((d_open, "daily open"), (pdh, "prev-day level"), (pdl, "prev-day level")):
        if lvl is not None and c2["low"] <= lvl <= c1["low"] + 0.25 * a4:
            kinds.append(label)
    keylevel = ", ".join(sorted(set(kinds)))

    # --- 1H swing structure (raw, in buy orientation) ---
    w = d1h.iloc[-120:]
    hs, ls = bot.swing_points(w, "high", 2), bot.swing_points(w, "low", 2)
    swing = "NEUTRAL"
    if len(hs) >= 2 and len(ls) >= 2:
        if hs[-1] > hs[-2] and ls[-1] > ls[-2]:
            swing = "UP"
        elif hs[-1] < hs[-2] and ls[-1] < ls[-2]:
            swing = "DOWN"
    struct_ok = swing == "UP" or (swing == "NEUTRAL" and STRUCT_ALLOW_NEUTRAL)

    # --- liquidity swept before entry ---
    liq = []
    if any(L < s <= c1["low"] + 0.25 * a4 for s in bot.swing_points(d1h.iloc[-72:], "low", 3)):
        liq.append("structure")
    pre = d15[d15.index <= x.index[li]].iloc[-40:]
    if any(L < s <= L + 1.5 * a15 for s in bot.swing_points(pre, "low", 2)):
        liq.append("inducement")
    lows = pre["low"].iloc[:-1].tolist()
    if any(abs(a - b) <= 0.1 * a15 and L < min(a, b) <= L + 2 * a15 for a, b in combinations(lows, 2)):
        liq.append("engineered")

    # --- targets: internal first, then external CRT high / HTF roadblock ---
    tp2, tp2_label = float(c1["high"]), "external CRT high"
    roads = [lo for lo, hi in bear_fvgs(ref) if entry < lo < tp2]
    if roads:
        tp2, tp2_label = float(min(roads)), "HTF roadblock (4H FVG)"
    cands = [s for s in bot.swing_points(d15.iloc[-96:], "high", 2) if entry < s < tp2]
    tp1, tp1_label = (float(min(cands)), "internal swing high") if cands else (None, "")
    if tp1 is None:
        mid = float((c1["high"] + c1["low"]) / 2)
        if entry < mid < tp2:
            tp1, tp1_label = mid, "CRT 50% (internal)"
    rr = (tp2 - entry) / risk
    tp_ok = tp2 > entry and rr >= bot.MIN_RR

    checks = {"structure": struct_ok, "crt": True, "keylevel": bool(kinds),
              "cisd": True, "liquidity": bool(liq), "tp": tp_ok}
    if REQUIRE_ALL and not all(checks.values()):
        return None
    return {"entry": entry, "sl": sl, "tp1": tp1, "tp1_label": tp1_label,
            "tp2": tp2, "tp2_label": tp2_label, "rr": rr, "price": price,
            "swing": swing, "keylevel": keylevel, "liq": liq, "checks": checks,
            "crt_lo": float(c1["low"]), "crt_hi": float(c1["high"]),
            "swept": float(c1["low"]), "c2_start": c2_start}


def analyze(name, d15, d4, d1h, now):
    if len(d4) < 20:
        return None
    c1, c2 = d4.iloc[-2], d4.iloc[-1]
    if c2["high"] > c1["high"] and c2["low"] < c1["low"]:
        return None                      # both sides swept: no clean CRT
    if c2["low"] < c1["low"] and c1["low"] < c2["close"] <= c1["high"]:
        mir = False
    elif c2["high"] > c1["high"] and c1["low"] <= c2["close"] < c1["high"]:
        mir = True
    else:
        return None
    if mir:
        d15, d4, d1h = mirror(d15), mirror(d4), mirror(d1h)
    r = _buy(d15, d4, d1h, now)
    if not r:
        return None
    r["symbol"], r["side"] = name, "BUY"
    if mir:
        r["side"] = "SELL"
        for key in ("entry", "sl", "tp1", "tp2", "price", "swept"):
            if r[key] is not None:
                r[key] = -r[key]
        r["crt_lo"], r["crt_hi"] = -r["crt_hi"], -r["crt_lo"]
        r["swing"] = {"UP": "DOWN", "DOWN": "UP"}.get(r["swing"], r["swing"])
        for key in ("tp1_label", "tp2_label"):
            r[key] = r[key].replace("high", "low")
    return r


# ----------------------------- MESSAGE --------------------------------
def format_msg(s):
    dp = bot.DECIMALS[s["symbol"]]
    f = lambda v: f"{v:.{dp}f}"
    side = s["side"]
    icon = "📈" if side == "BUY" else "📉"
    ok = lambda b: "✅" if b else "❌"
    c = s["checks"]
    swept_word = "low" if side == "BUY" else "high"
    tp1 = f"🎯 TP1: <code>{f(s['tp1'])}</code> ({s['tp1_label']})\n" if s["tp1"] is not None else ""
    return (
        f"{icon} <b>{s['symbol']} — CRT {side} LIMIT</b>  ({CRT_HTF_HOURS}H CRT • 15m entry)\n"
        f"🕐 {s['session']} session\n\n"
        f"CRT range: <code>{f(s['crt_lo'])} – {f(s['crt_hi'])}</code>\n"
        f"Entry (CISD level): <code>{f(s['entry'])}</code>   Price: <code>{f(s['price'])}</code>\n"
        f"🛑 SL: <code>{f(s['sl'])}</code>\n"
        f"{tp1}"
        f"🎯 TP2: <code>{f(s['tp2'])}</code> ({s['tp2_label']})\n"
        f"R:R to TP2 ≈ <b>{s['rr']:.1f}</b>\n\n"
        f"<b>Checklist</b>\n"
        f"{ok(c['structure'])} Structure: 1H swing {s['swing']} aligned with 15m fractal\n"
        f"{ok(c['crt'])} CRT {swept_word} swept ({f(s['swept'])}), closed back inside range\n"
        f"{ok(c['keylevel'])} HTF key level mitigated: {s['keylevel'] or '-'}\n"
        f"{ok(c['cisd'])} LTF CISD confirmation after the sweep\n"
        f"{ok(c['liquidity'])} Liquidity swept: {', '.join(s['liq']) or '-'}\n"
        f"{ok(c['tp'])} TP at internal → external CRT / HTF roadblock\n\n"
        f"⚠️ Not auto-checked: A/V-shape, OB, QMR, BB entry models — confirm on your chart.\n"
        f"<i>Signal only. Not financial advice.</i>"
    )


# ----------------------------- MAIN -----------------------------------
def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def scan_once():
    state = load_state()
    cutoff = time.time() - 3 * 86400
    state = {k: v for k, v in state.items() if v > cutoff}
    for name, ticker in bot.SYMBOLS.items():
        try:
            now = pd.Timestamp.now(tz="UTC")
            d15 = bot.fetch(ticker)
            if (now - d15.index[-1]).total_seconds() > bot.MAX_DATA_AGE_MIN * 60:
                log.info("%s: market closed / stale data", name)
                continue
            session = bot.current_session(now)
            if session not in bot.ENABLED_SESSIONS:
                log.info("%s: %s session disabled", name, session)
                continue
            d1h = bot.fetch(ticker, "1h", "60d")
            sig = analyze(name, d15, to_htf(d1h, now), d1h, now)
            if sig:
                sig["session"] = session
        except Exception as e:
            log.error("%s: %s", name, e)
            continue
        if not sig:
            log.info("%s: no CRT setup", name)
            continue
        key = f"CRT|{name}|{sig['side']}|{sig['c2_start']}"
        if key in state:
            log.info("%s: CRT setup already sent", name)
            continue
        log.info("%s: new CRT %s setup", name, sig["side"])
        if bot.send(format_msg(sig)):
            state[key] = time.time()
    STATE_FILE.write_text(json.dumps(state))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.parse_args()
    scan_once()


if __name__ == "__main__":
    main()

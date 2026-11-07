"""
Daily-Open + FVG signal bot (signals only, no order execution).
Run:  python bot.py --once | --test | (no flag = loop forever)
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ----------------------------- CONFIG ---------------------------------
SYMBOLS = {
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD",
    "EURUSD": "EURUSD=X",
}
DECIMALS = {"XAUUSD": 2, "BTCUSD": 2, "EURUSD": 5}

INTERVAL = "15m"
SCAN_EVERY_SEC = 300
LOOKBACK_BARS = 80
MIN_GAP_ATR = 0.15
MAX_ENTRY_DIST_ATR = 4.0
SL_BUFFER_ATR = 0.25
MIN_RR = 1.5
SWING_N = 3
ENTRY_MODE = "mid"          # "mid" or "edge"
DAY_START_UTC_HOUR = 0

SESSIONS_UTC = {"Asia": (0, 7), "London": (7, 13), "New York": (13, 21), "After-hours": (21, 24)}
ENABLED_SESSIONS = ["Asia", "London", "New York", "After-hours"]

TREND_FILTER = True
TREND_INTERVAL = "1h"
EMA_FAST, EMA_SLOW = 50, 200
ALLOW_NEUTRAL = False
MAX_DATA_AGE_MIN = 45

STATE_FILE = Path(__file__).with_name("sent_signals.json")


def _clean(v: str) -> str:
    return v.strip().strip("\"'").strip()


TOKEN = _clean(os.getenv("TELEGRAM_BOT_TOKEN", ""))
if TOKEN.lower().startswith("bot"):   # in case "bot" was pasted in front
    TOKEN = TOKEN[3:]
CHAT_ID = _clean(os.getenv("TELEGRAM_CHAT_ID", ""))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("signalbot")


# ----------------------------- DATA -----------------------------------
def fetch(ticker: str, interval: str = INTERVAL, period: str = "7d") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def atr(df: pd.DataFrame, n: int = 14) -> float:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def session_start(now: pd.Timestamp) -> pd.Timestamp:
    start = now.normalize() + pd.Timedelta(hours=DAY_START_UTC_HOUR)
    return start if start <= now else start - pd.Timedelta(days=1)


def current_session(ts: pd.Timestamp) -> str:
    h = ts.hour
    for name, (a, b) in SESSIONS_UTC.items():
        if a <= h < b:
            return name
    return "After-hours"


def get_trend(htf: pd.DataFrame) -> str:
    c = htf["close"]
    fast = c.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
    slow = c.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
    price = c.iloc[-1]
    if fast > slow and price > slow:
        return "UP"
    if fast < slow and price < slow:
        return "DOWN"
    return "NEUTRAL"


# ----------------------------- LOGIC ----------------------------------
def daily_levels(df: pd.DataFrame):
    now = df.index[-1]
    start = session_start(now)
    today = df[df.index >= start]
    prev = df[(df.index >= start - pd.Timedelta(days=1)) & (df.index < start)]
    if today.empty:
        return None, None, None
    d_open = float(today["open"].iloc[0])
    pdh = float(prev["high"].max()) if not prev.empty else None
    pdl = float(prev["low"].min()) if not prev.empty else None
    return d_open, pdh, pdl


def find_fvgs(df: pd.DataFrame, side: str, min_gap: float):
    d = df.iloc[-LOOKBACK_BARS:].reset_index()
    out = []
    for i in range(2, len(d)):
        c1, c3 = d.iloc[i - 2], d.iloc[i]
        if side == "BUY" and c1["high"] < c3["low"]:
            lo, hi = float(c1["high"]), float(c3["low"])
        elif side == "SELL" and c1["low"] > c3["high"]:
            lo, hi = float(c3["high"]), float(c1["low"])
        else:
            continue
        if hi - lo < min_gap:
            continue
        later = d.iloc[i + 1:]
        if side == "BUY" and not later.empty and later["low"].min() <= hi:
            continue
        if side == "SELL" and not later.empty and later["high"].max() >= lo:
            continue
        out.append({"low": lo, "high": hi, "time": d.iloc[i][d.columns[0]]})
    return out


def swing_points(df: pd.DataFrame, kind: str, n: int = SWING_N):
    s = df[kind].reset_index(drop=True)
    pts = []
    for i in range(n, len(s) - n):
        w = s.iloc[i - n:i + n + 1]
        if kind == "high" and s.iloc[i] == w.max():
            pts.append(float(s.iloc[i]))
        elif kind == "low" and s.iloc[i] == w.min():
            pts.append(float(s.iloc[i]))
    return pts


def build_signal(name: str, df: pd.DataFrame, trend: str = "NEUTRAL"):
    d_open, pdh, pdl = daily_levels(df)
    if d_open is None:
        return None
    price = float(df["close"].iloc[-1])
    a = atr(df)
    if not a or a != a:
        return None

    side = "BUY" if price < d_open else "SELL"
    if TREND_FILTER:
        want = "UP" if side == "BUY" else "DOWN"
        if trend != want and not (trend == "NEUTRAL" and ALLOW_NEUTRAL):
            return None
    fvgs = find_fvgs(df, side, MIN_GAP_ATR * a)

    if side == "BUY":
        pools = [(p, "swing high") for p in swing_points(df.iloc[-LOOKBACK_BARS * 2:], "high")]
        if pdh:
            pools.append((pdh, "prev day high"))
    else:
        pools = [(p, "swing low") for p in swing_points(df.iloc[-LOOKBACK_BARS * 2:], "low")]
        if pdl:
            pools.append((pdl, "prev day low"))

    best = None
    for g in fvgs:
        if side == "BUY":
            entry = (g["low"] + g["high"]) / 2 if ENTRY_MODE == "mid" else g["high"]
            if entry >= price or price - entry > MAX_ENTRY_DIST_ATR * a:
                continue
            sl = g["low"] - SL_BUFFER_ATR * a
            cands = sorted(p for p in pools if p[0] > price)
        else:
            entry = (g["low"] + g["high"]) / 2 if ENTRY_MODE == "mid" else g["low"]
            if entry <= price or entry - price > MAX_ENTRY_DIST_ATR * a:
                continue
            sl = g["high"] + SL_BUFFER_ATR * a
            cands = sorted((p for p in pools if p[0] < price), reverse=True)

        risk = abs(entry - sl)
        if risk <= 0:
            continue
        for level, label in cands:
            rr = abs(level - entry) / risk
            if rr >= MIN_RR:
                cand = {"symbol": name, "side": side, "entry": entry, "sl": sl,
                        "tp": level, "tp_label": label, "rr": rr, "price": price,
                        "daily_open": d_open, "fvg": (g["low"], g["high"]),
                        "fvg_time": g["time"], "trend": trend}
                if best is None or abs(price - entry) < abs(price - best["entry"]):
                    best = cand
                break
    return best


# ----------------------------- TELEGRAM -------------------------------
def format_msg(s: dict) -> str:
    dp = DECIMALS[s["symbol"]]
    f = lambda x: f"{x:.{dp}f}"
    icon = "📈" if s["side"] == "BUY" else "📉"
    return (
        f"{icon} <b>{s['symbol']} — {s['side']} LIMIT</b>  ({INTERVAL})\n\n"
        f"Daily open: <code>{f(s['daily_open'])}</code>  |  Price: <code>{f(s['price'])}</code>\n"
        f"🔎 FVG: <code>{f(s['fvg'][0])} – {f(s['fvg'][1])}</code>\n\n"
        f"Entry: <code>{f(s['entry'])}</code>\n"
        f"🛑 SL: <code>{f(s['sl'])}</code>\n"
        f"🎯 TP: <code>{f(s['tp'])}</code> ({s['tp_label']})\n"
        f"R:R ≈ <b>{s['rr']:.1f}</b>\n"
        f"🕐 {s['session']} session  |  1H trend: {s['trend']}\n\n"
        f"<i>Signal only — confirm on your chart before entering. Not financial advice.</i>"
    )


def send(text: str) -> bool:
    if not TOKEN or not CHAT_ID:
        log.warning("Telegram not configured; printing instead:\n%s", text)
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for attempt in range(1, 5):          # retry on network hiccups
        try:
            r = requests.post(
                url,
                json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=20,
            )
            if r.ok:
                return True
            log.error("Telegram error %s: %s", r.status_code, r.text)
            if r.status_code < 500 and r.status_code != 429:
                return False             # bad token / chat id: retrying won't help
        except requests.RequestException as e:
            log.warning("Telegram send failed (attempt %s/4): %s", attempt, e)
        time.sleep(3 * attempt)
    return False


# ----------------------------- STATE ----------------------------------
def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state))


def signal_key(s: dict) -> str:
    return f"{s['symbol']}|{s['side']}|{s['fvg_time']}"


# ----------------------------- MAIN -----------------------------------
def scan_once():
    state = load_state()
    cutoff = time.time() - 3 * 86400
    state = {k: v for k, v in state.items() if v > cutoff}

    for name, ticker in SYMBOLS.items():
        try:
            df = fetch(ticker)
            now = pd.Timestamp.now(tz="UTC")
            if (now - df.index[-1]).total_seconds() > MAX_DATA_AGE_MIN * 60:
                log.info("%s: market closed / stale data", name)
                continue
            session = current_session(now)
            if session not in ENABLED_SESSIONS:
                log.info("%s: %s session disabled", name, session)
                continue
            trend = "NEUTRAL"
            if TREND_FILTER:
                trend = get_trend(fetch(ticker, TREND_INTERVAL, "60d"))
            sig = build_signal(name, df, trend)
            if sig:
                sig["session"] = session
        except Exception as e:
            log.error("%s: %s", name, e)
            continue
        if not sig:
            log.info("%s: no setup", name)
            continue
        key = signal_key(sig)
        if key in state:
            log.info("%s: setup already sent", name)
            continue
        msg = format_msg(sig)
        log.info("%s: new %s setup", name, sig["side"])
        if send(msg):
            state[key] = time.time()
    save_state(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        ok = send("✅ Signal bot connected. XAUUSD / BTCUSD / EURUSD scanner is ready.")
        sys.exit(0 if ok else 1)
    if args.once:
        scan_once()
        return
    log.info("Bot running. Scanning every %ss", SCAN_EVERY_SEC)
    while True:
        try:
            scan_once()
        except Exception as e:
            log.exception("scan failed: %s", e)
        time.sleep(SCAN_EVERY_SEC)


if __name__ == "__main__":
    main()

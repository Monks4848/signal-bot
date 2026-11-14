"""
Runs all scanners (daily-open + CRT + trendline), then tracks every signal sent:
did it hit SL or TP? Keeps a win/loss tally in trades.json.

  python main.py             scan + track (used by GitHub Actions)
  python main.py --summary   send the current tally to Telegram
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import bot
import crt
import tl

TRADES_FILE = Path(__file__).with_name("trades.json")
PENDING_MAX_HOURS = 8     # cancel a limit order not filled within this time
MAX_AGE_DAYS = 5          # stop tracking anything older than this
NAMES = {"daily": "daily-open", "crt": "CRT", "tl": "trendline"}

_orig_send = bot.send
_orig_fmt_bot, _orig_fmt_crt, _orig_fmt_tl = bot.format_msg, crt.format_msg, tl.format_msg
_pending = {}
trades = []


# ----------------------------- STORAGE --------------------------------
def load_trades():
    try:
        return json.loads(TRADES_FILE.read_text())
    except Exception:
        return []


def save_trades():
    TRADES_FILE.write_text(json.dumps(trades, indent=1, default=str))


def record(kind: str, s: dict):
    if kind == "tl":
        tid = s["tid"]
    else:
        key = s["fvg_time"] if kind == "daily" else s["c2_start"]
        tid = f"{kind}|{s['symbol']}|{s['side']}|{key}"
    if any(t["id"] == tid for t in trades):
        return
    trades.append({
        "id": tid, "strategy": kind, "symbol": s["symbol"], "side": s["side"],
        "entry": s["entry"], "sl": s["sl"],
        "tp1": s.get("tp1") if kind == "crt" else None,
        "tp2": s["tp2"] if kind == "crt" else s["tp"],
        "rr": s["rr"], "created": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "pending", "result": None, "tp1_hit": False,
    })


# ----------------------------- SIMULATION -----------------------------
def simulate(t: dict, bars: pd.DataFrame, now: pd.Timestamp) -> dict:
    """Replay 15m bars after the signal. Same-bar clashes are resolved
    pessimistically: SL is checked before TP, and TP is only checked
    from the bar after the entry fills."""
    buy = t["side"] == "BUY"
    created = pd.Timestamp(t["created"])
    start = created.floor("15min") + pd.Timedelta(minutes=15)
    entry, sl, tp1, tp2 = t["entry"], t["sl"], t["tp1"], t["tp2"]
    out = {"status": "pending", "result": None, "tp1_hit": False}
    for ts, row in bars[bars.index >= start].iterrows():
        hi, lo = float(row["high"]), float(row["low"])
        hit_sl = lo <= sl if buy else hi >= sl
        hit_entry = lo <= entry if buy else hi >= entry
        hit_tp1 = tp1 is not None and (hi >= tp1 if buy else lo <= tp1)
        hit_tp2 = hi >= tp2 if buy else lo <= tp2
        if out["status"] == "pending":
            if hit_tp2 and not hit_entry:
                out["result"] = "missed"
                break
            if hit_entry:
                out["status"] = "open"
                if hit_sl:
                    out["result"] = "loss"
                    break
            elif ts - start > pd.Timedelta(hours=PENDING_MAX_HOURS):
                out["result"] = "expired"
                break
            continue
        if hit_sl:
            out["result"] = "partial" if out["tp1_hit"] else "loss"
            break
        if hit_tp2:
            out["result"] = "win"
            break
        if hit_tp1:
            out["tp1_hit"] = True
    if out["result"] is None:
        age = now - created
        if out["status"] == "pending" and age > pd.Timedelta(hours=PENDING_MAX_HOURS):
            out["result"] = "expired"
        elif age > pd.Timedelta(days=MAX_AGE_DAYS):
            out["result"] = "expired"
    return out


# ----------------------------- MESSAGES -------------------------------
def tally_text() -> str:
    def n(res, kind=None):
        return sum(1 for t in trades if t["result"] == res and (kind is None or t["strategy"] == kind))

    w, l, p = n("win"), n("loss"), n("partial")
    total = w + l + p
    wr = f"{w / total * 100:.0f}%" if total else "n/a"
    live = sum(1 for t in trades if t["result"] is None)
    lines = [f"📊 <b>Tally</b>: {w}W • {l}L • {p} partial • win rate {wr}"]
    for kind, name in NAMES.items():
        kw, kl, kp = n("win", kind), n("loss", kind), n("partial", kind)
        if kw + kl + kp:
            lines.append(f"   {name}: {kw}W • {kl}L • {kp}P")
    if live:
        lines.append(f"   Active/pending: {live}")
    lines.append("<i>Partial = TP1 hit, then stopped out. Win = final TP hit.</i>")
    return "\n".join(lines)


def result_msg(t: dict) -> str:
    dp = bot.DECIMALS[t["symbol"]]
    f = lambda v: f"{v:.{dp}f}"
    head = f"{t['symbol']} {t['side']} ({NAMES[t['strategy']]})"
    res = t["result"]
    if res == "win":
        body = f"✅ <b>{head} — TP HIT</b>\nEntry {f(t['entry'])} → TP {f(t['tp2'])}  (+{t['rr']:.1f}R)"
    elif res == "loss":
        body = f"❌ <b>{head} — SL HIT</b>\nEntry {f(t['entry'])} → SL {f(t['sl'])}  (−1R)"
    elif res == "partial":
        body = (f"🟡 <b>{head} — TP1 hit, then SL</b>\n"
                f"TP1 {f(t['tp1'])} was reached before price stopped out at {f(t['sl'])}")
    elif res == "missed":
        body = f"⚪ <b>{head} — missed</b>\nPrice reached the target without ever filling the entry. Not counted."
    else:
        body = f"⌛ <b>{head} — cancelled</b>\nEntry never filled in time. Not counted."
    if res in ("win", "loss", "partial"):
        body += "\n\n" + tally_text()
    return body


# ----------------------------- TRACKING -------------------------------
def update_trades():
    open_t = [t for t in trades if t["result"] is None]
    if not open_t:
        return
    now = pd.Timestamp.now(tz="UTC")
    data = {}
    for sym in {t["symbol"] for t in open_t}:
        try:
            data[sym] = bot.fetch(bot.SYMBOLS[sym])
        except Exception as e:
            bot.log.error("tracker %s: %s", sym, e)
    for t in open_t:
        if t["symbol"] not in data:
            continue
        r = simulate(t, data[t["symbol"]], now)
        if r["tp1_hit"] and not t["tp1_hit"] and r["result"] is None:
            dp = bot.DECIMALS[t["symbol"]]
            _orig_send(f"🎯 <b>{t['symbol']} {t['side']} ({NAMES[t['strategy']]}) — TP1 hit</b>\n"
                       f"TP1 {t['tp1']:.{dp}f} reached. Still running toward TP2.")
        t["tp1_hit"], t["status"] = r["tp1_hit"], r["status"]
        if r["result"]:
            t["result"] = r["result"]
            t["status"] = "closed"
            t["closed"] = now.isoformat()
            _orig_send(result_msg(t))


# ----------------------------- WIRING ---------------------------------
def _fmt_bot(s):
    _pending["sig"] = ("daily", s)
    return _orig_fmt_bot(s)


def _fmt_crt(s):
    _pending["sig"] = ("crt", s)
    return _orig_fmt_crt(s)


def _fmt_tl(s):
    _pending["sig"] = ("tl", s)
    return _orig_fmt_tl(s)


def _seen(tid):
    return any(t["id"] == tid for t in trades)


def _send(text):
    ok = _orig_send(text)
    sig = _pending.pop("sig", None)
    if ok and sig:
        record(*sig)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    trades[:] = load_trades()
    if args.summary:
        _orig_send("✅ Signal bot connected.\n\n" + tally_text())
        return

    bot.send, bot.format_msg, crt.format_msg, tl.format_msg = _send, _fmt_bot, _fmt_crt, _fmt_tl
    try:
        update_trades()
        for scan in (bot.scan_once, crt.scan_once, lambda: tl.scan_once(_seen)):
            try:
                scan()
            except Exception as e:
                bot.log.exception("scan failed: %s", e)
    finally:
        save_trades()


if __name__ == "__main__":
    main()

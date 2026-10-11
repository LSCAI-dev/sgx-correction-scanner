#!/usr/bin/env python3
"""
SGX Correction Early-Warning Scanner
====================================
Scores each counter daily for "impending correction" risk using the
signals that flagged the Oct 2026 Singapore-bank sell-off 1-3 weeks early.

Signals (1 point each unless noted)
  1  RSI_DIV     Price within 2% of 40-day high, RSI(14) >= 6 pts below RSI at that high
  2  MACD_FADE   Price within 2% of 40-day high, MACD line <= 60% of its 40-day peak
  3  MACD_NEG    MACD histogram < 0 (or crossed below 0 in last 3 bars)
  4  OBV_DIV     Price within 2% of 40-day high, OBV below its level at that high
  5  CMF_NEG     Chaikin Money Flow(20) < 0 while price within 3% of 40-day high
  6  DIST_DAYS   >= 3 distribution days in last 15 bars (close -0.5%+ on above-avg volume)
  7  LOWER_HIGH  60-day high set > 25 bars ago and every rally since stalled >= 2% below it
  8  BELOW_20    Close below 20-day SMA and 20-SMA slope (5 bars) <= 0
  9  HA_BEAR     Last 2 Heikin Ashi candles red and the latest has no upper wick
 10  SECTOR      >= 2 of the 3 SG banks score >= 3 (sector-wide distribution)  [+1]
 11  YIELD_UP    US 10Y yield up >= 20bp over 10 sessions                       [+1]
 12  EARNINGS    Results within the next 30 calendar days                       [+1]

Hard triggers (shown separately, always alert)
  BREAKDOWN   Close below the 20-day low (neckline) on volume >= 1.5x 20-day average
  GAP_DOWN    Open >= 2% below prior close

Alert levels
  GREEN 0-2 hold | AMBER 3-4 tighten stops, no adding | RED >= 5 reduce / exit
  Any hard trigger = BREAK: act on your pre-set plan.

Re-entry checklist (printed once RSI has dipped < 35 in the last 10 bars)
  WASHED_OUT, RSI_RECOVER (>45), MACD_TURN (hist rising 2 bars),
  HA_GREEN2 (2 green HA candles), HIGHER_LOW (5-bar low above 10-bar low)
  4-5 of 5 = CONFIRMED re-entry setup; fewer = still watching.

Usage
  python correction_scanner.py                      # live via yfinance
  python correction_scanner.py --csv-dir data/      # offline CSVs (date,open,high,low,close,volume)
  python correction_scanner.py --history 40         # also print score history for last 40 bars

Env (optional, for push alerts)
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Data: Yahoo Finance via yfinance (delayed). Verify on your broker terminal
(Tiger / Moomoo / IBKR) before acting. Not financial advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- config
WATCHLIST = {
    "D05.SI": "DBS",
    "U11.SI": "UOB",
    "O39.SI": "OCBC",
    # add more, e.g. "Z74.SI": "Singtel", "BS6.SI": "YZJ", "U96.SI": "Sembcorp", "BN4.SI": "Keppel"
}
BANKS = {"D05.SI", "U11.SI", "O39.SI"}
# Next results dates — update each quarter
EARNINGS = {
    "D05.SI": "2026-11-05",
    "O39.SI": "2026-11-06",
    "U11.SI": "2026-11-12",   # estimate; confirm on SGX
}
YIELD_TICKER = "^TNX"         # US 10Y (Yahoo quotes it in % x1)
LOOKBACK = "1y"

AMBER, RED = 3, 5


# ----------------------------------------------------------------- indicators
def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macd(c: pd.Series):
    line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, line - sig


def obv(c: pd.Series, v: pd.Series) -> pd.Series:
    return (np.sign(c.diff()).fillna(0) * v.fillna(0)).cumsum()


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    rng = (df.high - df.low).replace(0, np.nan)
    mfm = ((df.close - df.low) - (df.high - df.close)) / rng
    mfv = mfm.fillna(0) * df.volume.fillna(0)
    return mfv.rolling(n).sum() / df.volume.rolling(n).sum()


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df.open + df.high + df.low + df.close) / 4
    o = [(df.open.iloc[0] + df.close.iloc[0]) / 2]
    for i in range(1, len(df)):
        o.append((o[-1] + ha.close.iloc[i - 1]) / 2)
    ha["open"] = o
    ha["high"] = pd.concat([df.high, ha.open, ha.close], axis=1).max(axis=1)
    ha["low"] = pd.concat([df.low, ha.open, ha.close], axis=1).min(axis=1)
    return ha


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df.close)
    df["rsi"] = rsi(df.close)
    df["macd"], df["sig"], df["mh"] = macd(df.close)
    df["obv"] = obv(df.close, df.volume)
    df["cmf"] = cmf(df)
    df["sma20"] = df.close.rolling(20).mean()
    df["sma50"] = df.close.rolling(50).mean()
    df["vavg20"] = df.volume.rolling(20, min_periods=10).mean()
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    ha = heikin_ashi(df)
    df["ha_open"], df["ha_close"], df["ha_high"] = ha.open, ha.close, ha.high
    return df


# ----------------------------------------------------------------- scoring
def score_bar(df: pd.DataFrame, i: int) -> dict:
    """Score bar i (positional) using only data up to i (no look-ahead)."""
    w = df.iloc[: i + 1]
    r = w.iloc[-1]
    flags = {}
    win = w.tail(40)
    hi_idx = win.close.idxmax()
    hi = win.close.max()
    near2 = r.close >= hi * 0.98
    near3 = r.close >= hi * 0.97
    bars_since_hi = len(w.loc[hi_idx:]) - 1

    # 1 RSI divergence (only meaningful once the high is >= 5 bars old)
    flags["RSI_DIV"] = bool(near2 and bars_since_hi >= 5 and
                            w.rsi.loc[hi_idx] - r.rsi >= 6)
    # 2 MACD fade
    mpk = win.macd.max()
    flags["MACD_FADE"] = bool(near2 and mpk > 0 and r.macd <= 0.6 * mpk)
    # 3 MACD histogram negative / fresh bear cross
    flags["MACD_NEG"] = bool(r.mh < 0)
    # 4 OBV divergence
    flags["OBV_DIV"] = bool(near2 and bars_since_hi >= 5 and r.obv < w.obv.loc[hi_idx])
    # 5 CMF negative near highs
    flags["CMF_NEG"] = bool(near3 and pd.notna(r.cmf) and r.cmf < 0)
    # 6 distribution days
    last15 = w.tail(15)
    chg = last15.close.pct_change()
    dist = ((chg <= -0.005) & (last15.volume > last15.vavg20)).sum()
    flags["DIST_DAYS"] = bool(dist >= 3)
    # 7 lower-high structure vs 60-day high
    w60 = w.tail(60)
    h60_idx = w60.close.idxmax()
    since = w.loc[h60_idx:]
    flags["LOWER_HIGH"] = bool(len(since) - 1 > 25 and
                               since.close.iloc[1:].max() <= w60.close.max() * 0.98)
    # 8 below 20SMA with flat/down slope
    sl = w.sma20.iloc[-1] - w.sma20.iloc[-6] if len(w) > 25 else np.nan
    flags["BELOW_20"] = bool(pd.notna(r.sma20) and r.close < r.sma20 and pd.notna(sl) and sl <= 0)
    # 9 Heikin Ashi bearish
    ha_red = (w.ha_close < w.ha_open).tail(2).all()
    no_upper = abs(r.ha_high - max(r.ha_open, r.ha_close)) < 1e-9
    flags["HA_BEAR"] = bool(ha_red and no_upper)

    # hard triggers
    prior20_low = w.low.iloc[-21:-1].min() if len(w) > 21 else np.nan
    vol_ok = pd.notna(r.volume) and pd.notna(r.vavg20) and r.volume >= 1.5 * r.vavg20
    hard = {
        "BREAKDOWN": bool(pd.notna(prior20_low) and r.close < prior20_low and
                          (vol_ok or pd.isna(r.volume))),
        "GAP_DOWN": bool(len(w) > 1 and r.open <= w.close.iloc[-2] * 0.98 and r.open != r.close),
    }

    # re-entry checklist (after a correction): need 4 of 5 for "CONFIRMED"
    last10 = w.tail(10)
    reentry = {
        "WASHED_OUT": bool(last10.rsi.min() < 35),                       # RSI hit <35 recently
        "RSI_RECOVER": bool(r.rsi > 45),                                 # momentum back
        "MACD_TURN": bool(len(w) > 3 and w.mh.iloc[-1] > w.mh.iloc[-2] > w.mh.iloc[-3]),
        "HA_GREEN2": bool((w.ha_close > w.ha_open).tail(2).all()),
        "HIGHER_LOW": bool(len(w) > 10 and w.low.tail(5).min() > w.low.tail(10).min()),
    }
    return {
        "date": str(w.index[-1].date()),
        "close": round(float(r.close), 2),
        "rsi": round(float(r.rsi), 1),
        "macd": round(float(r.macd), 3),
        "hist": round(float(r.mh), 3),
        "cmf": None if pd.isna(r.cmf) else round(float(r.cmf), 3),
        "sma20": None if pd.isna(r.sma20) else round(float(r.sma20), 2),
        "atr": None if pd.isna(r.atr) else round(float(r.atr), 2),
        "hi40": round(float(hi), 2),
        "low20": None if pd.isna(prior20_low) else round(float(prior20_low), 2),
        "dist_days": int(dist),
        "flags": flags,
        "hard": hard,
        "reentry": reentry,
    }


def level(score: int, hard: dict) -> str:
    if any(hard.values()):
        return "BREAK"
    return "RED" if score >= RED else "AMBER" if score >= AMBER else "GREEN"


# ----------------------------------------------------------------- data
def load_live(tickers):
    import yfinance as yf
    out = {}
    for t in tickers:
        d = yf.download(t, period=LOOKBACK, interval="1d", auto_adjust=False, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        out[t] = d
    return out


def load_csv(folder, tickers):
    out = {}
    for t in tickers:
        p = Path(folder) / f"{t.split('.')[0]}.csv"
        if p.exists():
            d = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
            out[t] = d
    return out


def yield_change_bp(live: bool) -> float | None:
    if not live:
        return None
    try:
        import yfinance as yf
        y = yf.download(YIELD_TICKER, period="3mo", progress=False)["Close"].squeeze().dropna()
        return float((y.iloc[-1] - y.iloc[-11]) * 100)
    except Exception:
        return None


# ----------------------------------------------------------------- report
def run(data: dict, live: bool, history: int = 0):
    today = dt.date.today()
    ybp = yield_change_bp(live)
    results = {}
    for t, raw in data.items():
        df = enrich(raw)
        results[t] = (df, score_bar(df, len(df) - 1))

    # sector confirmation needs base scores first
    base = {t: sum(r["flags"].values()) for t, (_, r) in results.items()}
    bank_hot = sum(1 for t in BANKS if base.get(t, 0) >= AMBER) >= 2

    rows = []
    for t, (df, r) in results.items():
        r["flags"]["SECTOR"] = bool(t in BANKS and bank_hot)
        r["flags"]["YIELD_UP"] = bool(ybp is not None and ybp >= 20)
        ed = EARNINGS.get(t)
        r["flags"]["EARNINGS"] = bool(ed and 0 <= (dt.date.fromisoformat(ed) - today).days <= 30)
        s = sum(r["flags"].values())
        r.update(ticker=t, name=WATCHLIST.get(t, t), score=s, level=level(s, r["hard"]))
        rows.append(r)

        if history:
            print(f"\n{WATCHLIST.get(t, t)} score history (base signals 1-9 + hard triggers)")
            for i in range(max(30, len(df) - history), len(df)):
                h = score_bar(df, i)
                hs = sum(h["flags"].values())
                on = [k for k, v in h["flags"].items() if v] + [k for k, v in h["hard"].items() if v]
                print(f"  {h['date']} {h['close']:>7.2f} score {hs}  {level(hs, h['hard']):5} {' '.join(on)}")

    order = {"BREAK": 0, "RED": 1, "AMBER": 2, "GREEN": 3}
    rows.sort(key=lambda x: (order[x["level"]], -x["score"]))
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"SGX Correction Scan — {stamp} SGT (data: {'Yahoo/yfinance' if live else 'CSV'}, delayed)"]
    if ybp is not None:
        lines.append(f"US10Y 10-session change: {ybp:+.0f}bp")
    for r in rows:
        on = [k for k, v in r["flags"].items() if v]
        hd = [k for k, v in r["hard"].items() if v]
        lines.append(
            f"\n[{r['level']}] {r['name']} {r['close']}  score {r['score']}/12"
            f"\n  RSI {r['rsi']} | MACD hist {r['hist']} | CMF {r['cmf']} | 20SMA {r['sma20']}"
            f"\n  40d high {r['hi40']} | neckline (20d low) {r['low20']} | ATR {r['atr']}"
            f"\n  signals: {', '.join(on) or '-'}" + (f"\n  HARD: {', '.join(hd)}" if hd else "")
        )
        re_on = [k for k, v in r["reentry"].items() if v]
        if r["reentry"]["WASHED_OUT"] or len(re_on) >= 3:
            tag = "CONFIRMED" if len(re_on) >= 4 else "watching"
            lines.append(f"  RE-ENTRY {tag} ({len(re_on)}/5): {', '.join(re_on) or '-'}")
    lines.append("\nGREEN 0-2 hold | AMBER 3-4 tighten stops | RED 5+ reduce | BREAK act on plan."
                 "\nVerify on broker terminal before acting.")
    text = "\n".join(lines)
    print(text)

    Path("output").mkdir(exist_ok=True)
    Path("output/latest.txt").write_text(text, encoding="utf-8")
    Path("output/latest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows, text


def notify(rows, text):
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    if not any(r["level"] in ("AMBER", "RED", "BREAK") for r in rows) and os.getenv("ALERT_ONLY", "1") == "1":
        return
    import urllib.parse
    import urllib.request
    data = urllib.parse.urlencode({"chat_id": chat, "text": text[:4000]}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=20)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir")
    ap.add_argument("--history", type=int, default=0)
    a = ap.parse_args()
    live = not a.csv_dir
    data = load_live(WATCHLIST) if live else load_csv(a.csv_dir, WATCHLIST)
    if not data:
        sys.exit("no data loaded")
    rows, text = run(data, live, a.history)
    notify(rows, text)

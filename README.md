# SGX Correction Early-Warning Scanner

Daily risk score (0–12) per counter for an impending sharp correction, built from the
signals that flagged the Oct 2026 DBS/OCBC/UOB sell-off 1–3 weeks early.

| Level | Score | Action |
|---|---|---|
| GREEN | 0–2 | Hold |
| AMBER | 3–4 | Tighten stops, no adding |
| RED | 5+ | Reduce / exit |
| BREAK | hard trigger | Close below 20-day low on 1.5x volume, or ≥2% gap-down — act on plan |

## Setup (≈10 min, free)
1. Create a GitHub repo and upload this folder (keep `.github/workflows/`).
2. Telegram alerts (optional): message @BotFather → `/newbot` → copy token.
   Message your bot once, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` to get your chat id.
   Repo → Settings → Secrets → Actions → add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
3. Actions tab → enable workflows → "SGX correction scan" → **Run workflow** to test.
4. It then runs every weekday 17:35 SGT. Reports land in `docs/latest.txt` and `docs/history/`
   (turn on GitHub Pages from `/docs` to view in a browser).

## Run locally
```
pip install -r requirements.txt
python correction_scanner.py              # live
python correction_scanner.py --history 40 # also show daily score history
```

## Maintain
- Edit `WATCHLIST` to add counters (BS6.SI, Z74.SI, U96.SI, BN4.SI …).
- Update `EARNINGS` dates each quarter.

Data is Yahoo Finance (delayed). Verify on your broker terminal before acting. Not financial advice.

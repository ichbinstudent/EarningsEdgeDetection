#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python3.12}"
# Options Basic provides 2 years historical data. With a 120d feature lookback,
# default to events safely after the two-year floor from this run date.
START="${START:-2024-10-01}"
END="${END:-2025-12-31}"
RATE_SLEEP="${RATE_SLEEP:-13}"
MIN_ROWS="${MIN_ROWS:-100}"
MODEL="${MODEL:-random_forest}"
TARGETS="${TARGETS:-beat_expected_move large_move direction_up}"
# Liquid option universe: broad mega-cap / index-heavy names to avoid the Investing.com OTC junk problem.
TICKERS="${TICKERS:-AAPL,MSFT,NVDA,AMZN,META,GOOGL,GOOG,TSLA,AVGO,AMD,NFLX,CRM,ORCL,ADBE,INTC,QCOM,CSCO,IBM,SHOP,UBER,ABNB,SNOW,PLTR,COIN,SMCI,JPM,BAC,GS,MS,V,MA,AXP,WMT,COST,HD,LOW,TGT,DIS,NKE,LLY,UNH,JNJ,MRK,PFE,XOM,CVX,CAT,BA,GE}"

mkdir -p data/models logs
LOG="logs/backtest_train_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "Backtest/train started: $(date -u --iso-8601=seconds)"
echo "Range: $START to $END"
echo "Rate sleep: ${RATE_SLEEP}s"
echo "Universe: $TICKERS"
echo "Log: $LOG"

"$PYTHON" polygon_backfill.py \
  --start "$START" \
  --end "$END" \
  --events-source investing \
  --tickers "$TICKERS" \
  --rate-sleep "$RATE_SLEEP"

ROWS=$("$PYTHON" - <<'PY'
import sqlite3
con=sqlite3.connect('data/earnings_ml.db')
cur=con.cursor()
print(cur.execute("""
    SELECT count(*) FROM snapshots
    WHERE actual_move_pct IS NOT NULL
      AND expected_move_pct IS NOT NULL
      AND collection_error IS NULL
""").fetchone()[0])
PY
)
echo "Usable labeled rows after backfill: $ROWS"

if [ "$ROWS" -lt "$MIN_ROWS" ]; then
  echo "Not training: need at least $MIN_ROWS usable labeled rows."
  exit 2
fi

for TARGET in $TARGETS; do
  OUT="data/models/${TARGET}_${MODEL}.joblib"
  EXTRA=()
  if [ "$TARGET" = "large_move" ]; then
    EXTRA=(--large-move-threshold 5)
  fi
  echo "Training $TARGET -> $OUT"
  "$PYTHON" train.py \
    --target "$TARGET" \
    --model "$MODEL" \
    --min-rows "$MIN_ROWS" \
    --output "$OUT" \
    "${EXTRA[@]}"
done

echo "Backtest/train complete: $(date -u --iso-8601=seconds)"

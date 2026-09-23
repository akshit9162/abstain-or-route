#!/bin/bash
# The six-run grid: probe / LoRA r=8 / full fine-tuning, each trained on English only or on a
# multilingual mix of the same size. Seed 0. Finished runs are skipped, so it is safe to re-run.
# Full fine-tuning runs first because it is the most memory-hungry.
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
run() {
  [ -f "results/$1/meta.json" ] && { echo "skip $1 (done)"; return; }
  shift; $PY -m src.train "$@"
}
run full_en_s0     --method full  --train en    --seed 0
run full_multi_s0  --method full  --train multi --seed 0
run probe_en_s0    --method probe --train en    --seed 0
run probe_multi_s0 --method probe --train multi --seed 0
run lora8_en_s0    --method lora --r 8 --train en    --seed 0
run lora8_multi_s0 --method lora --r 8 --train multi --seed 0
$PY -m src.evaluate

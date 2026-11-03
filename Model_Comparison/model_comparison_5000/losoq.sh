#!/bin/bash
# LOSO queue: runs leave-one-session-out for every finished 5000-draw fit that lacks it; waits for new fits.
# Usage: losoq.sh BUDGET_MIN model...
cd "$(dirname "$0")"; mkdir -p logs fits5000; PY=${PY:-python3}
BUDGET=$1; shift
T0=$(date +%s)
while true; do
  left=0; ran=0
  for m in "$@"; do
    [ -f "fits5000/$m.loso_s1.pkl" ] && continue
    left=1
    [ -f "fits5000/$m.diag.json" ] || continue
    EL=$(( ($(date +%s) - T0) / 60 ))
    if [ "$EL" -ge "$BUDGET" ]; then echo "LOSOQ_BUDGET_EXIT after ${EL} min (next: $m)"; exit 0; fi
    S=$(date +%s)
    OMP_NUM_THREADS=1 nice -n 5 $PY -I loso.py "$m" 1 fits5000 > "logs/loso5000_$m.log" 2>&1
    echo "$m loso exit=$? $(( ($(date +%s) - S) / 60 )) min $(grep '^{' logs/loso5000_$m.log | head -1)"
    ran=1
  done
  [ "$left" -eq 0 ] && { echo LOSOQ_DONE; exit 0; }
  EL=$(( ($(date +%s) - T0) / 60 ))
  if [ "$EL" -ge "$BUDGET" ]; then echo "LOSOQ_BUDGET_EXIT after ${EL} min (waiting for fits)"; exit 0; fi
  [ "$ran" -eq 0 ] && sleep 60
done

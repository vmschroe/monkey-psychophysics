#!/bin/bash
# Fit queue: fits each model with 4 x 5000 draws unless already done; stops starting new fits after
# BUDGET minutes so the job never gets cut off mid-fit. Usage: runq.sh BUDGET_MIN model...
cd "$(dirname "$0")"; mkdir -p logs fits5000; PY=${PY:-python3}
BUDGET=$1; shift
T0=$(date +%s)
for m in "$@"; do
  [ -f "fits5000/$m.diag.json" ] && continue
  EL=$(( ($(date +%s) - T0) / 60 ))
  if [ "$EL" -ge "$BUDGET" ]; then echo "FITQ_BUDGET_EXIT after ${EL} min (next: $m)"; exit 0; fi
  OMP_NUM_THREADS=2 $PY -I fit5000.py "$m" 5000 fits5000 > "logs/5000_$m.log" 2>&1
  echo "$m exit=$? $(date +%H:%M) $(grep '^{' logs/5000_$m.log | head -1)"
done
echo FITQ_DONE

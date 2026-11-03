#!/bin/bash
# Fit every model in turn (about 1.5 h on 2 cores), then the leave-one-session-out estimates and the tables.
cd "$(dirname "$0")"
mkdir -p fits results logs
for m in A B C B-wide C+L C+LK C+LP C+LKP C+Mmax C+Mdist C+LPS C+LKPS C+LMmax C+LPMmax; do
  python fit_model.py "$m" > "logs/$m.log" 2>&1 && python loso.py "$m" 1 >> "logs/$m.log" 2>&1
  echo "$m done"
done
python compare.py && python effects.py && python export_pointwise.py && python plot_comparison.py

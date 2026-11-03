#!/bin/bash
# Fit every model (4 chains x 5000 draws), leave-one-session-out for each, then tables, figures and the PDF.
# About 4 hours on 2 cores. Set PY to the Python of an environment with the repo's requirements plus nutpie,
# h5netcdf and statsmodels. The scripts find the repository two levels up (or set MP_REPO).
cd "$(dirname "$0")"; PY=${PY:-python3}; mkdir -p logs fits5000 results5000
MODELS="A B B-wide B-fixlapse C C+L C+LP C+LPS C+LK C+LKP C+LKPS C+Mmax C+Mdist C+LMmax C+LPMmax C+LPMdist D1 D2 D3 D0 D4 D+shared D+crit D+order D+crit+order D4+crit D3+crit"
for m in $MODELS; do
  [ -f "fits5000/$m.diag.json" ] || OMP_NUM_THREADS=2 $PY -I fit5000.py "$m" 5000 fits5000 > "logs/5000_$m.log" 2>&1
  [ -f "fits5000/$m.loso_s1.pkl" ] || OMP_NUM_THREADS=1 $PY -I loso.py "$m" 1 fits5000 > "logs/loso5000_$m.log" 2>&1
done
$PY -I count_params.py && $PY -I compare2.py fits5000 && $PY -I effects2.py fits5000 && $PY -I report_tables.py fits5000 \
  && $PY -I figures.py fits5000 && $PY -I export_pointwise5000.py fits5000
$PY -I rt_check.py > results5000/rt_check.txt
(cd report && pdflatex -interaction=nonstopmode report.tex > /dev/null && pdflatex -interaction=nonstopmode report.tex > /dev/null)

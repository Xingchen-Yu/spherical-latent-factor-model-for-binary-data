#!/usr/bin/env bash
# Rebuilds the test binaries from the repo's own C++ (extracted verbatim from ../source) and runs every check.
# Needs: g++ (C++17, OpenMP), curl, uv.  No R required.  ~10 minutes on a 12-core machine.
set -euo pipefail
cd "$(dirname "$0")"
ARMA=armadillo-14.0.3
if [ ! -d "$ARMA" ]; then curl -sL -o arma.tar.xz "https://sourceforge.net/projects/arma/files/$ARMA.tar.xz/download"; tar -xJf arma.tar.xz; rm arma.tar.xz; fi
PY="uv run --quiet --with numpy --with scipy --with jax --with mpmath python"
FLAGS="-O2 -std=c++17 -fopenmp -DARMA_DONT_USE_WRAPPER -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I $ARMA/include"
$PY build_harness.py && g++ $FLAGS harness.cpp -o harness
$PY build_geweke.py  && g++ $FLAGS geweke.cpp  -o geweke
export OMP_WAIT_POLICY=passive
mkdir -p logs
$PY test_gradients.py   2>&1 | grep -vi warning | tee logs/gradients.txt
$PY test_likelihood.py  2>&1 | grep -vi warning | tee logs/likelihood.txt
$PY test_sampler.py     2>&1 | grep -vi warning | tee logs/sampler.txt
$PY test_prior_var.py   2>&1 | grep -vi warning | tee logs/prior_variance.txt
$PY test_claims.py      2>&1 | grep -vi warning | tee logs/paper_claims.txt
STEP="0.05 0.25 0.05 0.25"   $PY test_geweke.py 3 4 5 400000 11 2>&1 | grep -vi warning | tee logs/geweke_p3.txt
STEP="0.05 0.25 0.05 0.25"   $PY test_geweke.py 5 4 5 400000 12 2>&1 | grep -vi warning | tee logs/geweke_p5.txt
STEP="0.02 0.10 0.02 0.10"   $PY test_geweke.py 6 5 4 600000 21 2>&1 | grep -vi warning | tee logs/geweke_p6.txt
STEP="0.015 0.06 0.015 0.06" $PY test_geweke.py 8 4 4 600000 22 2>&1 | grep -vi warning | tee logs/geweke_p8.txt

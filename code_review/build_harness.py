# Extract the pure C++ functions (lines up to gradient_no) verbatim from the repo's R wrapper and append a test harness.
import re, sys
src = open('../source/spherical_factor_model_wrapper.R').read()
body = src.split("RcppCode = '",1)[1]
body = body.split('// [[Rcpp::export]]\nList update_beta',1)[0]          # everything before the samplers
body = body.replace('#include <RcppArmadillo.h>','#include <armadillo>\n#include <cmath>\n#include <iostream>\n#include <string>')
body = body.replace('#include <Rmath.h>','').replace('using namespace Rcpp;','')
# impute_NA is exported between; keep it (pure arma)
harness = open('harness_main.cpp').read()
open('harness.cpp','w').write(body + '\n' + harness)

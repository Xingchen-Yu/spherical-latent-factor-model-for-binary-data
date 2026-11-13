# Full repo C++ (wrapper string + omp file), verbatim except: Rcpp headers removed, a 12-line stand-in for Rcpp::List added.
src = open('../source/spherical_factor_model_wrapper.R').read()
body = src.split("RcppCode = '",1)[1].rsplit("'",1)[0]
shim = r'''
#include <armadillo>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
struct List { std::vector<arma::mat> v;
  static arma::mat tm(const arma::mat& a){return a;} static arma::mat tm(double a){arma::mat m(1,1); m(0,0)=a; return m;}
  template<class... T> static List create(T... a){ List l; (l.v.push_back(tm(a)),...); return l; }
  arma::mat operator[](int i){return v[i];} };
'''
body = body.replace('#include <RcppArmadillo.h>',shim).replace('#include <Rmath.h>','').replace('using namespace Rcpp;','')
omp = open('../source/spherical_factor_model_omp.cpp').read()
for s in ['#include <RcppArmadillo.h>','#include <Rmath.h>','#include<functional>  ','#include <omp.h>','using namespace Rcpp;']: omp = omp.replace(s,'')
open('geweke.cpp','w').write(body + '\nnamespace omg {\n' + omp + '\n}\n' + open('geweke_main.cpp').read())

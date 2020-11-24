#include <RcppArmadillo.h>
#include <Rmath.h>
#include<functional>  
//#include <boost/math/special_functions/bessel.hpp>
#include <omp.h>
using namespace Rcpp;


// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::plugins(openmp)]]

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]] 

const int threads = 6;//for omega//
const double log_pi = log(M_PI);
const double log_2 = log(2.0);

double bessi0_exp(double x)
{
  double ax,ans;
  double y; 
  if ((ax=fabs(x)) < 3.75) { 
    y=x/3.75;
    y*=y;
    ans=(1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
                                            +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2))))))*exp(-ax);
  } else {
    y=3.75/ax;
    ans=(1/sqrt(ax))*(0.39894228+y*(0.1328592e-1
                                      +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
                                                                           +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
                                                                           +y*0.392377e-2))))))));
  }
  return ans;
}
double bessel_sum(arma::vec x){
  const int n = x.size();
  double total = 0;
  for(int i = 0; i < n; i++) {
    total += log(bessi0_exp(x[i]));
  }
  return total;
}

double dvm_log_scaled_sum(arma::vec a, arma::vec b, arma::vec omega_vec ){
  const int n = a.size();
  return -log_pi * n - bessel_sum(omega_vec) - sum(omega_vec % (2 * a / b));
}

double dvm_log_sum(double a, double b, double omega_1 ){
  return - log_2- log_pi - log(bessi0_exp(omega_1)) - omega_1 * (a / b + 1);
}

double dgamma_log (double x, double a, double b){
  return a * log(b) - lgamma(a) + (a - 1) * log(x) - b * x;
}

double likeliC_noj(arma::vec x, arma::vec omega_vec, int p){
  arma::vec x2 = square(x);
  arma::vec xsum = cumsum(x2);
  int p1 = p - 1;
  return dvm_log_scaled_sum(x2.subvec(2,p1),xsum.subvec(2,p1),omega_vec.subvec(1,p1-1)) + dvm_log_sum(x[1],pow(xsum[1],0.5),omega_vec[0]);
}

double likeli_omega_omp(arma::vec omega_vec, arma::mat beta,int nr,int p){
  double s_beta = 0;
  omp_set_num_threads(threads); 
  #pragma omp parallel for reduction(+:s_beta)
  
  for(int i = 0; i < nr ; i++){
    arma::vec temp = vectorise(beta.row(i));
    s_beta += likeliC_noj(temp,omega_vec,p);
  }
  return s_beta;
}

// [[Rcpp::export]]
List update_omega(arma::vec dummy, arma::vec omega_vec,arma::mat beta, int nr,double a, double b,double omega_sd,int p ){
  double omega = omega_vec[0];
  double eta = log(omega);
  double eta_new = omega_sd * as_scalar(arma::randn(1)) + eta;
  double omega_new = exp(eta_new);
  double accept;
  int omega_accept = 1;
  arma::vec omega_vec_new = omega_new * dummy;
  
  accept = exp(likeli_omega_omp(omega_vec_new,beta,nr,p) + eta_new  + dgamma_log(omega_new,a,b) -
    likeli_omega_omp(omega_vec,beta,nr,p) - eta - dgamma_log(omega,a,b));
  
  if(accept < as_scalar(arma::randu(1))){
    omega_vec_new = omega_vec;
    omega_accept = 0;
  }
  return List::create(omega_vec_new,omega_accept);
}

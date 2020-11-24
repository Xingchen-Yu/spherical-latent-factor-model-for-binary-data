RcppCode = '
#include <RcppArmadillo.h>
#include <Rmath.h>
#include<functional>  
//#include <boost/math/special_functions/bessel.hpp>
using namespace Rcpp;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]] 

const int MAXIT = 1000;
const double EPS = 3e-12;
const double FPMIN = 1e-30;
const double pi2 = pow(M_PI,2);
const double pi22 = 2 * pow(M_PI,2);
const double pi22_i = 1/pi22;
const double log_pi = log(M_PI);
const double log_2 = log(2.0);

double betaContFrac(double a, double b, double x) {
  double qab = a + b;
  double qap = a + 1;
  double qam = a - 1;
  double c = 1;
  double d = 1 - qab * x / qap;
  if (fabs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  double h = d;
  int m;
  for (m = 1; m <= MAXIT; m++) {
    int m2 = 2 * m;
    double aa = m * (b-m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (fabs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (fabs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= (d * c);
    aa = -(a+m) * (qab+m) * x / ((a+m2) * (qap+m2));
    d = 1 + aa * d;
    if (fabs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (fabs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    double del = d*c;
    h *= del;
    if (fabs(del - 1) < EPS) break;
  }
  return h;
}
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
double betaInc(double a, double b, double x) {
  if (x == 0)
    return 0;
  else if (x == 1)
    return 1;
  else {
    double logBeta = lgamma(a+b) - lgamma(a) - lgamma(b)+ a * log(x) + b * log(1-x);
    if (x < (a+1) / (a+b+2))
      return exp(logBeta) * betaContFrac(a, b, x) / a;
    else
      return 1 - exp(logBeta) * betaContFrac(b, a, 1-x) / b;
  }
}
double betaInc_log(double a, double b, double x) {
  if (x == 1)
    return 0;
  else {
    double logBeta = lgamma(a+b) - lgamma(a) - lgamma(b)+ a * log(x) + b * log(1-x);
    if (x < (a+1) / (a+b+2))
      //return exp(logBeta) * betaContFrac(a, b, x) / a;
      return logBeta + log(betaContFrac(a,b,x))-log(a);
    else
      return log(1 - exp(logBeta) * betaContFrac(b, a, 1-x) / b);
  }
}

double betaInc_log_lower(double a, double b, double x) {
  if (x == 0)
    return 0;
  else {
    double logBeta = lgamma(a+b) - lgamma(a) - lgamma(b)+ a * log(x) + b * log(1-x);
    if (1-x < (b+1) / (a+b+2))
      return logBeta + log(betaContFrac(b,a,1-x))-log(b);
    else
      return log(1 - exp(logBeta) * betaContFrac(a, b, x) / a);
  }
}
double bessel_sum(arma::vec x){
  const int n = x.size();
  double total = 0;
  for(int i = 0; i < n; i++) {
    total += log(bessi0_exp(x(i)));
  }
  return total;
}

double dvm_log_scaled_sum(arma::vec a, arma::vec b, arma::vec omega_vec ){
  const int n = a.size();
  return -log_pi * n - bessel_sum(omega_vec) - sum(omega_vec % (2 * a / b));
}


double dvm_log_sum(double a, double b, double omega_1 ){
  return - log_2 - log_pi - log(bessi0_exp(omega_1)) - omega_1 * (a / b + 1);
}

double sb_c(double x){
  return (x + pi2)/pi22;
}
arma::vec sb_c_vec(arma::vec x){
  return (x + pi2)/pi22;
}
double dgamma_log (double x, double a, double b){
  return a * log(b) - lgamma(a) + (a - 1) * log(x) - b * x;
}

arma::vec Rf_rbinom_vec(arma::vec x,arma::vec shape1,const int n){
  arma::vec kobe = arma::zeros(n);
  double temp,shape;
  for(int i=0 ;i < n; i++){
    shape = shape1(i);
    temp = betaInc(shape,shape,x(i));
    //kobe(i) = Rf_rbinom(1,temp);
	if(as_scalar(arma::randu(1))<= temp){
      kobe(i) = 1;
    }else{
      kobe(i) = 0;
    }
  }
  return kobe;
}
// [[Rcpp::export]]
arma::mat impute_NA(arma::uvec na, arma::uvec i_index, arma::uvec j_index, arma::mat ymat,
                    arma::mat tau_yes, arma::mat tau_no, arma::mat beta, arma::vec kappa, const int n){
  
  arma::mat beta_na = beta.rows(i_index);
  arma::mat yes_na = tau_yes.rows(j_index);
  arma::mat no_na = tau_no.rows(j_index);
  arma::vec kappa_na = kappa.rows(j_index);
  
  arma::vec impute = sb_c_vec(square(acos(sum(no_na % beta_na,1))) - square(acos(sum(yes_na % beta_na,1)))) ;
  ymat.elem(na) = Rf_rbinom_vec(impute, kappa_na,n);
  
  return ymat;
}

arma::vec gradient_pj(arma::vec omega_vec, arma::vec x, arma::mat jac_mat, arma::mat jac_mat2, int p ){
  const arma::vec x_2 = square(x);
  const arma::vec kobe = cumsum(x_2);
  const arma::vec kobe_2 = square(kobe);
  arma::vec prior_grad = arma::zeros(p);
  arma::vec jacobian_grad = arma::zeros(p);
  arma::vec temp = arma::zeros(p-1);
  const arma::vec temp_x = x.subvec(0,p-2);
  const arma::vec temp12 = { x(1),-x(0)};
  if(p>3){
    temp.subvec(2,p-2) = omega_vec.subvec(1,p-3) / kobe.subvec(2,p-2);
    prior_grad.subvec(0,p-2) = 4.0 *  (temp_x % (jac_mat2 * (omega_vec.subvec(1,p-3) %
    x_2.subvec(2,p-2)/ kobe_2.subvec(2,p-2)) - temp)  + omega_vec(p-2) * temp_x) ;
    jacobian_grad.subvec(0,p-2) = - jac_mat * (1.0 / kobe.subvec(1,p-2)) % temp_x;
  }else{
    prior_grad.subvec(0,1) = 4.0 * omega_vec(p-2) * temp_x;
    jacobian_grad.subvec(0,p-2) = - (1.0 / kobe(1)) * temp_x;
  }
  prior_grad.subvec(0,1) = prior_grad.subvec(0,1) + pow(kobe(1),-1.5) * omega_vec(0) * x(0) * temp12;
  
  return prior_grad + jacobian_grad;
}


double likeli_kappa(int nr, arma::vec asset,double shape,double kappa_a, double ccc,arma::vec ymat_col){
  arma::vec avec = arma::zeros(nr);
  
  for(int i = 0 ;i < nr; i++){
    if(ymat_col(i) == 1){
      avec(i) = betaInc_log(shape,shape,asset(i));
    }else{
      avec(i) = betaInc_log_lower(shape,shape,asset(i));
    }
  }
  return sum(avec) + dgamma_log(shape,kappa_a,ccc);
}

double likeliC(arma::vec x, arma::vec omega_vec, int p){
  arma::vec x2 = square(x);
  arma::vec xsum = cumsum(x2);
  int p1 = p - 1;
  return -0.5 * sum(log(xsum.subvec(1,p1))) + dvm_log_scaled_sum(x2.subvec(2,p1),xsum.subvec(2,p1),omega_vec.subvec(1,p1-1)) + dvm_log_sum(x(1),pow(xsum(1),0.5),omega_vec(0));
}


double likeli_beta (arma::vec omega_vec, arma::vec x, arma::mat tau_yes,
                    arma::mat tau_no,arma::vec kappa ,int nc, int p,arma::vec ymat_row){
  
  const arma::vec asset = square(acos(tau_no * x)) - square(acos(tau_yes * x));
  arma::vec avec = arma::zeros(nc);
  double temp,shape;
  
  for(int i = 0 ;i < nc; i++){
    shape = kappa(i);
    temp = sb_c(asset(i));
    if(ymat_row(i) == 1){
      avec(i) = betaInc_log(shape,shape,temp);
    }else{
      avec(i) = betaInc_log_lower(shape,shape,temp);
    }
  }
  return sum(avec) + likeliC(x,omega_vec,p);
}


double likeli_yes (arma::vec temp_no,arma::vec omega_vec, arma::vec x, arma::mat beta,
                   double shape,int nr, int p,arma::vec ymat_col){
  
  const arma::vec asset = temp_no - pow(acos(beta * x),2);
  arma::vec avec = arma::zeros(nr);
  double temp;
  
  for(int i = 0 ;i < nr; i++){
    temp = sb_c(asset(i));
    if(ymat_col(i) == 1){
      avec(i) = betaInc_log(shape,shape,temp);
    }else{
      avec(i) = betaInc_log_lower(shape,shape,temp);
    }
  }
  return sum(avec) + likeliC(x,omega_vec,p);
}

double likeli_no (arma::vec temp_yes,arma::vec omega_vec, arma::vec x, arma::mat beta,
                   double shape,int nr, int p,arma::vec ymat_col){
  
  const arma::vec asset = pow(acos(beta * x),2) - temp_yes;
  arma::vec avec = arma::zeros(nr);
  double temp;
  
  for(int i = 0 ;i < nr; i++){
    temp = sb_c(asset(i));
    if(ymat_col(i) == 1){
      avec(i) = betaInc_log(shape,shape,temp);
    }else{
      avec(i) = betaInc_log_lower(shape,shape,temp);
    }
  }
  return sum(avec) + likeliC(x,omega_vec,p);
}

arma::vec gradient_beta(arma::vec omega_vec, arma::vec x, arma::mat tau_yes, arma::mat tau_no,arma::vec kappa,
                        arma::mat jac_mat, arma::mat jac_mat2, int p,const int nc, arma::vec ymat_row){
  const arma::vec buffer = tau_no * x; //nc x 1 = [nc x p][p x 1]
  const arma::vec buffer2 = tau_yes * x; //nc x 1 = [nc x p][p x 1]
  const arma::vec a_buffer = acos(buffer); //nc x 1
  const arma::vec a_buffer2 = acos(buffer2); //nc x 1
  //this is faster then putting in below loop
  const arma::vec asset = square(a_buffer) - square(a_buffer2); //nc x 1
  const double p1 = p-1;
  arma::vec avec = arma::zeros(nc);//nc x 1
  arma::vec grad = arma::zeros(p);//p x 1
  double xp = x(p1);
  arma::mat xpt = x.head_rows(p1).t(); //1 x K
  double temp,shape,shape2,logbeta,check;
  for(int i = 0 ;i < nc; i++){
    shape = kappa(i);
    shape2 = 2 * shape;
    check = (shape + 1) / (shape2 +2);
    temp = sb_c(asset(i));
    if(ymat_row(i) == 1){
      if (temp < check){
        avec(i) = shape * pi22_i/(temp * (1-temp) * betaContFrac(shape,shape,temp));
      }else{
        logbeta = lgamma(shape) + lgamma(shape) - lgamma(shape2) - shape * (log(temp) + log(1-temp));
        avec(i) = pi22_i/( temp * (1-temp) *(exp(logbeta)- betaContFrac(shape,shape,1-temp)/shape));
      }
    }else{
      if (1 - temp < check){
        avec(i) = -shape * pi22_i/(temp * (1-temp) * betaContFrac(shape,shape,1-temp));
      }else{
        logbeta = lgamma(shape) + lgamma(shape) - lgamma(shape2) - shape * (log(temp) + log(1-temp));
        avec(i) =  pi22_i/( temp * (1-temp) *(betaContFrac(shape,shape,temp)/shape - exp(logbeta)));
      }
    }
  }
  grad.head_rows(p1) = (tau_yes.head_cols(p1) - tau_yes.col(p1)/xp * xpt).t() * (avec % a_buffer2/sqrt(1-square(buffer2))) -
   (tau_no.head_cols(p1) - tau_no.col(p1)/xp * xpt).t() * (avec % a_buffer/sqrt(1-square(buffer)));

  return 2 * grad + gradient_pj(omega_vec,x,jac_mat,jac_mat2,p);

}

arma::vec gradient_yes(arma::vec temp_no,arma::vec omega_vec, arma::vec x, arma::mat beta,double shape,
                        arma::mat jac_mat, arma::mat jac_mat2, int p,const int nr, arma::vec ymat_col){
  const arma::vec buffer2 = beta * x; //nr x 1 = [nc x p][p x 1]
  const arma::vec a_buffer2 = acos(buffer2); //nr x 1
  //this is faster then putting in below loop
  const arma::vec asset = temp_no - square(a_buffer2); //nr x 1
  const double p1 = p-1;
  arma::vec avec = arma::zeros(nr);//nr x 1
  arma::vec grad = arma::zeros(p);//p x 1
  double xp = x(p1);
  arma::mat xpt = x.head_rows(p1).t(); //1 x K
  double temp,shape2,logbeta,check;
  for(int i = 0 ;i < nr; i++){
    shape2 = 2 * shape;
    check = (shape + 1) / (shape2 +2);
    temp = sb_c(asset(i));
    if(ymat_col(i) == 1){
      if (temp < check){
        avec(i) = shape * pi22_i/(temp * (1-temp) * betaContFrac(shape,shape,temp));
      }else{
        logbeta = lgamma(shape) + lgamma(shape) - lgamma(shape2) - shape * (log(temp) + log(1-temp));
        avec(i) = pi22_i/( temp * (1-temp) *(exp(logbeta)- betaContFrac(shape,shape,1-temp)/shape));
      }
    }else{
      if (1 - temp < check){
        avec(i) = -shape * pi22_i/(temp * (1-temp) * betaContFrac(shape,shape,1-temp));
      }else{
        logbeta = lgamma(shape) + lgamma(shape) - lgamma(shape2) - shape * (log(temp) + log(1-temp));
        avec(i) =  pi22_i/( temp * (1-temp) *(betaContFrac(shape,shape,temp)/shape - exp(logbeta)));
      }
    }
  }
  grad.head_rows(p1) = (beta.head_cols(p1) - beta.col(p1)/xp * xpt).t() * (avec % a_buffer2/sqrt(1-square(buffer2)));
  
  return 2 * grad + gradient_pj(omega_vec,x,jac_mat,jac_mat2,p);
  
}


arma::vec gradient_no(arma::vec temp_yes,arma::vec omega_vec, arma::vec x, arma::mat beta,double shape,
                       arma::mat jac_mat, arma::mat jac_mat2, int p,const int nr, arma::vec ymat_col){
  const arma::vec buffer = beta * x; //nr x 1 = [nc x p][p x 1]
  const arma::vec a_buffer = acos(buffer); //nr x 1
  //this is faster then putting in below loop
  const arma::vec asset = square(a_buffer) - temp_yes; //nr x 1
  const double p1 = p-1;
  arma::vec avec = arma::zeros(nr);//nr x 1
  arma::vec grad = arma::zeros(p);//p x 1
  double xp = x(p1);
  arma::mat xpt = x.head_rows(p1).t(); //1 x K
  double temp,shape2,logbeta,check;
  for(int i = 0 ;i < nr; i++){
    shape2 = 2 * shape;
    check = (shape + 1) / (shape2 +2);
    temp = sb_c(asset(i));
    if(ymat_col(i) == 1){
      if (temp < check){
        avec(i) = shape * pi22_i/(temp * (1-temp) * betaContFrac(shape,shape,temp));
      }else{
        logbeta = lgamma(shape) + lgamma(shape) - lgamma(shape2) - shape * (log(temp) + log(1-temp));
        avec(i) = pi22_i/( temp * (1-temp) *(exp(logbeta)- betaContFrac(shape,shape,1-temp)/shape));
      }
    }else{
      if (1 - temp < check){
        avec(i) = -shape * pi22_i/(temp * (1-temp) * betaContFrac(shape,shape,1-temp));
      }else{
        logbeta = lgamma(shape) + lgamma(shape) - lgamma(shape2) - shape * (log(temp) + log(1-temp));
        avec(i) =  pi22_i/( temp * (1-temp) *(betaContFrac(shape,shape,temp)/shape - exp(logbeta)));
      }
    }
  }
  grad.head_rows(p1) = -(beta.head_cols(p1) - beta.col(p1)/xp * xpt).t() * (avec % a_buffer/sqrt(1-square(buffer)));
  
  return 2 * grad + gradient_pj(omega_vec,x,jac_mat,jac_mat2,p);
  
}

// [[Rcpp::export]]
List update_beta(int t, arma::vec nr_par, arma::vec epsilon_vec, arma::vec epsilon2_vec, arma::vec leap_vec, int nr, int nc,arma::vec omega_vec,
                      arma::mat beta,arma::mat tau_yes,arma::mat tau_no,arma::vec kappa,
                      arma::mat jac_mat, arma::mat jac_mat2,int p,arma::mat ymat){
  arma::mat dp = arma::eye(p,p);
  arma::vec nu = arma::zeros(p);
  arma::vec x_temp = arma::zeros(p);
  arma::vec x = arma::zeros(p);
  arma::vec x_prev = arma::zeros(p);
  double alpha, ae, cosat, sinat, h, h_new, accept;
  int count = 0;
  int start = nr_par(t);
  int end = nr_par(t+1);
  int e_s = end - start;
  arma::vec accept_chain = arma::ones(e_s);  
  arma::mat beta_out = arma::zeros(e_s,p);
  
  for(int i = start; i < end; i++,count++) {
    double epsilon = epsilon_vec(i);
    double epsilon2 = epsilon2_vec(i);
    int leap = leap_vec(i);
    
    arma::vec ymat_row = vectorise(ymat.row(i));
    x_prev = x = vectorise(beta.row(i));

    nu = arma::randn(p);
    nu = (dp - x * x.t()) * nu;
    h = likeli_beta(omega_vec,x,tau_yes,tau_no,kappa,nc,p,ymat_row) - as_scalar(0.5 * nu.t() * nu);
    for(int j = 0; j < leap; j++) {
      nu = nu + epsilon2 * gradient_beta(omega_vec, x, tau_yes, tau_no,kappa, jac_mat, jac_mat2, p, nc, ymat_row);
      nu = (dp - x * x.t()) * nu;

      alpha = norm(nu);
      ae = alpha * epsilon;
      cosat = cos(ae);
      sinat = sin(ae);
      x_temp = x;

      x = x * cosat + nu/alpha * sinat;
      x = x/norm(x);
      nu = nu * cosat - alpha * x_temp * sinat;

      nu = nu + epsilon2 * gradient_beta(omega_vec, x, tau_yes, tau_no,kappa, jac_mat, jac_mat2, p,nc,ymat_row);
      nu = nu = (dp - x * x.t()) * nu;
      
    }
    h_new = likeli_beta(omega_vec,x,tau_yes,tau_no,kappa,nc,p,ymat_row) - as_scalar(0.5 * nu.t() * nu);
    accept = exp(h_new - h);
    if(accept < as_scalar(arma::randu(1))){
      x = x_prev;
      e_s = e_s - 1;
      accept_chain(count) = 0;
    }
    beta_out.row(count) = x.t();
  }

  return List::create(beta_out,e_s,accept_chain);
}

// [[Rcpp::export]]
List update_yes(int t, arma::vec nc_par, arma::vec epsilon_vec, arma::vec epsilon2_vec, arma::vec leap_vec, int nr, int nc,arma::vec omega_tau_vec,
                      arma::mat beta,arma::mat tau_yes,arma::mat tau_no,arma::vec kappa,
                      arma::mat jac_mat, arma::mat jac_mat2,int p,arma::mat ymat){
  
  arma::mat dp = arma::eye(p,p);
  arma::vec nu = arma::zeros(p);
  arma::vec x_temp = arma::zeros(p);
  arma::vec x = arma::zeros(p);
  arma::vec x_prev = arma::zeros(p);
  double alpha, ae, cosat, sinat, h, h_new, accept;
  int count = 0;
  int start = nc_par(t);
  int end = nc_par(t+1);
  int e_s = end - start;
  arma::vec accept_chain = arma::ones(e_s);
  arma::mat yes_out = arma::zeros(e_s,p);

  for(int i = start; i < end; i++,count++) {
    double epsilon = epsilon_vec(i);
    double epsilon2 = epsilon2_vec(i);
    int leap = leap_vec(i);
    
    double kappa_j = kappa(i);
    arma::vec temp_no = square(acos(beta * tau_no.row(i).t()));
    arma::vec ymat_col= vectorise(ymat.col(i));
    x_prev = x = vectorise(tau_yes.row(i));

    nu = arma::randn(p);
    nu = (dp - x * x.t()) * nu;
    h = likeli_yes(temp_no,omega_tau_vec,x,beta,kappa_j,nr,p,ymat_col) - as_scalar(0.5 * nu.t() * nu);
    for(int j = 0; j < leap; j++) {
      nu = nu + epsilon2 * gradient_yes(temp_no,omega_tau_vec, x, beta,kappa_j, jac_mat, jac_mat2, p, nr, ymat_col);
      nu = (dp - x * x.t()) * nu;

      alpha = norm(nu);
      ae = alpha * epsilon;
      cosat = cos(ae);
      sinat = sin(ae);
      x_temp = x;

      x = x * cosat + nu/alpha * sinat;
      x = x/norm(x);
      nu = nu * cosat - alpha * x_temp * sinat;

      nu = nu + epsilon2 * gradient_yes(temp_no,omega_tau_vec, x, beta,kappa_j, jac_mat, jac_mat2, p,nr,ymat_col);
      nu = (dp - x * x.t()) * nu;

    }
    h_new = likeli_yes(temp_no,omega_tau_vec,x,beta,kappa_j,nr,p,ymat_col) - as_scalar(0.5 * nu.t() * nu);
    accept = exp(h_new - h);
    if(accept < as_scalar(arma::randu(1))){
      x = x_prev;
      e_s = e_s - 1;
      accept_chain(count) = 0;
    }
      yes_out.row(count) = x.t();
  }

  return List::create(yes_out,e_s,accept_chain);
}
// [[Rcpp::export]]
List update_no(int t, arma::vec nc_par,arma::vec epsilon_vec, arma::vec epsilon2_vec, arma::vec leap_vec, int nr, int nc,arma::vec omega_tau_vec,
               arma::mat beta,arma::mat tau_yes,arma::mat tau_no,arma::vec kappa,
               arma::mat jac_mat, arma::mat jac_mat2,int p,arma::mat ymat){
  arma::mat dp = arma::eye(p,p);
  arma::vec nu = arma::zeros(p);
  arma::vec x_temp = arma::zeros(p);
  arma::vec x = arma::zeros(p);
  arma::vec x_prev = arma::zeros(p);
  double alpha, ae, cosat, sinat, h, h_new, accept;
  int count = 0;
  int start = nc_par(t);
  int end = nc_par(t+1);
  int e_s = end - start;
  arma::vec accept_chain = arma::ones(e_s);
  arma::mat no_out = arma::zeros(e_s,p);
  
  for(int i = start; i < end; i++,count++) {
    double epsilon = epsilon_vec(i);
    double epsilon2 = epsilon2_vec(i);
    int leap = leap_vec(i);
    
    double kappa_j = kappa(i);
    arma::vec temp_yes = square(acos(beta * tau_yes.row(i).t()));
    arma::vec ymat_col= vectorise(ymat.col(i));
    x_prev = x = vectorise(tau_no.row(i));
    
    nu = arma::randn(p);
    nu = (dp - x * x.t()) * nu;
    h = likeli_no(temp_yes,omega_tau_vec,x,beta,kappa_j,nr,p,ymat_col) - as_scalar(0.5 * nu.t() * nu);
    for(int j = 0; j < leap; j++) {
      nu = nu + epsilon2 * gradient_no(temp_yes,omega_tau_vec, x, beta,kappa_j, jac_mat, jac_mat2, p, nr, ymat_col);
      nu = (dp - x * x.t()) * nu;
      
      alpha = norm(nu);
      ae = alpha * epsilon;
      cosat = cos(ae);
      sinat = sin(ae);
      x_temp = x;
      
      x = x * cosat + nu/alpha * sinat;
      x = x/norm(x);
      nu = nu * cosat - alpha * x_temp * sinat;
      
      nu = nu + epsilon2 * gradient_no(temp_yes,omega_tau_vec, x, beta,kappa_j, jac_mat, jac_mat2, p,nr,ymat_col);
      nu = (dp - x * x.t()) * nu;
      
    }
    h_new = likeli_no(temp_yes,omega_tau_vec,x,beta,kappa_j,nr,p,ymat_col) - as_scalar(0.5 * nu.t() * nu);
    accept = exp(h_new - h);
    if(accept < as_scalar(arma::randu(1))){
      x = x_prev;
      e_s = e_s - 1;
      accept_chain(count) = 0;
    }
    no_out.row(count) = x.t();
  }
  
  return List::create(no_out,e_s,accept_chain);
}  

// [[Rcpp::export]]
List update_kappa(int t,arma::vec nc_par, int nr,arma::mat beta, arma::mat tau_yes, arma::mat tau_no, 
    arma::vec kappa,arma::mat ymat, double kappa_a, double ccc, arma::vec t_sig_vec){
  int count = 0;
  int start = nc_par(t);
  int end = nc_par(t+1);
  int e_s = end - start;
  arma::vec accept_chain = arma::ones(e_s);
  arma::vec kappa_out = arma::zeros(e_s);
  
  for(int j = start; j< end; j++,count++ ){
    double t_sig = t_sig_vec(j);
    arma::vec asset = sb_c_vec(pow(acos(beta * tau_no.row(j).t()),2) - pow(acos(beta * tau_yes.row(j).t()),2));
    double kappa_last = kappa(j);
    double tt = log(kappa_last);
    double tt_new = t_sig * as_scalar(arma::randn(1)) + tt;
    double kappa_new = exp(tt_new);
    double accept;
    arma::vec ymat_col = ymat.col(j);
    
    accept = exp(likeli_kappa(nr,asset,kappa_new,kappa_a,ccc,ymat_col) + tt_new -
      likeli_kappa(nr,asset,kappa_last,kappa_a,ccc,ymat_col) - tt);
    
    if(accept < as_scalar(arma::randu(1))){
      kappa_new = kappa_last;
      e_s = e_s - 1;
      accept_chain(count) = 0;
    }
    kappa_out(count) = kappa_new;
  }
  return List::create(kappa_out,e_s,accept_chain);
}

// [[Rcpp::export]]
List waic_cpp(int t, arma::vec nc_par,int nr,arma::mat beta, arma::mat tau_yes, arma::mat tau_no, arma::vec kappa,arma::mat ymat){
  int start = nc_par(t);
  int end = nc_par(t+1);
  arma::mat amat = arma::zeros(nr,end-start);
  arma::mat amat_exp = amat;
  arma::mat amat_2 = amat;
  int count = 0;
  double temp = 0;
  for(int j = start; j< end; j++,count++){
    double kappa_j = kappa(j);
    arma::vec ymat_col = ymat.col(j);
    arma::vec asset = sb_c_vec(pow(acos(beta * tau_no.row(j).t()),2) - pow(acos(beta * tau_yes.row(j).t()),2));
    
    for(int i = 0 ;i < nr; i++){
      if(ymat_col(i) == 1){
        amat(i,count) = temp = betaInc_log(kappa_j,kappa_j,asset(i));
      }else{
        amat(i,count) = temp = betaInc_log_lower(kappa_j,kappa_j,asset(i));
      }
      amat_exp(i,count) = exp(temp);
      amat_2(i,count) = pow(temp,2);
    }  
  }
  return List::create(amat,amat_exp,amat_2);
} 


  // [[Rcpp::export]]
arma::mat predict(int t, arma::vec nc_par,int nr,arma::mat beta, arma::mat tau_yes, 
                  arma::mat tau_no,arma::vec kappa){
  int start = nc_par(t);
  int end = nc_par(t+1);
  int count = 0;
  arma::mat amat = arma::zeros(nr,end-start);
  double temp;
  for(int j = start; j< end; j++,count++){
    double kappa_j = kappa(j);
    arma::vec asset = sb_c_vec(pow(acos(beta * tau_no.row(j).t()),2) - pow(acos(beta * tau_yes.row(j).t()),2));
    
    for(int i = 0 ;i < nr; i++){
      temp = betaInc(kappa_j,kappa_j,asset(i));
      if(temp>=0.5){
        amat(i,count) = 1;
      }
    }  
  }
  
  return amat;
}

  // [[Rcpp::export]]
arma::mat predict_prob(int t, arma::vec nc_par,int nr,arma::mat beta, arma::mat tau_yes, 
                  arma::mat tau_no,arma::vec kappa){
  int start = nc_par(t);
  int end = nc_par(t+1);
  int count = 0;
  arma::mat amat = arma::zeros(nr,end-start);
  for(int j = start; j< end; j++,count++){
    double kappa_j = kappa(j);
    arma::vec asset = sb_c_vec(pow(acos(beta * tau_no.row(j).t()),2) - pow(acos(beta * tau_yes.row(j).t()),2));
    
    for(int i = 0 ;i < nr; i++){
      amat(i,count) = betaInc(kappa_j,kappa_j,asset(i));;
    }  
  }
  
  return amat;
  
}
'
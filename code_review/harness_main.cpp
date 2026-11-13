// ---------------- test harness (not part of the repo code) ----------------
using namespace std;
static arma::vec rdv(int n){ arma::vec v(n); for(int i=0;i<n;i++) cin>>v(i); return v; }
static arma::mat rdm(int r,int c){ arma::mat m(r,c); for(int i=0;i<r;i++) for(int j=0;j<c;j++) cin>>m(i,j); return m; }
static void pv(const char* tag, arma::vec v){ cout<<tag; cout.precision(17); for(unsigned i=0;i<v.n_elem;i++) cout<<" "<<v(i); cout<<"\n"; }

// GHMC loop copied line by line from update_beta, with target = likeliC (prior+Jacobian) and gradient = gradient_pj
static void sample_prior(int p, arma::vec omega_vec, arma::mat jac_mat, arma::mat jac_mat2, arma::vec x0,
                         int iters, int thin, double emin, double emax, int lmax, int vmf, double vk){
  arma::mat dp = arma::eye(p,p);
  arma::vec nu, x_temp, x = x0, x_prev;
  arma::vec mu = arma::zeros(p); mu(p-1) = 1;
  double alpha, ae, cosat, sinat, h, h_new, accept; long acc = 0;
  auto lp = [&](arma::vec z){ return vmf ? vk*arma::dot(mu,z) : likeliC(z,omega_vec,p); };
  auto gr = [&](arma::vec z){ return vmf ? arma::vec(vk*mu) : gradient_pj(omega_vec,z,jac_mat,jac_mat2,p); };
  for(int it=0; it<iters; it++){
    double epsilon = emin + (emax-emin)*arma::randu(); double epsilon2 = epsilon/2; int leap = 1 + (int)(arma::randu()*lmax);
    x_prev = x;
    nu = arma::randn(p);
    nu = (dp - x * x.t()) * nu;
    h = lp(x) - arma::as_scalar(0.5 * nu.t() * nu);
    for(int j = 0; j < leap; j++) {
      nu = nu + epsilon2 * gr(x);
      nu = (dp - x * x.t()) * nu;
      alpha = norm(nu); ae = alpha * epsilon; cosat = cos(ae); sinat = sin(ae);
      x_temp = x;
      x = x * cosat + nu/alpha * sinat;
      x = x/norm(x);
      nu = nu * cosat - alpha * x_temp * sinat;
      nu = nu + epsilon2 * gr(x);
      nu = (dp - x * x.t()) * nu;
    }
    h_new = lp(x) - arma::as_scalar(0.5 * nu.t() * nu);
    accept = exp(h_new - h);
    if(accept < arma::randu()){ x = x_prev; } else acc++;
    if(it % thin == 0) pv("S", x);
  }
  cout<<"ACC "<<(double)acc/iters<<"\n";
}

int main(){
  string mode; cin>>mode; int p; cin>>p;
  arma::mat jac_mat, jac_mat2;
  if(p>=4){ int r,c; cin>>r>>c; jac_mat = rdm(r,c); cin>>r>>c; jac_mat2 = rdm(r,c);} else { jac_mat = arma::zeros(1,1); jac_mat2 = jac_mat; }
  arma::vec omega = rdv(p-1);
  if(mode=="sample"){
    arma::vec x0 = rdv(p); int iters, thin, lmax, vmf, seed; double emin, emax, vk;
    cin>>iters>>thin>>emin>>emax>>lmax>>vmf>>vk>>seed; arma::arma_rng::set_seed(seed);
    sample_prior(p,omega,jac_mat,jac_mat2,x0,iters,thin,emin,emax,lmax,vmf,vk); return 0;
  }
  int T; cin>>T;
  for(int t=0;t<T;t++){
    int n; cin>>n;
    arma::vec x = rdv(p); arma::mat A = rdm(n,p), B = rdm(n,p); arma::vec kap = rdv(n), y = rdv(n);
    arma::vec z = rdv(p); double shape; cin>>shape;
    arma::vec one(1);
    one(0) = likeliC(x,omega,p); pv("likeliC", one);
    pv("gradient_pj", gradient_pj(omega,x,jac_mat,jac_mat2,p));
    // beta update: x is beta_i, A = tau_yes, B = tau_no
    one(0) = likeli_beta(omega,x,A,B,kap,n,p,y); pv("likeli_beta", one);
    pv("gradient_beta", gradient_beta(omega,x,A,B,kap,jac_mat,jac_mat2,p,n,y));
    // yes update: x is tau_yes_j, A plays beta (n subjects), z is tau_no_j
    arma::vec temp_no = square(acos(A * z));
    one(0) = likeli_yes(temp_no,omega,x,A,shape,n,p,y); pv("likeli_yes", one);
    pv("gradient_yes", gradient_yes(temp_no,omega,x,A,shape,jac_mat,jac_mat2,p,n,y));
    // no update: x is tau_no_j, z is tau_yes_j
    arma::vec temp_yes = square(acos(A * z));
    one(0) = likeli_no(temp_yes,omega,x,A,shape,n,p,y); pv("likeli_no", one);
    pv("gradient_no", gradient_no(temp_yes,omega,x,A,shape,jac_mat,jac_mat2,p,n,y));
  }
  return 0;
}

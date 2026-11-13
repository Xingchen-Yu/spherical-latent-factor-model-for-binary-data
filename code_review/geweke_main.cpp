// ---- Geweke (2004) successive-conditional test driver: y | params, then the repo's updates in the main script's order ----
using namespace std;
static arma::vec rdv(int n){ arma::vec v(n); for(int i=0;i<n;i++) cin>>v(i); return v; }
static arma::mat rdm(int r,int c){ arma::mat m(r,c); for(int i=0;i<r;i++) for(int j=0;j<c;j++) cin>>m(i,j); return m; }
int main(){
  int p,I,J,seed; long sweeps; int thin; cin>>p>>I>>J>>sweeps>>thin>>seed;
  double a,b,a_tau,b_tau,kappa_a,ccc_a,ccc_b, bmin,bmax,ymin,ymax; cin>>a>>b>>a_tau>>b_tau>>kappa_a>>ccc_a>>ccc_b>>bmin>>bmax>>ymin>>ymax;
  arma::mat jac_mat, jac_mat2; int r,c;
  if(p>=4){ cin>>r>>c; jac_mat=rdm(r,c); cin>>r>>c; jac_mat2=rdm(r,c);} else { jac_mat=arma::zeros(1,1); jac_mat2=jac_mat; }
  arma::mat beta=rdm(I,p), tau_yes=rdm(J,p), tau_no=rdm(J,p); arma::vec kappa=rdv(J); double omega,tau,ccc; cin>>omega>>tau>>ccc;
  arma::arma_rng::set_seed(seed);
  int K=p-1; arma::vec dummy(K); for(int k=0;k<K;k++) dummy(k)=(k+1.0)*(k+1.0);
  arma::vec omega_vec=omega*dummy, omega_tau_vec=tau*dummy;
  arma::vec nr_par={0.0,(double)I}, nc_par={0.0,(double)J};
  arma::mat ymat(I,J); arma::vec t_sig=arma::ones(J)*0.8; double omega_sd=0.3, omega_tau_sd=0.3;
  long accb=0,accy=0,accn=0,acck=0,acco=0,acct=0; cout.precision(10);
  for(long it=0; it<sweeps; it++){
    // 1. data given parameters (same probability function the repo uses for imputation)
    for(int j=0;j<J;j++){ arma::vec as = sb_c_vec(square(acos(beta*tau_no.row(j).t())) - square(acos(beta*tau_yes.row(j).t())));
      for(int i=0;i<I;i++) ymat(i,j) = (arma::randu() <= betaInc(kappa(j),kappa(j),as(i))) ? 1 : 0; }
    // 2. jittered tuning, independent of the state
    arma::vec delta(I), leap(I), dy(J), leapt(J);
    for(int i=0;i<I;i++){ delta(i)=bmin+(bmax-bmin)*arma::randu(); leap(i)=1+(int)(10*arma::randu()); }
    for(int j=0;j<J;j++){ dy(j)=ymin+(ymax-ymin)*arma::randu(); leapt(j)=1+(int)(10*arma::randu()); }
    // 3. the repo's updates, in the order of main_script
    List o = update_beta(0,nr_par,delta,delta/2,leap,I,J,omega_vec,beta,tau_yes,tau_no,kappa,jac_mat,jac_mat2,p,ymat); beta=o[0]; accb+=(long)o[1](0,0);
    o = update_yes(0,nc_par,dy,dy/2,leapt,I,J,omega_tau_vec,beta,tau_yes,tau_no,kappa,jac_mat,jac_mat2,p,ymat); tau_yes=o[0]; accy+=(long)o[1](0,0);
    o = update_no(0,nc_par,dy,dy/2,leapt,I,J,omega_tau_vec,beta,tau_yes,tau_no,kappa,jac_mat,jac_mat2,p,ymat); tau_no=o[0]; accn+=(long)o[1](0,0);
    o = update_kappa(0,nc_par,I,beta,tau_yes,tau_no,kappa,ymat,kappa_a,ccc,t_sig); kappa=arma::vectorise(o[0]); acck+=(long)o[1](0,0);
    o = omg::update_omega(dummy,omega_vec,beta,I,a,b,omega_sd,p); omega_vec=arma::vectorise(o[0]); acco+=(long)o[1](0,0);
    o = omg::update_omega(dummy,omega_tau_vec,arma::join_cols(tau_yes,tau_no),2*J,a_tau,b_tau,omega_tau_sd,p); omega_tau_vec=arma::vectorise(o[0]); acct+=(long)o[1](0,0);
    ccc = arma::randg(arma::distr_param(kappa_a*J+ccc_a, 1.0/(ccc_b+arma::accu(kappa))));   // R: rgamma(1,c_alpha,ccc_b+sum(kappa))
    if(it%thin==0){ cout<<omega_vec(0)<<" "<<omega_tau_vec(0)<<" "<<ccc<<" "<<kappa(0);
      for(int k=0;k<p;k++) cout<<" "<<beta(0,k); for(int k=0;k<p;k++) cout<<" "<<tau_yes(0,k); for(int k=0;k<p;k++) cout<<" "<<tau_no(J-1,k); cout<<"\n"; }
  }
  cerr<<"acc beta "<<(double)accb/sweeps/I<<" yes "<<(double)accy/sweeps/J<<" no "<<(double)accn/sweeps/J<<" kappa "<<(double)acck/sweeps/J<<" omega "<<(double)acco/sweeps<<" tau "<<(double)acct/sweeps<<"\n";
}

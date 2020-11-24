library(Rcpp)
library(circular)
library(snowfall)
library(wnominate)
########################################################################
source("./source/spherical_factor_model_wrapper.R") ## Wrapper for cpp code
source(file="./source/read_kh2.R")
source(file='./source/ymat_spit_2.R')
sourceCpp("./source/spherical_factor_model_omp.cpp") ## require openmp
sourceCpp(code = RcppCode)
####################################
core = 10 ### number of cores
sfInit(parallel=TRUE,cpus=core)
sfClusterSetupRNG(type="RNGstream")
cluster_seed = 88 + 1122 ### cluster seed for parallel
sfClusterSetupRNGstream(seed=cluster_seed)
sfLibrary("Rcpp", character.only=TRUE)
sfLibrary("RcppArmadillo", character.only=TRUE)
####################################
seed = 8888
continue_seed = seed + 1122
set.seed(seed)
data = 1 ### 1 is for House/Senate (100-116) Data, 2 is simulation or input your own data
House = T ### T if data = 1 and you want to run for House data 
test = F ### random split 5% to be test data if T
continue = T ### continue from the previous run
tunning = T ### Jittering for HMC and warmup for MH
save_to_continue = T  #### T if you'd like to run more
iter = 30000 ### Total iterations
burnin = 20000 ### Burnin period
###############################
p = 10 ### embedding dimension p >=3 , p=2 is the circular factor model (see circular factor model in the github)
K = p - 1 ### dimension of spheres
####HMC paramters######################
n_pos = iter - burnin  ### Number of posterior samples

if(p>=4){ ###pre-generate dummy matrix for automatic gradient construction
  jac_mat = jacobian_init(p) 
  jac_mat2 = jacobian_init2(p)
}else{
  jac_mat2 = jac_mat = matrix(0) ##dummy not used when p = 3
}
dummy = c(1:K)^2 ### penalizing constant for \omega and \omega_tau

start_tune = 51 ### tunning start
skip = 50 ### tune every 50 iterations

b_range = c(0.01,0.04) ### step size jittering range for \beta, if tunning is F, will be set to 0.015
yn_range = c(0.01,0.07) ### step size jittering range for \tau_yes(\psi) and \tau_no (\zeta), if tunning is F, will be set to 0.02
l_range = c(1,10) ### leap steps jittering range, if tunning is F, will be set to 10
###hyperparameter########################
##\omega_{\beta} ~ Gamma(a,b)
a = 1 
b = 1/10
##\omega_{\tau} ~ Gamma(a_tau,b_tau)
a_tau = 1
b_tau = 5
## \kappa_j ~ Gamma(kappa_a,ccc), ccc ~ Gamma(ccc_a,ccc_b)
ccc_a = 2
ccc_b = 150
kappa_a = 1

##MH proposal sd
omega_sd = 0.1
omega_tau_sd = 0.1

#################################################
if(data == 1){
  hn = 116
  h_s = ifelse(House==T,'H','S')
  out = ymat_spit(hn=hn,House) ####U.S Senate/House analysis
  ymat_true = ymat = out[[1]]
  pol = out[[2]]
  rm(out)
  dem = grep("\\(D",pol)
  gop = grep("\\(R",pol)
  ind = grep("\\(I",pol)
}else{
  hn = 8888
  h_s = 888
  load('./data/sim_2.Rdata') ##### change to your own data here, make sure it's in matrix form
  ymat_true = ymat
}

nr = nrow(ymat)
nc = ncol(ymat)

nr_nc = nr*nc
n_j = 2 * nc
n_param = nr + n_j

if(continue==F){
  ### initialization ###
  t_sig = rep(1,nc) ### initial proposal sd for \kappa_j
  ccc = ccc_a/ccc_b
  kappa = rep(0.1,nc)
  if(data == 1){
    beta1d = rep(0,nr)
    beta1d[dem] = runif(length(dem),pi/2,pi)
    beta1d[gop] = runif(length(gop),pi,pi*1.5)
  }else{
    beta1d = rvonmises(nr,pi,2)
  }
  omega_1d = 1
  omega_tau_vec = omega_vec = omega_1d * dummy

  x_beta = cbind(beta1d,sapply(omega_vec[-1],function(x) as.numeric(rvonmises(nr,pi,x)/2)))
  x_yes = cbind(runif(nc,0,2*pi),sapply(omega_tau_vec[-1],function(x) as.numeric(rvonmises(nc,pi,x)/2)))
  x_no = cbind(runif(nc,0,2*pi),sapply(omega_tau_vec[-1],function(x) as.numeric(rvonmises(nc,pi,x)/2)))
  # 
  tau_yes = getcord_auto(x_yes,K)
  tau_no = getcord_auto(x_no,K)
  beta = getcord_auto(x_beta,K)
  
  rm(beta1d)
}else{
  load(file=paste0("./continue/",h_s,hn,"_beta_start_",p,"d.Rdata"))
  load(file=paste0("./continue/",h_s,hn,"_tau_yes_start_",p,"d.Rdata"))
  load(file=paste0("./continue/",h_s,hn,"_tau_no_start_",p,"d.Rdata"))
  load(file=paste0("./continue/",h_s,hn,"_kappa_start_",p,"d.Rdata"))
  load(file=paste0("./continue/",h_s,hn,"_ccc_start_",p,"d.Rdata"))
  load(file=paste0("./continue/",h_s,hn,"_omega_start_",p,"d.Rdata"))
  load(file=paste0("./continue/",h_s,hn,"_omega_tau_start_",p,"d.Rdata"))
  load(file=paste0("./continue/",h_s,hn,"_tsig_start_",p,"d.Rdata"))
  load(file=paste0("./continue/",h_s,hn,"_omega_sd_",p,"d.Rdata"))
  load(file=paste0("./continue/",h_s,hn,"_omega_tau_sd_",p,"d.Rdata"))
}
if(test == T){
  ###creating 5% test set
  na_ori_true = which(is.na(ymat==T))
  na.position = which(is.na(ymat)==T, arr.ind = T)
  for(j in 1:nc){
    tom = na.position[which(na.position[,2]==j),1]
    if(length(tom)>0){
      ymat[sample(c(1:nr)[-tom],ceiling(0.05 * (nr-length(tom)))),j] = NA ###5% test set if test = T
    }else{
      ymat[sample(c(1:nr),ceiling(0.05 * nr)),j] = NA
    }
  }
  temp_na = which(is.na(ymat==T))
  na_test = temp_na[which(temp_na %!in% na_ori_true)]
  len_test = length(na_test)
}
impute = any(is.na(ymat)) ### T if there is missing value

if(tunning==T){
  ### initialization for leap size jittering
  leap = sample(l_range[1]:l_range[2],nr,replace=T)
  leap_tau = sample(l_range[1]:l_range[2],nc,replace=T)
  
  delta = runif(nr,b_range[1],b_range[2])
  delta_yes = delta_no = runif(nc,yn_range[1],yn_range[2])
  
}else{
  delta = rep(0.015,nr) ###default step size for \beta without jittering 
  delta_no = delta_yes = rep(0.02,nc) ### default step size for \tau_yes (\psi), \tau_no(\zeta) without jittering 
  leap = rep(10,nr)  ### default leap steps without jittering
  leap_tau = rep(10,nc) ### default leap steps without jittering
}
if(continue==T){
  set.seed(continue_seed)
}
delta2 = delta/2
delta2_yes = delta_yes/2
delta2_no = delta_no/2

##################################
c_alpha = kappa_a*nc+ccc_a
##################################
na = which(is.na(ymat==T)) -1
len_na = length(na) 
no_na = which(is.na(ymat)==F)
len_no_na = length(no_na)

na.position = which(is.na(ymat)==T, arr.ind = T)
i_index = as.numeric(na.position[,1]) - 1
j_index = as.numeric(na.position[,2]) - 1
#############
nr_par = round(seq(0,nr,length.out = core+1))
nc_par = round(seq(0,nc,length.out = core+1))
if(impute==T){
  ymat = impute_NA(na, i_index, j_index, ymat, tau_yes, tau_no, beta, kappa, len_na)
}
sfExportAll(except=list('ymat_true','na_ori_true','na.position','tom','na_test','len_test',
                        'dem','gop','pol','no_na','len_no_na'))
if(test==T){
  y_test = ymat_true[na_test]
}
rm(ymat_true)

if(n_pos>0){
  pos_pred = pos_pred2 = pos_pred3 = matrix(0,nr,nc)
  beta_rs = matrix(0,nr,p)
  no_rs = yes_rs = matrix(0,nc,p)
  omega_master = ccc_master = rep(0,n_pos)
  omega_tau_master = rep(0,n_pos)
}

likeli_chain = rep(0,iter) ###joint likelihood
###################################

wrapper_beta = function(t){
  update_beta(t,nr_par, delta, delta2, leap, nr,nc,omega_vec,beta,tau_yes,tau_no,kappa,jac_mat,jac_mat2,p,ymat )
}
wrapper_yes = function(t){
  update_yes(t,nc_par,delta_yes,delta2_yes,leap_tau,nr,nc,omega_tau_vec,beta,tau_yes,tau_no,kappa,jac_mat,jac_mat2,p,ymat)
}
wrapper_no = function(t){
  update_no(t,nc_par,delta_no,delta2_no,leap_tau,nr,nc,omega_tau_vec,beta,tau_yes,tau_no,kappa,jac_mat,jac_mat2,p,ymat)
}
wrapper_kappa = function(t){
  update_kappa(t,nc_par,nr,beta,tau_yes,tau_no,kappa,ymat,kappa_a,ccc,t_sig)
}
wrapper_waic = function(t){
  waic_cpp(t,nc_par,nr,beta,tau_yes,tau_no,kappa,ymat)
}
wrapper_predict = function(t){
  predict(t,nc_par,nr,beta,tau_yes,tau_no,kappa)
}
wrapper_predict2 = function(t){
  predict(t,nc_par,nr,beta_pred,yes_pred,no_pred,kappa_mean)
}
wrapper_predict3 = function(t){
  predict(t,nc_par,nr,beta_pred,yes_pred,no_pred,kappa)
}
wrapper_predict_beta = function(t){
  predict(t,nc_par,nr,beta_pred,yes_mean,no_mean,kappa_mean)
}

sfClusterEval(sourceCpp(code = RcppCode))
core_1 = core - 1
node = 0:core_1
j = 1
beta_ratio = yes_ratio = no_ratio = kappa_ratio = omega_count = 0
omega_count_all = omega_count = 0

omega_tau_count = 0
omega_tau_count_all = omega_tau_count = 0

beta_accept_rs_all = rep(0,nr)

kappa_accept_rs = rep(0,nc)
kappa_accept_rs_all = no_accept_rs_all = yes_accept_rs_all = rep(0,nc)

stopifnot((len_no_na+len_na)==nr*nc)

for(i in 1:iter){

  ###jittering and Adaptive MH for tunning, the adaptive process for MH stops after burnin
  if(tunning == T){
    if(i %in% seq(start_tune,iter,skip)){
      
      leap = sample(l_range[1]:l_range[2],nr,replace=T)
      leap_tau = sample(l_range[1]:l_range[2],nc,replace=T)

      delta = runif(nr,b_range[1],b_range[2])
      delta_yes = delta_no = runif(nc,yn_range[1],yn_range[2])
      #######################################
      delta2 = delta/2
      delta2_yes = delta_yes/2
      delta2_no = delta_no/2
      
      if(i<burnin){
        ks = kappa_accept_rs/skip
        kappa_skip = min(ks)
        
        out = update_tsig(0.6,0.3,t_sig,ks,nc)
        t_sig = out[[1]]
        kappa_mod = out[[2]]
        
        kappa_accept_rs = rep(0,nc)
        
        os = omega_count/skip
        omega_sd = update_os(0.6,0.3,omega_sd,os)
        print(paste0('omega sd changed to ',omega_sd))
        omega_count = 0 
        
        os = omega_tau_count/skip
        omega_tau_sd = update_os(0.6,0.3,omega_tau_sd,os)
        print(paste0('omega tau sd changed to ',omega_tau_sd))
        omega_tau_count = 0 
        sfExport("omega_tau_sd")

        print(paste0('percent kappa changed ',kappa_mod))
        print(paste0('min kappa acceptance ',kappa_skip))
      }
      
      sfExport("delta",'delta2','delta_yes','delta2_yes','delta_no','delta2_no','t_sig','leap','leap_tau','omega_sd')
    }
  }
  ### GHMC for \beta
  out = sfLapply(node,wrapper_beta)
  beta = do.call('rbind',lapply(out,"[[",1))
  beta_ratio = sum(unlist(lapply(out,"[[",2)))/nr
  haha = unlist(lapply(out,"[[",3))
  beta_accept_rs_all = beta_accept_rs_all  + haha
  sfExport("beta")
  ### GHMC for \tau_yes (\psi)
  out = sfLapply(node,wrapper_yes)
  tau_yes = do.call('rbind',lapply(out,"[[",1))
  yes_ratio = sum(unlist(lapply(out,"[[",2)))/nc
  haha = unlist(lapply(out,"[[",3))
  yes_accept_rs_all = yes_accept_rs_all  + haha
  sfExport("tau_yes")
  ### GHMC for \tau_no (\zeta)
  out = sfLapply(node,wrapper_no)
  tau_no = do.call('rbind',lapply(out,"[[",1))
  no_ratio = sum(unlist(lapply(out,"[[",2)))/nc
  haha = unlist(lapply(out,"[[",3))
  no_accept_rs_all = no_accept_rs_all  + haha
  sfExport("tau_no")
  ### GHMC for \kappa (scale parameter for the link function)
  out = sfLapply(node,wrapper_kappa)
  kappa = unlist(lapply(out,"[[",1))
  kappa_ratio = sum(unlist(lapply(out,"[[",2)))/nc
  haha = unlist(lapply(out,"[[",3))
  kappa_accept_rs = kappa_accept_rs  + haha
  kappa_accept_rs_all = kappa_accept_rs_all  + haha
  sfExport("kappa")
  ### MH for \omega (require open mp)
  out = update_omega(dummy,omega_vec,beta,nr,a,b,omega_sd,p)
  omega_vec = out[[1]]
  haha = out[[2]]
  omega_count_all = omega_count_all + haha
  omega_count = omega_count + haha
  ### MH for \omega_tau (require open mp)
  out = update_omega(dummy,omega_tau_vec,rbind(tau_yes,tau_no),n_j,a_tau,b_tau,omega_tau_sd,p)
  omega_tau_vec =  out[[1]]
  haha = out[[2]]
  omega_tau_count_all = omega_tau_count_all + haha
  omega_tau_count = omega_tau_count + haha
    

  sfExport('omega_vec','omega_tau_vec')
  ### Conjugate update for ccc, \kappa_j ~ Gamma(kappa_a,ccc), ccc ~ Gamma(ccc_a,ccc_b)
  ccc = rgamma(1,c_alpha,ccc_b+sum(kappa))
  sfExport("ccc")

  dic_out = sfLapply(node,wrapper_waic)
  temp = do.call("cbind",lapply(dic_out,"[[",1))

  likeli_chain[i] = sum(temp[no_na])
  
  if(i>burnin){
    beta_rs = beta_rs + beta
    yes_rs = yes_rs + tau_yes
    no_rs = no_rs + tau_no
    omega_master[j] = omega_vec[1]
    omega_tau_master[j] = omega_tau_vec[1]
  
    ccc_master[j] = ccc

    pos_pred = pos_pred + do.call("cbind",lapply(dic_out,"[[",2))
    pos_pred2 = pos_pred2 + temp
    pos_pred3 = pos_pred3 + do.call("cbind",lapply(dic_out,"[[",3))
    
    j = j + 1
  }
  if(i %in% seq(1,iter,50)){
    cat("\rProgress: ",i,"/",iter)
    
    y_hat = do.call("cbind", sfLapply(node,wrapper_predict))
    print(paste('trainning accuracy is',length(which(y_hat[no_na]==ymat[no_na]))/len_no_na))

    print(paste0('ll = ',round(sum(temp),0)))

    if(impute==T){
      ymat = impute_NA(na, i_index, j_index, ymat, tau_yes, tau_no, beta, kappa, len_na)
      sfExport('ymat')
    }
    print(paste('beta ar is',beta_ratio))
    print(paste('yes ar is',yes_ratio))
    print(paste('no ar is',no_ratio))
    print(paste('kappa ar is',kappa_ratio))
    print(paste('omega ar is',omega_count_all/i))
    print(paste('min kappa ar is',min(kappa_accept_rs_all/i)))
    print(paste('min beta ar is',min(beta_accept_rs_all/i)))
    print(paste('min yes ar is',min(yes_accept_rs_all/i)))
    print(paste('min no ar is',min(no_accept_rs_all/i)))
    
  }
}
if(save_to_continue==T){
  save(file=paste0("./continue/",h_s,hn,"_beta_start_",p,"d.Rdata"),beta)
  save(file=paste0("./continue/",h_s,hn,"_tau_yes_start_",p,"d.Rdata"),tau_yes)
  save(file=paste0("./continue/",h_s,hn,"_tau_no_start_",p,"d.Rdata"),tau_no)
  save(file=paste0("./continue/",h_s,hn,"_kappa_start_",p,"d.Rdata"),kappa)
  save(file=paste0("./continue/",h_s,hn,"_ccc_start_",p,"d.Rdata"),ccc)
  save(file=paste0("./continue/",h_s,hn,"_omega_start_",p,"d.Rdata"),omega_vec)
  save(file=paste0("./continue/",h_s,hn,"_omega_tau_start_",p,"d.Rdata"),omega_tau_vec)
  save(file=paste0("./continue/",h_s,hn,"_tsig_start_",p,"d.Rdata"),t_sig)
  save(file=paste0("./continue/",h_s,hn,"_omega_sd_",p,"d.Rdata"),omega_sd)
  save(file=paste0("./continue/",h_s,hn,"_omega_tau_sd_",p,"d.Rdata"),omega_tau_sd)
}else{
  ###unscaled DIC (without multiplying -2 constant)
  DIC_out = dic_compute(n_pos,pos_pred,pos_pred2,pos_pred3,no_na,likeli_chain,burnin+1,iter)
  DIC = DIC_out[1]-DIC_out[3] ## correspond to unscaled DIC_alt in page 8 of http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf
  save.image(file=paste0(h_s,hn,'_',p,"_nd_workspace.Rdata"))
}


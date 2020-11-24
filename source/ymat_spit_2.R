'%!in%' <- function(x,y)!('%in%'(x,y))
z_to_x<-function(z){
  pi2<-pi^2
  pi22<-2*pi2
  x<-(z+pi2)/pi22
  return(x)
}
ymat_spit<-function(hn,House){
  if(House==T){
    vote2<-readKH2(file=paste0("./data/H",hn,"_votes.ord"))
  }else{
    vote2<-readKH2(file=paste0("./data/S",hn,"_votes.ord"))
  }
  if(hn==116 & House==T){
    vote<-as.matrix(vote2$votes)[,1:700]
    ####Amash switches party at 430, we combine the record for Amash together####
    vote[202,430:700] = vote[203,430:700]
    vote = vote[-203,]
  }else{
    vote<-as.matrix(vote2$votes)
  }
  pol<-rownames(vote)
  ####0,7,8,9 represents NA###
  ind3<-which(vote==0 | vote==7 | vote==8| vote==9)
  vote[ind3]<-NA
  ####1,2,3 represents Yea####
  ind1<-which(vote==1 | vote==2 | vote==3)
  vote[ind1]<-1
  ####4,5,6 represents Nah####
  ind2<-which(vote==4 | vote==5| vote==6)
  vote[ind2]<-0
  #####legislators who miss more than 40% of the vote is excluded####
  abs_percent<-as.numeric(apply(vote,1,function(xx) length(which(is.na(xx)==T))))/ncol(vote)
  abs_ind<-which(abs_percent>=0.4)
  if(length(abs_ind)>0){
    pol<-pol[-abs_ind]
    ymat<-vote[-abs_ind,]
  }else{
    ymat<-vote
  }
  colnames(ymat)<-NULL
  return(list(ymat,pol))
}
qf3<-function(tau_yes,tau_no,beta,kappa){
  buffer<-tcrossprod(beta,tau_yes)
  buffer2<-tcrossprod(beta,tau_no)
  asset<-acos(buffer2)^2-acos(buffer)^2
  kappa_mat<-matrix(kappa,nr,nc,byrow=T)
  x<-z_to_x(asset)
  sup<-pbeta(x,kappa_mat,kappa_mat)
  return(sup)
}
ymat_sim = function(K,nr,nc,omega_tau_vec,omega_vec,kappa_a,kappa_b,hard_zero,model_index){

  if(hard_zero==T){ 
    x_beta = cbind( circular::rvonmises(nr,pi,omega_vec[1]),
                    sapply(omega_vec[c(2:model_index)],function(x) circular::rvonmises(nr,pi,x)/2),
                    matrix(pi/2,nr,K - model_index))
    x_yes = cbind(circular::rvonmises(nc,pi,omega_tau_vec[1]),
                  sapply(omega_tau_vec[c(2:model_index)],function(x) circular::rvonmises(nc,pi,x)/2),
                  matrix(pi/2,nc,K - model_index))
    x_no = cbind(circular::rvonmises(nc,pi,omega_tau_vec[1]),
                 sapply(omega_tau_vec[c(2:model_index)],function(x) circular::rvonmises(nc,pi,x)/2),
                 matrix(pi/2,nc,K - model_index))
  }else{
    x_beta = cbind( circular::rvonmises(nr,pi,omega_vec[1]),sapply(omega_vec[-1],function(x) circular::rvonmises(nr,pi,x)/2))
    x_yes = cbind(circular::rvonmises(nc,pi,omega_tau_vec[1]),sapply(omega_tau_vec[-1],function(x) circular::rvonmises(nc,pi,x)/2))
    x_no = cbind(circular::rvonmises(nc,pi,omega_tau_vec[1]),sapply(omega_tau_vec[-1],function(x) circular::rvonmises(nc,pi,x)/2))
  }
  tau_yes = getcord_auto(x_yes,K)
  tau_no =  getcord_auto(x_no,K)
  beta = getcord_auto(x_beta,K)
  
  kappa = rgamma(nc,kappa_a,kappa_b)
  prob = qf3(tau_yes,tau_no,beta,kappa)
  ymat = ymat_true = matrix(0,nr,nc)
  for(i in 1:nr){
    for(j in 1:nc){
      temp = prob[i,j]
      ymat[i,j] = ymat_true[i,j] = sample(c(1,0),1,prob=c(temp,1-temp))
    }
  }
  
  list(ymat,x_beta,x_yes,x_no,kappa)
}
jacobian_init2<-function(p){
  kobe<-matrix(c(1,1,1))
  if(p<4){
    print('error')
  }else if(p>4){
    for(i in 5:p){
      kobe=cbind(rbind(kobe,0),1)
    }
  }
  return(kobe)
}

jacobian_init<-function(p){
  kobe = matrix(c(1,1))
  for(i in 4:p){
    kobe= cbind(rbind(kobe,0),1)
  }
  return(kobe)
}
getcord_auto<-function(test,d){
  sup = 1
  for(i in 1:d){
    nam <- paste("a", i, sep = "")
    assign(nam, test[,i] )
    sup =cbind(sup * sin(get(nam)),cos(get(nam)))
  }
  
  return(sup)
}

spit_omega_true<-function(w,index){
  return(c(rep(1,index),rep(c_shrink,K - index))*w)
}

wrapper_variance<-function(xxx){
  nnn<-length(xxx)
  rrr<-sqrt(sum(cos(xxx))^2+sum(sin(xxx))^2)/nnn
  vvv<-1-rrr
  return(vvv)
}
x_to_theta<-function(K,n_row,x_mat){
  theta_mat<-matrix(0,n_row,K)
  theta_mat[,1]<-atan2_new(x_mat[,2],x_mat[,1])
  for(i in 2:K){
    theta_mat[,i]<-atan2(rowSums((x_mat[,1:i]^2))^0.5,x_mat[,i+1])
  }
  theta_mat 
}

atan2_new<-function(x3,x4){
  temp = acos(x3/sqrt(x3^2+x4^2))
  sup = ifelse(x4>=0,temp,2*pi-temp)
  sup
}
waic_compute = function(nnn,pos_pred,pos_pred2,pos_pred3,no_na){
  pos_pred_master<-pos_pred[no_na]/nnn
  lpd<-sum(log(pos_pred_master))
  va<-sum((pos_pred3[no_na]/nnn-(pos_pred2[no_na]/nnn)^2))*nnn/(nnn-1)
  waic_spherical<-lpd-va
  return(waic_spherical)
}
atan2_kobe = function(x){
  y = apply(cbind(sin(x),cos(x)),2,mean)
  atan2_new(y[2],y[1])
}


update_tsig = function(upper,lower,t_sig,check,nnn){
  l_s = which(check<=lower)
  u_s = which(check>=upper)
  
  t_sig[l_s] = t_sig[l_s] * 0.95
  t_sig[u_s] = t_sig[u_s] * 1.05
  return(list(t_sig,length(c(l_s,u_s))/nnn))
}
update_os = function(upper,lower,omega_sd,os){
  if(os>upper | os<lower){
    omega_sd = ifelse(os>upper,1.2*omega_sd,0.8*omega_sd)
  }
  return(omega_sd)
}
###DIC computation, do not include NA 
dic_compute = function(nnn,pos_pred,pos_pred2,pos_pred3,no_na,likeli_chain,start,end){
  pos_pred_master = pos_pred[no_na]/nnn
  lpd = sum(log(pos_pred_master))
  penalty = 2 * (lpd - sum(pos_pred2[no_na]/nnn))
  penalty_dic_alt = 2 * var(likeli_chain[start:end],na.rm = T)
  return(c(lpd,penalty,penalty_dic_alt))
}

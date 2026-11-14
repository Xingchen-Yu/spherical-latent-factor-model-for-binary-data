# Spherical Latent Factor Model for Binary Data
This repo includes the implementation of spherical latent factor model for binary data https://arxiv.org/pdf/2008.05109.pdf.

During the implementation, we used the traditional coordinate systems https://en.wikipedia.org/wiki/N-sphere which has a one-to-one correspondance
with the one mentioned in the paper. The correspondance can be achieved by postprocessing.(eg., substrating the first angle by $\pi$ and the rest of angles by 0.5\pi)



 

`main_script/Euclidean_nd.R` (with `source/ymat_spit.R`) is the Euclidean probit factor model used as the comparison in the paper. Its prior on the latent positions has precision k^2 on dimension k (`p_mat = diag((1:p)^2)`), the Euclidean counterpart of the spherical von Mises prior's shrinkage.

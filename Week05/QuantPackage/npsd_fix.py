import numpy as np
from scipy import integrate
class NPSD:
    
    def __init__(self, eps=0):
        self.eps = eps
    
    def chol_psd(self, sigma):
        n = sigma.shape[0]
        root = np.zeros((n,n))
        for j in range(n):
            s = root[j,:j]
            diff = sigma[j,j] - np.sum(s*s)
            if diff <=0 and diff>= -1e-8:
                diff = 0
            root[j,j] = np.sqrt(diff)
            if root[j,j] == 0:
                root[j,(j+1):n] = 0
            else:
                ir = 1/root[j,j]
                for i in range(j+1,n):
                    if j ==0:
                        s = 0
                    else:
                        s = np.sum(root[i,:j]*root[j,:j])
                    root[i,j] = (sigma[i,j]-s)*ir
        return root

    def near_psd(self, sigma):
        # calculate the correlation matrix
        n = sigma.shape[0]
        inv_var = np.zeros((n,n))
        for i in range(n):
            inv_var[i,i] = 1/sigma[i,i]
        corr = inv_var@sigma@inv_var

        # SVD, update the eigen value and scale
        vals,vecs = np.linalg.eig(corr)
        vals[vals<self.eps] = self.eps
        T = 1/ (vecs*vecs@vals)
        T = np.diag(np.sqrt(T).tolist())
        l = np.diag(np.sqrt(vals).tolist())
        B = T@vecs@l
        out = B@B.T

        # back to the variance matrix
        var_mat = np.zeros((n,n))
        for i in range(n):
            var_mat[i,i] = sigma[i,i]
        cov = var_mat@out@var_mat

        return cov

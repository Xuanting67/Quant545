
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
import time

from scipy import integrate
class PortfolioCovariance:
    
    def __init__(self):
        pass
    
    def gen_weight(self, lam, X):
        w = np.array([(1-lam)*(lam**i) for i in range(X.shape[0])])
        w_scale = w/sum(w)
        return w_scale
    
    def weight_cov(self, lam, X):
        cov = np.zeros((X.shape[1], X.shape[1]))
        w = self.gen_weight(lam, X)
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                if i == j:
                    m = X[:, i]
                    scale_m = (m-m.mean())**2
                    cov[i, j] = np.sum(w*scale_m)
                if i != j:
                    m = X[:, i]
                    n = X[:, j]
                    scale_m = (m-m.mean())
                    scale_n = (n-n.mean())
                    cov[i, j] = np.sum(w*scale_m*scale_n)
                    cov[j, i] = cov[i, j]
        return cov
    
    def near_psd(self, sigma, eps):
        # calculate the correlation matrix
        n = sigma.shape[0]
        inv_var = np.zeros((n,n))
        for i in range(n):
            inv_var[i,i] = 1/sigma[i,i]
        corr = inv_var@sigma@inv_var

        # SVD, update the eigen value and scale
        vals, vecs = np.linalg.eig(corr)
        vals[vals<eps] = eps
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

    
    def gen_weight(self, X, lam):
        w = np.array([(1-lam)*(lam**i) for i in range(X.shape[0])])
        w_scale = w/sum(w)
        return w_scale
    
    def EWvar_EWcorr(self, X, lam=0.97):
        cov = np.zeros((X.shape[1], X.shape[1]))
        w = self.gen_weight(X, lam)
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                if i == j:
                    m = X[:, i]
                    scale_m = (m-m.mean())**2
                    cov[i, j] = np.sum(w*scale_m)
                if i!= j:
                    m = X[:, i]
                    n = X[:, j]
                    scale_m = (m-m.mean())
                    scale_n = (n-n.mean())
                    cov[i, j] = np.sum(w*scale_m*scale_n)
                    cov[j, i] = cov[i, j]
        return cov

    def Pvar_Pcorr(self, X):
        corr = np.corrcoef(X.T)
        var_lst = np.sqrt(np.var(X,axis=0))
        var_mat = np.diag(var_lst.tolist())
        cov = var_mat@corr@var_mat
        return cov

    def EWvar_Pcorr(self, X):
        n = X.shape[1]
        corr = np.corrcoef(X.T)
        w = self.gen_weight(X)
        var_mat = np.zeros((n, n))
        for i in range(n):
            m = X[:, i]
            scale_m = (m-m.mean())**2
            var_mat[i, i] = np.sqrt(np.sum(w*scale_m))
        print("corr_mat\n",corr)
        cov = var_mat@corr@var_mat
        return cov

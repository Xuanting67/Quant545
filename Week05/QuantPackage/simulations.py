from scipy.stats import t as tdist
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
import numpy as np
from scipy.stats import t
from pyomo.environ import *
from scipy.optimize import minimize_scalar
from scipy import integrate
import pandas as pd 

class Simulation:
    
    def return_calculate(prices, method='DISCRETE', dateColumn='date'):
        vars = list(prices.columns)
        nVars = len(vars)
        vars = [var for var in vars if var != dateColumn]
        if nVars == len(vars):
            raise ValueError(f'dateColumn: {dateColumn} not in DataFrame: {vars}')
        nVars = nVars - 1

        p = np.array(prices[vars])
        n = p.shape[0]
        m = p.shape[1]
        p2 = np.empty((n-1, m))
        
        for i in range(n-1):
            for j in range(m):
                p2[i,j] = p[i+1,j] / p[i,j]

        if method.upper() == 'DISCRETE':
            p2 = p2 - 1.0
        elif method.upper() == 'LOG':
            p2 = np.log(p2)
        else:
            raise ValueError(f'method: {method} must be in ("LOG","DISCRETE")')

        dates = pd.to_datetime(prices[dateColumn])[1:]
        out = pd.DataFrame({dateColumn: dates})
        for i in range(nVars):
            out[vars[i]] = p2[:, i]
        return out
        
    @staticmethod
    def general_t_ll(params, x):
        mu, s, nu = params
        td = tdist(nu, loc=mu, scale=s)
        return np.sum(np.log(td.pdf(x)))

    @staticmethod
    def fit_general_t(x):
        # approximate values based on moments
        start_m = np.mean(x)
        start_nu = 6.0 / np.kurtosis(x) + 4
        start_s = np.sqrt(np.var(x) * (start_nu - 2) / start_nu)

        # define objective function
        obj = lambda params: -Simulation.general_t_ll(params, x)

        # perform optimization
        bounds = [(None, None), (1e-6, None), (2.0001, None)]
        res = minimize_scalar(obj, bounds=bounds, method='bounded')

        # return the parameters as well as the Distribution Object
        m, s, nu = res.x
        return m, s, nu, tdist(nu, loc=m, scale=s)

    
    @staticmethod
    def ewCovar(x,λ):
        m,n = x.shape
        w = np.empty(m)

        # Remove the mean from the series
        xm = np.mean(x, axis=0)
        x = x - xm

        # Calculate weight. Realize we are going from oldest to newest
        for i in range(m):
            w[i] = (1-λ) * λ**(m-i)

        # Normalize weights to 1
        w /= np.sum(w)

        # covariance[i,j] = (w # x)' * x where # is elementwise multiplication.
        return np.dot(w * x.T, x)

    @staticmethod
    def pca_sim(cov_mtx, n_draws, percent_explain):
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mtx)
        sorted_idx = eigenvalues.argsort()[::-1]
        sorted_eigenvalues = eigenvalues[sorted_idx]
        sorted_eigenvectors = eigenvectors[:, sorted_idx]
        cumulative_explained_var = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
        n_eigenvalues = np.searchsorted(cumulative_explained_var, percent_explain, side="right") + 1
        truncated_eigenvectors = sorted_eigenvectors[:, :n_eigenvalues]
        truncated_eigenvalues = sorted_eigenvalues[:n_eigenvalues]
        normal_draws = np.random.normal(size=(n_eigenvalues, n_draws))
        scaled_draws = truncated_eigenvectors @ np.diag(np.sqrt(truncated_eigenvalues)) @ normal_draws
        return scaled_draws.T



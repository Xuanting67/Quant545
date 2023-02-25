import numpy as np
from scipy import stats
import pandas as pd
from numba import njit, prange
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy import integrate
class RiskMetrics:
    
    @staticmethod
    def return_calculate_1(prices, method='DISCRETE', dateColumn='date'):
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
    def gen_weight(lam, X):
        w = np.array([(1-lam)*(lam**i) for i in range(X.shape[0])])
        w_scale = w/sum(w)
        return w_scale

    @staticmethod
    def VaR(a, alpha=0.05):
        x = np.sort(a)
        nup = np.ceil(np.size(a)*alpha).astype(int)
        ndn = np.floor(np.size(a)*alpha).astype(int)
        v = 0.5*(x[nup]+x[ndn])

        return -v
    def process_portfolio_data(portfolio, prices, p_type):
        if p_type == "total":
            co_assets = portfolio.drop('Portfolio', axis=1)
            co_assets = co_assets.groupby('Stock', as_index=False)['Holding'].sum()
        else:
            co_assets = portfolio.loc[portfolio['Portfolio'] == p_type, ['Stock', 'Holding']]
        dailyprices = prices.loc[:, ['Date'] + co_assets['Stock'].tolist()]
        holdings = co_assets['Holding']
        portfolio_price = (prices[co_assets['Stock']] * holdings.values).sum(axis=1).tail(1).values[0]
        return portfolio_price, dailyprices, holdings

    @staticmethod
    def VaR(d, alpha=0.05):
        return -d.ppf(alpha)

    @staticmethod
    def ES_array(a, alpha=0.05):
        x = np.sort(a)
        nup = np.ceil(np.size(a)*alpha).astype(int)
        ndn = np.floor(np.size(a)*alpha).astype(int)
        v = 0.5*(x[nup]+x[ndn])

        es = np.mean(x[x<=v])
        return -es

    @staticmethod
    def ES_udis(d, alpha=0.05):
        v = RiskMetrics.VaR(d, alpha=alpha)
        f = lambda x: x*d.pdf(x)
        st = d.ppf(1e-12)
        return -quad(f, st, -v)[0]/alpha

    @staticmethod
    def VaR_ES(x, alpha=0.05):
        xs = np.sort(x)
        n = alpha*np.size(xs)
        iup = np.ceil(n).astype(int)
        idn = np.floor(n).astype(int)
        VaR = (xs[iup] + xs[idn])/2
        ES = np.mean(xs[0:idn])

        return -VaR, -ES


    def calculate_es(var, sim_data):
        sim_data_sorted = np.sort(sim_data)
        n = len(sim_data_sorted)
        p = int(n * 0.05)
        es = -np.mean(sim_data_sorted[0:p])
        return es

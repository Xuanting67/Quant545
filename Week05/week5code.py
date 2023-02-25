import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.integrate import quad
import seaborn as sns
import unittest
from QuantPackage.es import pre_process
from QuantPackage.es import cal_return
from QuantPackage.es import cal_es
from QuantPackage.simulations import pca_sim
import numpy as np
from scipy.stats import norm
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt


data = pd.read_csv('problem1.csv', usecols=['x'])['x']

n_mu, n_sigma = norm.fit(data)
n_normal = norm(n_mu, n_sigma)

t_df, t_loc, t_scale = t.fit(data)
t_generalized_t = t(df=t_df, loc=t_loc, scale=t_scale)

fig, ax = plt.subplots(figsize=(10, 5))

x = np.linspace(data.min(), data.max(), 100)
ax.plot(x, n_normal.pdf(x), label='Normal distribution')
ax.plot(x, t_generalized_t.pdf(x), label='Generalized T distribution')

n_var = n_normal.ppf(0.05)
n_es = n_normal.expect(lambda x: x, lb=n_var) / 0.05

t_var = -t_scale * t.ppf(0.05, t_df)
alpha = 0.05  # 95% confidence level
t_es, _ = quad(lambda x: x * t.pdf(x, t_df, loc=t_loc, scale=t_scale), -np.inf, t.ppf(alpha, t_df, loc=t_loc, scale=t_scale))
t_es = t_es / alpha

ax.axvline(n_var, color='r', linestyle='--', label='Normal VaR')
ax.axvline(-t_var, color='g', linestyle='--', label='Generalized T VaR')
ax.axvline(-n_es, color='b', linestyle=':', label='Normal ES')
ax.axvline(t_es, color='m', linestyle=':', label='Generalized T ES')

ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('PDF')

plt.show()

print("Normal distribution:")
print("VaR at 95% confidence level:", round(-n_var, 3))
print("Expected Shortfall at 95% confidence level:", round(n_es, 3))
print("Generalized T distribution:")
print("VaR at 95% confidence level:", round(t_var, 3))
print("Expected Shortfall at 95% confidence level:", round(-t_es, 3))


### Problem2

import unittest
from QuantPackage.simulations import Simulation
from QuantPackage.npsd_fix import NPSD
from QuantPackage.covariance import PortfolioCovariance

class TestRiskMetrics(unittest.TestCase):

    def setUp(self):
        self.prices = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03'],
                                    'asset1': [10, 11, 12],
                                    'asset2': [20, 22, 24]})
        self.returns = pd.DataFrame({'date': ['2022-01-02', '2022-01-03'],
                                     'asset1': [0.1, 0.0909],
                                     'asset2': [0.1, 0.0909]})
        self.a = np.array([1, 2, 3, 4, 5])
        self.d = norm(0, 1)
        print("Test passed.")

    def test_cal_return(self):
        expected_returns = self.returns
        calculated_returns = es.cal_return(self.prices, method='DISCRETE')
        self.assertTrue(expected_returns.equals(calculated_returns))
        print("Test passed.")


    def test_VaR_ES(self):
        expected_var = -3.0
        expected_es = -1.5
        calculated_var, calculated_es = es.VaR_ES(np.array([1, 2, 3, 4, 5]), alpha=0.4)
        self.assertAlmostEqual(expected_var, calculated_var)
        self.assertAlmostEqual(expected_es, calculated_es)
        print("Test passed.")


if __name__ == '__main__':
    unittest.main()



class TestSimulation(unittest.TestCase):
    def test_cal_return(self):
        dates = ['2022-01-01', '2022-01-02', '2022-01-03']
        prices = {'date': dates, 'AAPL': [100, 110, 120], 'GOOG': [500, 510, 520]}
        prices = pd.DataFrame(prices)
        
        returns = Simulation.cal_return(prices, method='DISCRETE')
        
        np.testing.assert_almost_equal(returns.iloc[0, 1], 0.1)
        np.testing.assert_almost_equal(returns.iloc[0, 2], 0.02)
        
    def test_fit_general_t(self):
        mu = 0.1
        s = 0.2
        nu = 5
        data = t.rvs(df=nu, loc=mu, scale=s, size=1000)
        
        m, s, nu, dist = Simulation.fit_general_t(data)
        np.testing.assert_almost_equal(m, mu, decimal=1)
        np.testing.assert_almost_equal(s, s, decimal=1)
        np.testing.assert_almost_equal(nu, nu, decimal=1)
        
if __name__ == '__main__':
    unittest.main()


class TestNPSD(unittest.TestCase):
    def test_near_psd(self):
        np.random.seed(0)
        n = 10
        sigma = np.random.normal(size=(n,n))
        sigma = sigma.T @ sigma
        psd = NPSD().near_psd(sigma)
        np.testing.assert_array_less(np.linalg.eigvals(psd), 0.0)
        
    def test_chol_psd(self):
        np.random.seed(0)
        n = 10
        sigma = np.random.normal(size=(n,n))
        sigma = sigma.T @ sigma
        root = NPSD().chol_psd(sigma)
        np.testing.assert_allclose(root @ root.T, sigma)

class TestCovariance(unittest.TestCase):
    def test_ewma_covariance(self):
        np.random.seed(0)
        n = 100
        m = 5
        X = np.random.normal(size=(n, m))
        
        lambda_ = 0.94
        cov = PortfolioCovariance.ewma_covariance(X, lambda_)
        
        np.testing.assert_allclose(cov, cov.T)
        np.testing.assert_array_less(np.linalg.eigvals(cov), 0.0)

if __name__ == '__main__':
    unittest.main()

###Problem3

def cvar(data, prices, p_type):
    cur_price, daily_prices, holdings = pre_process(data, prices, p_type)
    
    asset_returns = cal_return(daily_prices)
    asset_returns.drop('Date', axis=1, inplace=True)
    zero_mean_returns = asset_returns - asset_returns.mean()
    
    returns_transf = zero_mean_returns.copy()
    loc, scale, df = {}, {}, {}
    for asset in returns_transf.columns.tolist():
        result = t.fit(zero_mean_returns[asset], method="MLE")
        df[asset], loc[asset], scale[asset] = result[0], result[1], result[2]
        returns_transf[asset] = t.cdf(zero_mean_returns[asset], df=df[asset], loc=loc[asset], scale=scale[asset])
    returns_transf = pd.DataFrame(norm.ppf(returns_transf), index=returns_transf.index, columns=returns_transf.columns)
    
    spearman_corr_mtx = returns_transf.corr(method='spearman')
    
    simulations = pca_sim(spearman_corr_mtx, 10000, percent_explain = 1)
    simulations = pd.DataFrame(simulations, columns=returns_transf.columns)
    
    returns_back = pd.DataFrame(norm.cdf(simulations), index=simulations.index, columns=simulations.columns)
    
    for asset in returns_transf.columns.tolist():
        returns_back[asset] = t.ppf(returns_back[asset], df=df[asset], loc=loc[asset], scale=scale[asset])
    
    sim_returns = np.add(returns_back, zero_mean_returns.mean())
    daily_prices = daily_prices.drop('Date', axis=1)
    delta_sim = np.dot(sim_returns * daily_prices.tail(1).values.reshape(daily_prices.shape[1]), holdings)

    var = np.percentile(delta_sim, 0.05*100) * (-1)
    es = cal_es(var, delta_sim)
    
    return var, es, delta_sim, cur_price




data = pd.read_csv('portfolio.csv')
prices = pd.read_csv('DailyPrices.csv')

sns.set_style("whitegrid")

portfolios = {'A': 'Portfolio A', 'B': 'Portfolio B', 'C': 'Portfolio C', 'total': 'Portfolio TOTAL'}

for p_type, p_title in portfolios.items():
    var, es, dis, p = cvar(data, prices, p_type)
    print(p_title)
    
    print('VaR: ', var)
    print('ES: ', es)

    fig, ax = plt.subplots()

    ax.hist(dis, bins=30, density=True, alpha=0.5, color='cornflowerblue')
    sns.kdeplot(dis, color='green', linewidth=1, ax=ax)
    ax.axvline(-var, label='VaR', color='red', linestyle='--')
    ax.axvline(-es, label='ES', color='purple', linestyle='--')
    ax.set_xlabel('Returns')
    ax.set_ylabel('Density')
    ax.set_title(f'{p_title} ')
    ax.legend()
    plt.show()
#Problem 1

import numpy as np
import pandas as pd
from scipy.stats import norm
import inspect
import scipy
import datetime as dt

S = 165
K = 165
X = 165
T = (dt.datetime(2023,4,15) - dt.datetime(2023,3,13)).days / 365
r = 0.0425
q = 0.0053
b = r - q
sigma = 0.2

import numpy as np
from scipy.stats import norm
def delta_gbsm(S, X, T, b, r, sigma, opt_type="call"):
    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    try:
        if opt_type == "call":
            delta = np.exp((b - r) * T) * norm.cdf(d1)
        elif opt_type == "put":
            delta = np.exp((b - r) * T) * (norm.cdf(d1) - 1)
        else:
            raise ValueError("Invalid option type")
        return delta
    except ValueError as error:
        print(error)


def gbsm_gamma(S, X, T, b, r, sigma, opt_type="call"):
    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    #d2 = calculate_d2(d1, T, sigma)
    try:
        gamma_calc = norm.pdf(d1, 0, 1) * np.exp((b-r)*T)/(S*sigma*np.sqrt(T))
        return gamma_calc
    except:
        print("Wrong option type")

def option_theta(S, X, T, b, r, sigma, opt_type="call"):
    """Compute theta for an option."""

    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if opt_type == "call":
            theta = (-S * np.exp((b - r) * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - (b - r) * S * np.exp((b - r) * T) * norm.cdf(d1)
                     - r * X * np.exp(-r * T) * norm.cdf(d2))
        elif opt_type == "put":
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     + (b - r) * S * np.exp((b - r) * T) * norm.cdf(-d1)
                     + r * X * np.exp(-r * T) * norm.cdf(-d2))
        else:
            raise ValueError("Invalid option type")
        return theta

    except ValueError as error:
        print(error)

def gbsm_rho(S, X, T, b, r, sigma, opt_type="call"):
    "Calculate rho of an option"
    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    try:
        if type == "call":
            rho_calc = X*T*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "put":
            rho_calc = -X*T*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        return rho_calc
    except:
        print("Wrong option type")


def gbsm_carry_rho(S, X, T, b, r, sigma, opt_type="call"):
    "Calculate rho of an option"
    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    try:
        if type == "call":
            rho_calc = S*T*np.exp((b-r)*T)*norm.cdf(d1, 0, 1)
        elif type == "put":
            rho_calc = -S*T*np.exp((b-r)*T)*norm.cdf(-d1, 0, 1)
        return rho_calc
    except:
        print("Wrong option type")


print("Call's Delta ", delta_gbsm(S, X, T, b, r, sigma, opt_type="call"))
print("Put's Delta ", delta_gbsm(S, X, T, b, r, sigma, opt_type="put"))
print("Call's Gamma ", option_gamma(S, X, T, b, r, sigma, opt_type="call"))
print("Put's Gamma ", option_gamma(S, X, T, b, r, sigma, opt_type="put"))
print("Call's Vega ", option_vega(S, X, T, b, r, sigma, opt_type="call"))
print("Put's Vega ", option_vega(S, X, T, b, r, sigma, opt_type="put"))\
print("Put's Theta is ", option_theta(S, X, T, b, r, sigma, opt_type="put"))
print("Call's Rho is ", gbsm_rho(S, X, T, b, r, sigma, type="call"))
print("Put's Rho is ", gbsm_rho(S, X, T, b, r, sigma, type="put")) 
print("Call's Carry Rho ", gbsm_carry_rho(S, X, T, b, r, sigma, type="call"))
print("Put's Carry Rho ", gbsm_carry_rho(S, X, T, b, r, sigma, type="put")) 

delta_call = delta_gbsm(S, K, T, b, r, sigma, "call")
delta_put = delta_gbsm(S, K, T, b, r, sigma, "put")
gamma = option_gamma(S, K, T, b, r, sigma)
vega = option_vega(S, K, T, b, r, sigma)
theta_call = option_theta(S, K, T, b, r, sigma, "call")
theta_put = option_theta(S, K, T, b, r, sigma, "put")
rho_call = gbsm_rho(S, K, T, b, r, sigma, "call")
rho_put = gbsm_rho(S, K, T, b, r, sigma, "put")
carry_rho_call = gbsm_carry_rho(S, K, T, b, r, sigma, "call")
carry_rho_put = gbsm_carry_rho(S, K, T, b, r, sigma, "put")

import inspect

def first_order_derivative(func, x, delta):
    return (func(x + delta) - func(x - delta)) / (2 * delta)

def second_order_derivative(func, x, delta):
    return (func(x + delta) + func(x - delta) - 2 * func(x)) / delta ** 2

def partial_derivative(target_func, order, variable_name, delta=1e-3):
    argument_names = list(inspect.signature(target_func).parameters.keys())
    derivative_functions = {1: first_order_derivative, 2: second_order_derivative}

    def partial_deriv(*args, **kwargs):
        arguments_dict = dict(zip(argument_names, args))
        arguments_dict.update(kwargs)
        variable_value = arguments_dict.pop(variable_name)

        def single_argument_func(x):
            updated_kwargs = {variable_name: x}
            updated_kwargs.update(arguments_dict)
            return target_func(**updated_kwargs)

        return derivative_functions[order](single_argument_func, variable_value, delta)

    return partial_deriv


def gbsm(option_type, S, X, T, sigma, r, b):
  d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  is_call = 1 if option_type == "Call" else -1

  res = is_call * (S * np.e ** ((b - r) * T) * scipy.stats.norm(0, 1).cdf(is_call * d1) - X * np.e ** (-r * T) * scipy.stats.norm(0, 1).cdf(is_call * d2))

  return res



delta_gbsm = partial_derivative(gbsm, 1, 'S')
print("Call's Delta ", delta_gbsm( "Call", S, K, T, sigma, r, b))
print("Put's Delta ", delta_gbsm( "Put", S, K, T, sigma, r, b)) 
gbsm_gamma = partial_derivative(gbsm, 2 ,'S')
print("Call's Gamma  ", gbsm_gamma( "Call", S, K, T, sigma, r, b))
print("Put's Gamma  ", gbsm_gamma( "Put", S, K, T, sigma, r, b)) 
gbsm_vega = partial_derivative(gbsm, 1 ,'sigma')
print("Call's Vega  ", gbsm_vega( "Call", S, K, T, sigma, r, b))
print("Put's Vega  ", gbsm_vega( "Put", S, K, T, sigma, r, b)) 
gbsm_theta = partial_derivative(gbsm, 1 ,'T')
print("Call's Theta  ", -gbsm_theta( "Call", S, K, T, sigma, r, b))
print("Put's Theta  ", -gbsm_theta( "Put", S, K, T, sigma, r, b))
gbsm_rho = partial_derivative(gbsm, 1 ,'r')
print("Call's Rho  ", gbsm_rho( "Call", S, K, T, sigma, r, b))
print("Put's Rho  ", gbsm_rho( "Put", S, K, T, sigma, r, b))
gbsm_carry_rho = partial_derivative(gbsm, 1 ,'b')
print("Call's Carry Rho  ", gbsm_carry_rho( "Call", S, K, T, sigma, r, b))
print("Put's Carry Rho  ", gbsm_carry_rho( "Put", S, K, T, sigma, r, b))


def binomial_tree_american_none(S, K, T, r, q, sigma, N, option_type='call'):
    dt = T / N
    up_factor = np.exp(sigma * np.sqrt(dt))
    down_factor = 1 / up_factor
    prob_up = (np.exp((r - q) * dt) - down_factor) / (up_factor - down_factor)
    prob_down = 1 - prob_up
    df = np.exp(-r * dt)
    z = 1 if option_type == 'call' else -1

    def num_nodes_at_level(level):
        return (level + 2) * (level + 1) // 2

    def node_index(i, j):
        return num_nodes_at_level(j - 1) + i

    n_nodes = num_nodes_at_level(N)
    option_values = np.empty(n_nodes, dtype=float)

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = node_index(i, j)
            price = S * up_factor ** i * down_factor ** (j - i)
            intrinsic_value = max(0, z * (price - K))
            option_values[idx] = intrinsic_value

            if j < N:
                expected_value = df * (prob_up * option_values[node_index(i + 1, j + 1)]
                                       + prob_down * option_values[node_index(i, j + 1)])
                option_values[idx] = max(intrinsic_value, expected_value)

    return option_values[0]


def biotree_ame_div(S, K, r, T, sigma, N, option_type, dividend_dates=None, dividend_amounts=None):
    if dividend_dates is None or dividend_amounts is None or (len(dividend_amounts)==0) or (len(dividend_dates)==0) or dividend_dates[0] > N:
        return binomial_tree_american_none(S, K, T, r, 0, sigma, N, option_type)

    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(r*dt)-d)/(u-d)
    pd = 1-pu
    df = np.exp(-r*dt)
    z = 1 if option_type == 'call' else -1
    n_dividends = len(dividend_dates)
    optionValues = np.zeros((n_dividends+1)*(n_dividends+2)//2, dtype=float)
    nodes_per_period = [n_dividends+2-i for i in range(n_dividends+1)]
    for i in range(n_dividends+1):
        for j in range(i, -1, -1):
            idx = (i+1)*i//2 + j
            price = S * u**(j) * d**(i-j)
            
            if i == n_dividends:
                optionValues[idx] = max(0, z * (price - K))
            else:
                no_ex = df * (pu * optionValues[idx+1] + pd * optionValues[idx])
                ex = max(0, z * (price - K - dividend_amounts[i]))
                optionValues[idx] = max(no_ex, ex)

    return optionValues[0]


N = 200
b = 0.0425
T = (dt.datetime(2023,4,15) - dt.datetime(2023,3,13)).days / 365
dividend_amounts=[0.88]
dividend_dates = [round((dt.datetime(2023,4,11)-dt.datetime(2023,3,13)).days/(dt.datetime(2023,4,15)-dt.datetime(2023,3,13)).days*N)]

print("The value of call option with dividend ", biotree_ame_div(S, K, r, T, sigma, N, 'call',dividend_dates, dividend_amounts))
print("The value of put option is ", biotree_ame_div(S, K, r, T, sigma, N, 'put',dividend_dates, dividend_amounts))
print("The value of call option without dividend ", binomial_tree_american_none(S, K, T, r, q, sigma, N, option_type='call',))
print("The value of put option without dividend ", binomial_tree_american_none(S, K, T, r, q, sigma, N, option_type='put'))
btree_delta = partial_deriv(biotree_ame_div, 1, 'S')
print("Call's Delta ", btree_delta(S, K, r, T, sigma, N, 'call',dividend_dates, dividend_amounts))
print("Put's Delta ", btree_delta(S, K, r, T, sigma, N, 'put',dividend_dates, dividend_amounts)) 


btree_gamma = partial_deriv(biotree_ame_div, 2 ,'S')
print("Call's Gamma ", btree_gamma(S, K, r, T, sigma, N, 'call',dividend_dates, dividend_amounts))
print("Put's Gamma ", btree_gamma(S, K, r, T, sigma, N, 'put',dividend_dates, dividend_amounts)) 

bt_vega = partial_deriv(biotree_ame_div, 1 ,'sigma')
print("Call's Vega ", bt_vega(S, K, r, T, sigma, N, 'call',dividend_dates, dividend_amounts))
print("Put's Vega ", bt_vega(S, K, r, T, sigma, N, 'put',dividend_dates, dividend_amounts) )

bt_theta = partial_deriv(biotree_ame_div, 1 ,'T')
print("Call's Theta ", -bt_theta(S, K, r, T, sigma, N, 'call',dividend_dates, dividend_amounts))
print("Put's Theta ", -bt_theta(S, K, r, T, sigma, N, 'put',dividend_dates, dividend_amounts))

bt_rho = partial_deriv(biotree_ame_div, 1 ,'r')
print("Call's Rho ", bt_rho(S, K, r, T, sigma, N, 'call',dividend_dates, dividend_amounts))
print("Put's Rho ", bt_rho(S, K, r, T, sigma, N, 'put',dividend_dates, dividend_amounts))

bt_carry_rho = partial_deriv(biotree_ame_div, 1 ,'K')
print("Call's Carry Rho ", (S, K, r, T, sigma, N, 'call',dividend_dates, dividend_amounts))
print("Put's Carry Rho ", bt_carry_rho(S, K, r, T, sigma, N, 'put',dividend_dates, dividend_amounts))


#Problem 2
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import brentq

df = pd.read_csv('problem2.csv')

def calculate_implied_volatility(data):
    if data['Type'] != 'Option':
        return np.nan
    
    underlying_price = 151.03
    strike_price = data['Strike']
    is_call_option = data['OptionType'] == 'Call'
    market_price = data['CurrentPrice']

    expiration_date = datetime.datetime.strptime(data['ExpirationDate'], '%m/%d/%Y')
    dividend_date = datetime.datetime.strptime('03/15/2023', '%m/%d/%Y')
    time_to_dividend = (dividend_date - expiration_date).days
    time_to_expiration = (expiration_date - datetime.datetime.now()).days / 365

    div_amount = 1
    div_times = [time_to_dividend / time_to_expiration]
    num_dividends = len(div_times)

    def f(volatility):
        return binomial_tree_gbsm_american_div(
            underlying_price, strike_price, time_to_expiration, volatility,
            0.0425, 0.0425, num_dividends, is_call_option, [div_amount], div_times
        ) - market_price

    return brentq(f, 1e-6, 10)

df = pd.read_csv('problem2.csv')
df['Implied Volatility'] = df.apply(calculate_implied_volatility, axis=1)


import datetime
import numpy as np

def calculate_position_value(data, prices, ndays=0):
    if data['Type'] != 'Option':
        return prices * data['Holding']
    
    strike_price = data['Strike']
    expiration_date = datetime.datetime.strptime(data['ExpirationDate'], '%m/%d/%Y')
    dividend_date = datetime.datetime.strptime('03/15/2023', '%m/%d/%Y')
    time_to_dividend = (dividend_date - expiration_date).days
    time_to_expiration = (expiration_date - datetime.datetime.now()).days / 365

    is_call_option = data['OptionType'] == 'Call'
    risk_free_rate = 0.0425
    cost_of_carry = risk_free_rate
    time_to_expiration_in_years = time_to_expiration - ndays / 365
    volatility = data['Implied Volatility']
    dividend_amounts = [1]
    dividend_times = [time_to_dividend / time_to_expiration * 20]

    position_value = binomial_tree_gbsm_american_div(
        prices, strike_price, time_to_expiration_in_years, volatility,
        risk_free_rate, cost_of_carry, 20, is_call_option, dividend_amounts, dividend_times
    )

    return position_value * data['Holding']

df = pd.read_csv('problem2.csv')
prices = np.array([151.03] * len(df))

ndays = 0  # simulate price today
df['Position Value'] = calculate_position_value(df, prices, ndays)


import numpy as np
from scipy.stats import norm

price_data = pd.read_csv('DailyPrices.csv', index_col='Date')
log_returns = return_calculate(price_data.AAPL, option="CONTINUOUS", rm_means=True)

mu, sigma = norm.fit(log_returns)
simulated_returns = np.random.normal(mu, sigma, size=(100000, 10))
simulated_price = 151.03 * np.exp(simulated_returns.sum(axis=1))

simulated_returns = simulated_returns.flatten()


import numpy as np


strategies = df.Portfolio.unique()

position_values = df.apply(binomial_tree_gbsm_american_div_df, prices=simulated_price, ndays=10, axis=1)
initial_values = df.apply(binomial_tree_gbsm_american_div_df, prices=[currentPrice], axis=1)

total_portfolio_value = position_values.groupby(df.Portfolio).sum()
initial_portfolio_value = initial_values.groupby(df.Portfolio).sum()
portfolio_value_change = total_portfolio_value - initial_portfolio_value

VaR_p = VaR_historical(portfolio_value_change.values, alpha=0.05)
ES = ES_historical(portfolio_value_change.values, alpha=0.05)

ans = pd.DataFrame({
    'Mean(Portfolio Value)': total_portfolio_value.mean(),
    'Mean(Change)': portfolio_value_change.mean(),
    'VaR': VaR_p,
    'ES': ES
}, index=strategies)


### Problem 3
import statsmodels.api as sm
from scipy.optimize import fsolve, minimize


data = pd.read_csv('F-F_Research_Data_Factors_daily.csv', parse_dates=['Date']).set_index('Date')
mom = pd.read_csv('F-F_Momentum_Factor_daily.csv', parse_dates=['Date']).set_index('Date').rename(columns={'Mom   ':  "Mom"})
factor = (data.join(mom, how='right') / 100).loc['2013-1-31':]

def return_calculate(prices, method='Arithmetic'):
    price_new = prices.iloc[:, 1:].to_numpy()
    price_return = price_new[1:] / price_new[:-1]

    if method == 'Arithmetic':
        price_return = price_return - 1.0
    elif method == 'Log':
        price_return = np.log(price_return)
    else:
        raise ValueError(f'Wrong method: {method}')

    prices_da = pd.DataFrame({
        'Date': prices['Date'].iloc[1:],
        **{col: price_return[:, i] for i, col in enumerate(prices.columns[1:])},
    })

    return prices_da

prices = pd.read_csv('DailyPrices.csv', parse_dates=['Date'])
all_returns = pd.DataFrame(return_calculate(prices)).set_index('Date')
stocks = ['AAPL', 'META', 'UNH', 'MA',  
          'MSFT' ,'NVDA', 'HD', 'PFE',  
          'AMZN' ,'BRK-B', 'PG', 'XOM',  
          'TSLA' ,'JPM' ,'V', 'DIS',  
          'GOOGL', 'JNJ', 'BAC', 'CSCO']
factors = ['Mkt-RF', 'SMB', 'HML', 'Mom']
dataset = all_returns[stocks].join(factor)

subset = dataset.dropna()

X = subset[factors]
X = sm.add_constant(X)
y = subset[stocks] - subset['RF'].values.reshape(-1, 1)
betas = pd.DataFrame(index=stocks, columns=factors)
alphas = pd.DataFrame(index=stocks, columns=['Alpha'])
# Loop over each stock and fit the OLS model
for stock in stocks:
    model = sm.OLS(y[stock], X).fit() # use indexing to select the current stock from y
    betas.loc[stock] = model.params[factors]
    alphas.loc[stock] = model.params['const']



return_sub = pd.DataFrame(np.dot(factor[factors],betas.T), index=factor.index, columns=betas.index)
merged = pd.merge(return_sub,factor['RF'], left_index=True, right_index=True)
daily_expected_returns = merged.add(merged['RF'],axis=0).drop('RF',axis=1).add(alphas.T.loc['Alpha'], axis=1)
expected_annual_return = ((daily_expected_returns+1).cumprod().tail(1) ** (1/daily_expected_returns.shape[0]) - 1) * 252
expected_annual_return


from scipy.optimize import minimize

def efficient_portfolio(returns, risk_free_rate, cov_matrix):
    # Determine number of assets
    num_assets = returns.shape[1] if len(returns.shape) > 1 else returns.shape[0]
    
    # Define objective function to minimize (negative sharpe ratio)
    def negative_sharpe_ratio(weights):
        portfolio_return = np.sum(returns * weights)
        portfolio_std_dev = np.sqrt(weights @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return -sharpe_ratio
    
    # Define constraints and bounds
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # weights must add up to 1
        {'type': 'ineq', 'fun': lambda weights: weights},  # weights cannot be negative
    ]
    bounds = [(0, 1) for _ in range(num_assets)]
    
    # Solve for optimal weights
    initial_weights = np.ones(num_assets) / num_assets  # start with equal weights
    optimal_result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Compute optimal portfolio metrics
    optimal_weights = optimal_result.x * 100
    optimal_portfolio_return = np.sum(returns * optimal_weights / 100)
    optimal_portfolio_std_dev = np.sqrt(optimal_weights / 100 @ cov_matrix @ optimal_weights / 100)
    optimal_sharpe_ratio = (optimal_portfolio_return - risk_free_rate) / optimal_portfolio_std_dev
    
    return optimal_weights, optimal_sharpe_ratio


covariance_matrix = dataset[stocks].cov() * 252
weights, sharpe_ratio = efficient_portfolio(expected_annual_return.values[0], 0.0425, covariance_matrix)
print("The Portfolio's Sharpe Ratio is: {:.2f}" .format(sharpe_ratio))
weights = pd.DataFrame(weights, index=expected_annual_return.columns, columns=['weight %']).round(2).T
weights
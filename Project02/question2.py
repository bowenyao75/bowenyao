import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import t, norm
import matplotlib.pyplot as plt


def calculate_ewma_volatility(returns, lambda_param):
    """
    Calculate exponentially weighted moving average volatility.
    Parameters:
    -----------
    returns : Series
        Asset returns
    lambda_param : float
        EWMA decay factor
    Returns:
    --------
    float
        EWMA volatility
    """
    # Calculate EWMA variance
    ewma_var = 0
    weight_sum = 0
    for i in range(len(returns)-1, -1, -1):
        weight = lambda_param**(len(returns)-1-i)
        weight_sum += weight
        ewma_var += weight * returns.iloc[i]**2  # Assuming zero mean
    ewma_var /= weight_sum
    return np.sqrt(ewma_var)


def calculate_ewma_covariance_matrix(returns, lambda_param):
    """
    Calculate EWMA covariance matrix.
    Parameters:
    -----------
    returns : DataFrame
        Asset returns
    lambda_param : float
        EWMA decay factor
    Returns:
    --------
    ndarray
        EWMA covariance matrix
    """
    n_assets = returns.shape[1]
    cov_matrix = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                # Variance (diagonal elements)
                cov_matrix[i, j] = calculate_ewma_volatility(returns.iloc[:, i], lambda_param)**2
            else:
                # Covariance (off-diagonal elements)
                covar = 0
                weight_sum = 0          
                for k in range(len(returns)-1, -1, -1):
                    weight = lambda_param**(len(returns)-1-k)
                    weight_sum += weight
                    covar += weight * returns.iloc[k, i] * returns.iloc[k, j]  # Assuming zero mean       
                cov_matrix[i, j] = covar / weight_sum
    return cov_matrix


def calculate_portfolio_value_and_risks(file_path):
    """
    Calculate portfolio value and risk metrics for a portfolio of stocks.
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing daily prices
    Returns:
    --------
    dict
        Dictionary containing portfolio value and risk metrics
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Convert the Date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    # Portfolio composition
    portfolio = {
        'SPY': 100,
        'AAPL': 200,
        'EQIX': 150
    }
    # A. Calculate the current value of the portfolio (as of 1/3/2025)
    current_date = '2025-01-03'
    current_prices = df[df['Date'] == current_date][['SPY', 'AAPL', 'EQIX']]
    if current_prices.empty:
        raise ValueError(f"No data found for date {current_date}")
    portfolio_value = (current_prices['SPY'] * portfolio['SPY'] + 
                       current_prices['AAPL'] * portfolio['AAPL'] + 
                       current_prices['EQIX'] * portfolio['EQIX']).values[0]
    # Calculate current portfolio weights
    weights = {
        'SPY': (portfolio['SPY'] * current_prices['SPY'].values[0]) / portfolio_value,
        'AAPL': (portfolio['AAPL'] * current_prices['AAPL'].values[0]) / portfolio_value,
        'EQIX': (portfolio['EQIX'] * current_prices['EQIX'].values[0]) / portfolio_value
    }
    # Calculate arithmetic returns
    returns = df[['SPY', 'AAPL', 'EQIX']].pct_change().dropna()
    # Calculate portfolio returns using current weights
    portfolio_returns = (returns['SPY'] * weights['SPY'] + 
                         returns['AAPL'] * weights['AAPL'] + 
                         returns['EQIX'] * weights['EQIX'])
    # B. Calculate VaR and ES using three different methods
    results = {}
    results['portfolio_value'] = portfolio_value
    results['weights'] = weights
    # B.a. Normally distributed with exponentially weighted covariance (lambda=0.97)
    results['normal_ewma'] = calculate_normal_ewma(returns, portfolio_returns, weights, 
                                                 portfolio, current_prices, portfolio_value)
    # B.b. T distribution using a Gaussian Copula
    results['t_dist'] = calculate_t_distribution(returns, portfolio_returns, weights, 
                                               portfolio, current_prices, portfolio_value)
    # B.c. Historical simulation
    results['historical'] = calculate_historical(returns, portfolio_returns, weights, 
                                               portfolio, current_prices, portfolio_value)
    return results


def calculate_normal_ewma(returns, portfolio_returns, weights, portfolio, current_prices, portfolio_value):
    """
    Calculate VaR and ES using normally distributed returns with EWMA covariance.
    Parameters:
    -----------
    returns : DataFrame
        Stock returns dataframe
    portfolio_returns : Series
        Portfolio returns series
    weights : dict
        Portfolio weights
    portfolio : dict
        Portfolio quantities
    current_prices : DataFrame
        Current stock prices
    portfolio_value : float
        Current portfolio value
    Returns:
    --------
    dict
        Dictionary containing VaR and ES values
    """
    lambda_param = 0.97
    alpha = 0.05
    # Calculate EWMA volatility for individual stocks
    ewma_vol = {}
    for stock in returns.columns:
        ewma_vol[stock] = calculate_ewma_volatility(returns[stock], lambda_param)
    # Calculate EWMA covariance matrix
    ewma_cov_matrix = calculate_ewma_covariance_matrix(returns, lambda_param)
    # Calculate portfolio EWMA volatility
    weights_array = np.array([weights['SPY'], weights['AAPL'], weights['EQIX']])
    portfolio_ewma_var = weights_array @ ewma_cov_matrix @ weights_array.T
    portfolio_ewma_vol = np.sqrt(portfolio_ewma_var)
    # Calculate VaR using normal distribution (5% alpha level)
    normal_quantile = norm.ppf(alpha)
    var_normal = {}
    es_normal = {}
    # VaR for individual stocks
    for stock in returns.columns:
        var_normal[stock] = -normal_quantile * ewma_vol[stock] * current_prices[stock].values[0] * portfolio[stock]
        # ES for normal distribution
        # For normal distribution, ES = -E[X | X <= VaR] = -mean + (std * pdf(q) / cdf(q))
        es_factor = -normal_quantile + norm.pdf(normal_quantile) / alpha
        es_normal[stock] = es_factor * ewma_vol[stock] * current_prices[stock].values[0] * portfolio[stock]
    # VaR and ES for the portfolio
    var_normal['Portfolio'] = -normal_quantile * portfolio_ewma_vol * portfolio_value
    es_normal['Portfolio'] = (-normal_quantile + norm.pdf(normal_quantile) / alpha) * portfolio_ewma_vol * portfolio_value
    return {
        'var': var_normal,
        'es': es_normal,
        'ewma_vol': ewma_vol,
        'portfolio_vol': portfolio_ewma_vol
    }


def calculate_t_distribution(returns, portfolio_returns, weights, portfolio, current_prices, portfolio_value):
    """
    Calculate VaR and ES using t-distribution with a Gaussian Copula.
    Parameters:
    -----------
    returns : DataFrame
        Stock returns dataframe
    portfolio_returns : Series
        Portfolio returns series
    weights : dict
        Portfolio weights
    portfolio : dict
        Portfolio quantities
    current_prices : DataFrame
        Current stock prices
    portfolio_value : float
        Current portfolio value
    Returns:
    --------
    dict
        Dictionary containing VaR and ES values
    """
    lambda_param = 0.97
    alpha = 0.05
    # Calculate kurtosis to estimate degrees of freedom
    kurtosis_values = {}
    for stock in returns.columns:
        kurtosis_values[stock] = stats.kurtosis(returns[stock], fisher=False)
    avg_kurtosis = np.mean(list(kurtosis_values.values()))
    # For t-distribution, kurtosis = 3 + 6/(df-4) if df > 4
    # So df = 6/(kurtosis-3) + 4
    df = 6 / (avg_kurtosis - 3) + 4
    # Bound the degrees of freedom to reasonable values
    df = max(5, min(30, df))
    # Calculate EWMA volatility (same as in normal method)
    ewma_vol = {}
    for stock in returns.columns:
        ewma_vol[stock] = calculate_ewma_volatility(returns[stock], lambda_param)
    # Calculate EWMA covariance matrix
    ewma_cov_matrix = calculate_ewma_covariance_matrix(returns, lambda_param)
    # Calculate portfolio EWMA volatility
    weights_array = np.array([weights['SPY'], weights['AAPL'], weights['EQIX']])
    portfolio_ewma_var = weights_array @ ewma_cov_matrix @ weights_array.T
    portfolio_ewma_vol = np.sqrt(portfolio_ewma_var)
    # Calculate VaR using t-distribution
    t_quantile = t.ppf(alpha, df)
    # Calculate ES for t-distribution
    # For t-distribution, ES = -E[X | X <= VaR]
    # This is more complex than for normal distribution
    # Using the formula: ES_t = -t_quantile * (df + t_quantile^2) / (df - 1) * (df / (df - 2)) * (1 / alpha)
    es_factor_t = -t_quantile * (df + t_quantile**2) / (df - 1) * (df / (df - 2)) * (1 / alpha)
    var_t = {}
    es_t = {}
    # VaR for individual stocks
    for stock in returns.columns:
        var_t[stock] = -t_quantile * ewma_vol[stock] * current_prices[stock].values[0] * portfolio[stock]
        es_t[stock] = es_factor_t * ewma_vol[stock] * current_prices[stock].values[0] * portfolio[stock]
    # VaR and ES for the portfolio
    var_t['Portfolio'] = -t_quantile * portfolio_ewma_vol * portfolio_value
    es_t['Portfolio'] = es_factor_t * portfolio_ewma_vol * portfolio_value
    return {
        'var': var_t,
        'es': es_t,
        'df': df,
        'kurtosis': kurtosis_values
    }


def calculate_historical(returns, portfolio_returns, weights, portfolio, current_prices, portfolio_value):
    """
    Calculate VaR and ES using historical simulation.
    Parameters:
    -----------
    returns : DataFrame
        Stock returns dataframe
    portfolio_returns : Series
        Portfolio returns series
    weights : dict
        Portfolio weights
    portfolio : dict
        Portfolio quantities
    current_prices : DataFrame
        Current stock prices
    portfolio_value : float
        Current portfolio value
    Returns:
    --------
    dict
        Dictionary containing VaR and ES values
    """
    alpha = 0.05
    # Calculate dollar returns for each stock and portfolio
    dollar_returns = {}
    for stock in returns.columns:
        dollar_returns[stock] = returns[stock] * current_prices[stock].values[0] * portfolio[stock]
    # Ensure portfolio_returns is properly accessed
    portfolio_returns_array = portfolio_returns.values if hasattr(portfolio_returns, 'values') else portfolio_returns
    dollar_returns['Portfolio'] = portfolio_returns_array * portfolio_value
    # Calculate VaR using historical simulation (5th percentile of negative returns)
    var_hist = {}
    es_hist = {}
    for asset in dollar_returns.keys():
        # Sort returns in ascending order
        sorted_returns = np.sort(dollar_returns[asset])
        # Find the index for the alpha percentile
        idx = int(np.ceil(alpha * len(sorted_returns))) - 1
        # Calculate VaR (negative of the alpha percentile return)
        var_hist[asset] = -sorted_returns[idx]
        # Calculate ES (average of returns below VaR)
        es_hist[asset] = -np.mean(sorted_returns[:idx+1])
    return {
        'var': var_hist,
        'es': es_hist
    }


def plot_var_comparison(results):
    """
    Plot comparison of VaR results across different methods.
    Parameters:
    -----------
    results : dict
        Dictionary containing risk metrics from different methods
    """
    assets = list(results['normal_ewma']['var'].keys())
    methods = ['Normal w/ EWMA', 'T-Distribution', 'Historical']
    var_data = {
        'Normal w/ EWMA': [results['normal_ewma']['var'][asset] for asset in assets],
        'T-Distribution': [results['t_dist']['var'][asset] for asset in assets],
        'Historical': [results['historical']['var'][asset] for asset in assets]
    }
    es_data = {
        'Normal w/ EWMA': [results['normal_ewma']['es'][asset] for asset in assets],
        'T-Distribution': [results['t_dist']['es'][asset] for asset in assets],
        'Historical': [results['historical']['es'][asset] for asset in assets]
    }
    # Create VaR comparison plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(assets))
    width = 0.25
    for i, method in enumerate(methods):
        plt.bar(x + (i - 1) * width, var_data[method], width, label=method)
    plt.xlabel('Assets')
    plt.ylabel('Value at Risk (VaR) - 5% level')
    plt.title('Comparison of VaR Across Methods')
    plt.xticks(x, assets)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('var_comparison.png', dpi=300, bbox_inches='tight')
    # Create ES comparison plot
    plt.figure(figsize=(12, 6))
    for i, method in enumerate(methods):
        plt.bar(x + (i - 1) * width, es_data[method], width, label=method)
    plt.xlabel('Assets')
    plt.ylabel('Expected Shortfall (ES) - 5% level')
    plt.title('Comparison of ES Across Methods')
    plt.xticks(x, assets)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('es_comparison.png', dpi=300, bbox_inches='tight')


def main():
    """
    Main function to execute the analysis and display the results.
    """
    file_path = 'DailyPrices.csv'
    try:
        # Calculate portfolio value and risk metrics
        results = calculate_portfolio_value_and_risks(file_path)
        # Display portfolio value
        print(f"A. Portfolio Value as of 1/3/2025: ${results['portfolio_value']:.2f}")
        print(f"Portfolio Weights: SPY: {results['weights']['SPY']*100:.2f}%, AAPL: {results['weights']['AAPL']*100:.2f}%, EQIX: {results['weights']['EQIX']*100:.2f}%")
        print("\nB. Value at Risk (VaR) and Expected Shortfall (ES) at 5% level:")
        # Method a: Normal distribution with EWMA
        print("\nMethod a: Normal distribution with EWMA (lambda=0.97)")
        print("VaR (5%):")
        for asset, var in results['normal_ewma']['var'].items():
            print(f"{asset}: ${var:.2f}")
        print("\nES (5%):")
        for asset, es in results['normal_ewma']['es'].items():
            print(f"{asset}: ${es:.2f}")
        # Method b: T-distribution with Gaussian Copula
        print("\nMethod b: T-distribution with Gaussian Copula")
        print(f"Estimated degrees of freedom: {results['t_dist']['df']:.2f}")
        print("VaR (5%):")
        for asset, var in results['t_dist']['var'].items():
            print(f"{asset}: ${var:.2f}")
        print("\nES (5%):")
        for asset, es in results['t_dist']['es'].items():
            print(f"{asset}: ${es:.2f}")
        # Method c: Historical simulation
        print("\nMethod c: Historical Simulation")
        print("VaR (5%):")
        for asset, var in results['historical']['var'].items():
            print(f"{asset}: ${var:.2f}")
        print("\nES (5%):")
        for asset, es in results['historical']['es'].items():
            print(f"{asset}: ${es:.2f}")
        # Create comparison plots
        plot_var_comparison(results)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

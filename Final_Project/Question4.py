import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skewnorm, t
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_theme(style="whitegrid")

def read_data():
    """Read data from previous analysis"""
    try:
        # Try to read the stock returns from previous analysis
        stock_returns = pd.read_csv('stock_returns.csv', index_col=0, parse_dates=True)
        portfolio_returns = pd.read_csv('portfolio_returns.csv')
        return stock_returns, portfolio_returns
    except:
        # If files don't exist, create sample data
        print("Data files not found. Creating sample data.")
        # Generate sample returns for 30 stocks
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        symbols = [f'STOCK_{i}' for i in range(1, 31)]
        
        # Create a DataFrame with random returns
        returns_data = {}
        for symbol in symbols:
            # Generate returns with different skewness and kurtosis
            if symbol in ['STOCK_1', 'STOCK_2', 'STOCK_3']:
                # High positive skewness
                returns = skewnorm.rvs(a=5, loc=0.0005, scale=0.01, size=len(dates))
            elif symbol in ['STOCK_4', 'STOCK_5', 'STOCK_6']:
                # High negative skewness
                returns = skewnorm.rvs(a=-5, loc=0.0005, scale=0.01, size=len(dates))
            elif symbol in ['STOCK_7', 'STOCK_8', 'STOCK_9']:
                # High kurtosis (fat tails)
                returns = np.random.standard_t(df=3, size=len(dates)) * 0.01 + 0.0005
            else:
                # Normal distribution
                returns = np.random.normal(loc=0.0005, scale=0.01, size=len(dates))
            
            returns_data[symbol] = returns
        
        stock_returns = pd.DataFrame(returns_data, index=dates)
        
        # Create sample portfolio returns
        portfolio_returns = pd.DataFrame({
            'Portfolio': ['A', 'B', 'C'],
            'Total Return': [0.1468, 0.4128, 0.3800],
            'Systematic Return': [0.3447, 0.4023, 0.3634],
            'Idiosyncratic Return': [-1.8055, -0.5453, 2.8786],
            'Total Risk': [0.0002, 0.0002, 0.0001],
            'Systematic Risk': [0.0000, 0.0000, 0.0000],
            'Idiosyncratic Risk': [0.0001, 0.0001, 0.0001]
        })
        
        return stock_returns, portfolio_returns

def read_portfolio_weights():
    """Read portfolio weights from previous analysis"""
    try:
        # Try to read the portfolio weights from previous analysis
        portfolio_weights = pd.read_csv('portfolio_weights.csv')
        return portfolio_weights
    except:
        # If file doesn't exist, create sample data
        print("Portfolio weights file not found. Creating sample data.")
        # Create sample portfolio weights
        portfolios = ['A', 'B', 'C']
        symbols = [f'STOCK_{i}' for i in range(1, 31)]
        
        # Create a DataFrame with random weights
        weights_data = []
        for portfolio in portfolios:
            # Generate random weights that sum to 1
            weights = np.random.random(len(symbols))
            weights = weights / weights.sum()
            
            for i, symbol in enumerate(symbols):
                weights_data.append({
                    'Portfolio': portfolio,
                    'Symbol': symbol,
                    'Weight': weights[i]
                })
        
        portfolio_weights = pd.DataFrame(weights_data)
        return portfolio_weights

def fit_normal_distribution(data):
    """Fit a normal distribution to the data"""
    mu, sigma = norm.fit(data)
    return {'mu': mu, 'sigma': sigma}

def fit_skew_normal_distribution(data):
    """Fit a skew normal distribution to the data"""
    # Initial guess for parameters
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    alpha_init = 0  # No skewness initially
    
    # Define the negative log-likelihood function
    def neg_log_likelihood(params):
        mu, sigma, alpha = params
        return -np.sum(skewnorm.logpdf(data, a=alpha, loc=mu, scale=sigma))
    
    # Minimize the negative log-likelihood
    result = minimize(neg_log_likelihood, [mu_init, sigma_init, alpha_init], 
                     bounds=[(None, None), (0.0001, None), (None, None)])
    
    if result.success:
        return {'mu': result.x[0], 'sigma': result.x[1], 'alpha': result.x[2]}
    else:
        return {'mu': mu_init, 'sigma': sigma_init, 'alpha': alpha_init}

def fit_generalized_t_distribution(data):
    """Fit a generalized t distribution to the data"""
    # Initial guess for parameters
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    df_init = 5  # Initial degrees of freedom
    
    # Define the negative log-likelihood function
    def neg_log_likelihood(params):
        mu, sigma, df = params
        return -np.sum(t.logpdf(data, df=df, loc=mu, scale=sigma))
    
    # Minimize the negative log-likelihood
    result = minimize(neg_log_likelihood, [mu_init, sigma_init, df_init], 
                     bounds=[(None, None), (0.0001, None), (2.1, None)])
    
    if result.success:
        return {'mu': result.x[0], 'sigma': result.x[1], 'df': result.x[2]}
    else:
        return {'mu': mu_init, 'sigma': sigma_init, 'df': df_init}

def fit_nig_distribution(data):
    """Fit a Normal Inverse Gaussian (NIG) distribution to the data"""
    # This is a simplified implementation
    # For a full implementation, you would need a proper NIG distribution library
    
    # For demonstration, we'll use a mixture of normal distributions to approximate NIG
    # In practice, you would use a proper NIG distribution implementation
    
    # Initial guess for parameters
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    skew_init = stats.skew(data)
    kurt_init = stats.kurtosis(data)
    
    # Define the negative log-likelihood function for a mixture of normals
    def neg_log_likelihood(params):
        mu1, sigma1, mu2, sigma2, p = params
        return -np.sum(np.log(p * norm.pdf(data, mu1, sigma1) + 
                             (1-p) * norm.pdf(data, mu2, sigma2)))
    
    # Minimize the negative log-likelihood
    result = minimize(neg_log_likelihood, [mu_init, sigma_init, mu_init, sigma_init*2, 0.5], 
                     bounds=[(None, None), (0.0001, None), (None, None), (0.0001, None), (0, 1)])
    
    if result.success:
        return {
            'mu1': result.x[0], 
            'sigma1': result.x[1], 
            'mu2': result.x[2], 
            'sigma2': result.x[3], 
            'p': result.x[4]
        }
    else:
        return {
            'mu1': mu_init, 
            'sigma1': sigma_init, 
            'mu2': mu_init, 
            'sigma2': sigma_init*2, 
            'p': 0.5
        }

def calculate_aic(data, params, model_name):
    """Calculate AIC for a fitted model"""
    if model_name == 'normal':
        mu, sigma = params['mu'], params['sigma']
        log_likelihood = np.sum(norm.logpdf(data, mu, sigma))
        k = 2  # number of parameters
    elif model_name == 'skew_normal':
        mu, sigma, alpha = params['mu'], params['sigma'], params['alpha']
        log_likelihood = np.sum(skewnorm.logpdf(data, a=alpha, loc=mu, scale=sigma))
        k = 3  # number of parameters
    elif model_name == 'generalized_t':
        mu, sigma, df = params['mu'], params['sigma'], params['df']
        log_likelihood = np.sum(t.logpdf(data, df=df, loc=mu, scale=sigma))
        k = 3  # number of parameters
    elif model_name == 'nig':
        mu1, sigma1, mu2, sigma2, p = params['mu1'], params['sigma1'], params['mu2'], params['sigma2'], params['p']
        log_likelihood = np.sum(np.log(p * norm.pdf(data, mu1, sigma1) + 
                                      (1-p) * norm.pdf(data, mu2, sigma2)))
        k = 5  # number of parameters
    
    aic = 2 * k - 2 * log_likelihood
    return aic

def find_best_fit(data):
    """Find the best fitting distribution for the data"""
    # Fit all distributions
    normal_params = fit_normal_distribution(data)
    skew_normal_params = fit_skew_normal_distribution(data)
    generalized_t_params = fit_generalized_t_distribution(data)
    nig_params = fit_nig_distribution(data)
    
    # Calculate AIC for each model
    normal_aic = calculate_aic(data, normal_params, 'normal')
    skew_normal_aic = calculate_aic(data, skew_normal_params, 'skew_normal')
    generalized_t_aic = calculate_aic(data, generalized_t_params, 'generalized_t')
    nig_aic = calculate_aic(data, nig_params, 'nig')
    
    # Find the best model
    aics = {
        'normal': normal_aic,
        'skew_normal': skew_normal_aic,
        'generalized_t': generalized_t_aic,
        'nig': nig_aic
    }
    
    best_model = min(aics, key=aics.get)
    
    # Return the best model and its parameters
    if best_model == 'normal':
        return {'model': 'normal', 'params': normal_params}
    elif best_model == 'skew_normal':
        return {'model': 'skew_normal', 'params': skew_normal_params}
    elif best_model == 'generalized_t':
        return {'model': 'generalized_t', 'params': generalized_t_params}
    elif best_model == 'nig':
        return {'model': 'nig', 'params': nig_params}

def calculate_var_es(data, params, model_name, alpha=0.01):
    """Calculate VaR and ES for a fitted model"""
    if model_name == 'normal':
        mu, sigma = params['mu'], params['sigma']
        var = norm.ppf(alpha, mu, sigma)
        es = mu - sigma * norm.pdf(norm.ppf(alpha, 0, 1), 0, 1) / alpha
    elif model_name == 'skew_normal':
        mu, sigma, alpha_param = params['mu'], params['sigma'], params['alpha']
        # For skew normal, we need to use numerical integration for ES
        var = skewnorm.ppf(alpha, a=alpha_param, loc=mu, scale=sigma)
        
        # Numerical integration for ES
        x = np.linspace(var, var - 10*sigma, 1000)
        pdf = skewnorm.pdf(x, a=alpha_param, loc=mu, scale=sigma)
        es = np.trapz(x * pdf, x) / alpha
    elif model_name == 'generalized_t':
        mu, sigma, df = params['mu'], params['sigma'], params['df']
        var = t.ppf(alpha, df=df, loc=mu, scale=sigma)
        
        # For t-distribution, ES can be calculated analytically
        if df > 1:
            es = mu - sigma * (df + t.ppf(alpha, df=df)**2) / (df - 1) * t.pdf(t.ppf(alpha, df=df), df=df) / alpha
        else:
            # If df <= 1, ES is undefined, use numerical integration
            x = np.linspace(var, var - 10*sigma, 1000)
            pdf = t.pdf(x, df=df, loc=mu, scale=sigma)
            es = np.trapz(x * pdf, x) / alpha
    elif model_name == 'nig':
        mu1, sigma1, mu2, sigma2, p = params['mu1'], params['sigma1'], params['mu2'], params['sigma2'], params['p']
        
        # For NIG (mixture), we need to use numerical integration
        # First, find VaR
        x = np.linspace(min(data), max(data), 1000)
        cdf = p * norm.cdf(x, mu1, sigma1) + (1-p) * norm.cdf(x, mu2, sigma2)
        var_idx = np.argmin(np.abs(cdf - alpha))
        var = x[var_idx]
        
        # Then, calculate ES
        x = np.linspace(var, var - 10*max(sigma1, sigma2), 1000)
        pdf = p * norm.pdf(x, mu1, sigma1) + (1-p) * norm.pdf(x, mu2, sigma2)
        es = np.trapz(x * pdf, x) / alpha
    
    return var, es

def calculate_portfolio_var_es_gaussian_copula(stock_returns, portfolio_weights, fitted_models, alpha=0.01):
    """Calculate portfolio VaR and ES using Gaussian Copula"""
    # Extract portfolio weights
    portfolios = portfolio_weights['Portfolio'].unique()
    
    # Calculate correlation matrix
    correlation_matrix = stock_returns.corr()
    
    # Generate correlated standard normal variables
    n_stocks = len(stock_returns.columns)
    n_simulations = 10000
    
    # Generate correlated standard normal variables
    correlated_normals = np.random.multivariate_normal(
        mean=np.zeros(n_stocks),
        cov=correlation_matrix,
        size=n_simulations
    )
    
    # Convert to uniform variables
    uniform_variables = norm.cdf(correlated_normals)
    
    # Initialize results
    results = {}
    
    # Calculate VaR and ES for each portfolio
    for portfolio in portfolios:
        # Get weights for this portfolio
        weights = portfolio_weights[portfolio_weights['Portfolio'] == portfolio]
        
        # Initialize portfolio returns
        portfolio_returns_sim = np.zeros(n_simulations)
        
        # Calculate portfolio returns for each simulation
        for i in range(n_simulations):
            for _, row in weights.iterrows():
                symbol = row['Symbol']
                weight = row['Weight']
                
                if symbol in stock_returns.columns and symbol in fitted_models:
                    # Get the fitted model
                    model = fitted_models[symbol]
                    
                    # Generate return from the fitted distribution
                    if model['model'] == 'normal':
                        mu, sigma = model['params']['mu'], model['params']['sigma']
                        # Use inverse CDF method
                        u = uniform_variables[i, stock_returns.columns.get_loc(symbol)]
                        ret = norm.ppf(u, mu, sigma)
                    elif model['model'] == 'skew_normal':
                        mu, sigma, alpha = model['params']['mu'], model['params']['sigma'], model['params']['alpha']
                        # Use inverse CDF method
                        u = uniform_variables[i, stock_returns.columns.get_loc(symbol)]
                        ret = skewnorm.ppf(u, a=alpha, loc=mu, scale=sigma)
                    elif model['model'] == 'generalized_t':
                        mu, sigma, df = model['params']['mu'], model['params']['sigma'], model['params']['df']
                        # Use inverse CDF method
                        u = uniform_variables[i, stock_returns.columns.get_loc(symbol)]
                        ret = t.ppf(u, df=df, loc=mu, scale=sigma)
                    elif model['model'] == 'nig':
                        mu1, sigma1, mu2, sigma2, p = model['params']['mu1'], model['params']['sigma1'], model['params']['mu2'], model['params']['sigma2'], model['params']['p']
                        # For NIG (mixture), we need to use a more complex approach
                        # For simplicity, we'll use a weighted average of two normal distributions
                        u = uniform_variables[i, stock_returns.columns.get_loc(symbol)]
                        if u < p:
                            ret = norm.ppf(u/p, mu1, sigma1)
                        else:
                            ret = norm.ppf((u-p)/(1-p), mu2, sigma2)
                    
                    # Add to portfolio return
                    portfolio_returns_sim[i] += weight * ret
        
        # Calculate VaR and ES
        var = np.percentile(portfolio_returns_sim, alpha * 100)
        es = np.mean(portfolio_returns_sim[portfolio_returns_sim <= var])
        
        results[portfolio] = {'var': var, 'es': es}
    
    # Calculate VaR and ES for the total portfolio
    # Assume equal weights for the total portfolio
    total_weights = pd.DataFrame({
        'Portfolio': ['Total'] * len(stock_returns.columns),
        'Symbol': stock_returns.columns,
        'Weight': 1.0 / len(stock_returns.columns)
    })
    
    # Initialize portfolio returns
    portfolio_returns_sim = np.zeros(n_simulations)
    
    # Calculate portfolio returns for each simulation
    for i in range(n_simulations):
        for _, row in total_weights.iterrows():
            symbol = row['Symbol']
            weight = row['Weight']
            
            if symbol in fitted_models:
                # Get the fitted model
                model = fitted_models[symbol]
                
                # Generate return from the fitted distribution
                if model['model'] == 'normal':
                    mu, sigma = model['params']['mu'], model['params']['sigma']
                    # Use inverse CDF method
                    u = uniform_variables[i, stock_returns.columns.get_loc(symbol)]
                    ret = norm.ppf(u, mu, sigma)
                elif model['model'] == 'skew_normal':
                    mu, sigma, alpha = model['params']['mu'], model['params']['sigma'], model['params']['alpha']
                    # Use inverse CDF method
                    u = uniform_variables[i, stock_returns.columns.get_loc(symbol)]
                    ret = skewnorm.ppf(u, a=alpha, loc=mu, scale=sigma)
                elif model['model'] == 'generalized_t':
                    mu, sigma, df = model['params']['mu'], model['params']['sigma'], model['params']['df']
                    # Use inverse CDF method
                    u = uniform_variables[i, stock_returns.columns.get_loc(symbol)]
                    ret = t.ppf(u, df=df, loc=mu, scale=sigma)
                elif model['model'] == 'nig':
                    mu1, sigma1, mu2, sigma2, p = model['params']['mu1'], model['params']['sigma1'], model['params']['mu2'], model['params']['sigma2'], model['params']['p']
                    # For NIG (mixture), we need to use a more complex approach
                    # For simplicity, we'll use a weighted average of two normal distributions
                    u = uniform_variables[i, stock_returns.columns.get_loc(symbol)]
                    if u < p:
                        ret = norm.ppf(u/p, mu1, sigma1)
                    else:
                        ret = norm.ppf((u-p)/(1-p), mu2, sigma2)
                
                # Add to portfolio return
                portfolio_returns_sim[i] += weight * ret
    
    # Calculate VaR and ES
    var = np.percentile(portfolio_returns_sim, alpha * 100)
    es = np.mean(portfolio_returns_sim[portfolio_returns_sim <= var])
    
    results['Total'] = {'var': var, 'es': es}
    
    return results

def calculate_portfolio_var_es_multivariate_normal(stock_returns, portfolio_weights, alpha=0.01):
    """Calculate portfolio VaR and ES using multivariate normal distribution"""
    # Extract portfolio weights
    portfolios = portfolio_weights['Portfolio'].unique()
    
    # Calculate mean and covariance matrix
    mean_returns = stock_returns.mean()
    cov_matrix = stock_returns.cov()
    
    # Initialize results
    results = {}
    
    # Calculate VaR and ES for each portfolio
    for portfolio in portfolios:
        # Get weights for this portfolio
        weights = portfolio_weights[portfolio_weights['Portfolio'] == portfolio]
        
        # Extract weights as a vector
        weight_vector = np.zeros(len(stock_returns.columns))
        for _, row in weights.iterrows():
            symbol = row['Symbol']
            weight = row['Weight']
            if symbol in stock_returns.columns:
                weight_vector[stock_returns.columns.get_loc(symbol)] = weight
        
        # Calculate portfolio mean and variance
        portfolio_mean = np.dot(weight_vector, mean_returns)
        portfolio_var = np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector))
        portfolio_std = np.sqrt(portfolio_var)
        
        # Calculate VaR and ES
        var = portfolio_mean - portfolio_std * norm.ppf(1 - alpha, 0, 1)
        es = portfolio_mean - portfolio_std * norm.pdf(norm.ppf(1 - alpha, 0, 1), 0, 1) / alpha
        
        results[portfolio] = {'var': var, 'es': es}
    
    # Calculate VaR and ES for the total portfolio
    # Assume equal weights for the total portfolio
    weight_vector = np.ones(len(stock_returns.columns)) / len(stock_returns.columns)
    
    # Calculate portfolio mean and variance
    portfolio_mean = np.dot(weight_vector, mean_returns)
    portfolio_var = np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector))
    portfolio_std = np.sqrt(portfolio_var)
    
    # Calculate VaR and ES
    var = portfolio_mean - portfolio_std * norm.ppf(1 - alpha, 0, 1)
    es = portfolio_mean - portfolio_std * norm.pdf(norm.ppf(1 - alpha, 0, 1), 0, 1) / alpha
    
    results['Total'] = {'var': var, 'es': es}
    
    return results

def main():
    # Read data
    stock_returns, portfolio_returns = read_data()
    portfolio_weights = read_portfolio_weights()
    
    # Fit models to each stock
    print("\nFitting models to each stock:")
    print("-" * 80)
    
    fitted_models = {}
    for symbol in stock_returns.columns:
        print(f"\nFitting models for {symbol}:")
        data = stock_returns[symbol].values
        best_fit = find_best_fit(data)
        fitted_models[symbol] = best_fit
        print(f"Best fit: {best_fit['model']}")
        print(f"Parameters: {best_fit['params']}")
    
    # Calculate VaR and ES using Gaussian Copula
    print("\nCalculating VaR and ES using Gaussian Copula:")
    print("-" * 80)
    
    copula_results = calculate_portfolio_var_es_gaussian_copula(stock_returns, portfolio_weights, fitted_models)
    
    for portfolio, result in copula_results.items():
        print(f"\nPortfolio {portfolio}:")
        print(f"VaR (1%): {result['var']:.4f}")
        print(f"ES (1%): {result['es']:.4f}")
    
    # Calculate VaR and ES using multivariate normal
    print("\nCalculating VaR and ES using multivariate normal:")
    print("-" * 80)
    
    normal_results = calculate_portfolio_var_es_multivariate_normal(stock_returns, portfolio_weights)
    
    for portfolio, result in normal_results.items():
        print(f"\nPortfolio {portfolio}:")
        print(f"VaR (1%): {result['var']:.4f}")
        print(f"ES (1%): {result['es']:.4f}")
    
    # Compare the two approaches
    print("\nComparison of the two approaches:")
    print("-" * 80)
    
    for portfolio in copula_results.keys():
        print(f"\nPortfolio {portfolio}:")
        print(f"Gaussian Copula - VaR (1%): {copula_results[portfolio]['var']:.4f}, ES (1%): {copula_results[portfolio]['es']:.4f}")
        print(f"Multivariate Normal - VaR (1%): {normal_results[portfolio]['var']:.4f}, ES (1%): {normal_results[portfolio]['es']:.4f}")
        print(f"Difference - VaR: {copula_results[portfolio]['var'] - normal_results[portfolio]['var']:.4f}, ES: {copula_results[portfolio]['es'] - normal_results[portfolio]['es']:.4f}")
    
    print("\nAnalysis complete. The results show the differences between using Gaussian Copula and multivariate normal approaches for risk modeling.")

if __name__ == "__main__":
    main()

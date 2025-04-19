import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
                returns = stats.skewnorm.rvs(a=5, loc=0.0005, scale=0.01, size=len(dates))
            elif symbol in ['STOCK_4', 'STOCK_5', 'STOCK_6']:
                # High negative skewness
                returns = stats.skewnorm.rvs(a=-5, loc=0.0005, scale=0.01, size=len(dates))
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

def read_capm_betas():
    """Read CAPM betas from previous analysis"""
    try:
        # Try to read the CAPM betas from previous analysis
        capm_betas = pd.read_csv('capm_betas.csv')
        return capm_betas
    except:
        # If file doesn't exist, create sample data
        print("CAPM betas file not found. Creating sample data.")
        # Create sample CAPM betas
        symbols = [f'STOCK_{i}' for i in range(1, 31)]
        
        # Generate random betas
        betas = np.random.normal(loc=1.0, scale=0.3, size=len(symbols))
        
        capm_betas = pd.DataFrame({
            'Symbol': symbols,
            'Beta': betas
        })
        return capm_betas

def read_fitted_models():
    """Read fitted models from previous analysis"""
    try:
        # Try to read the fitted models from previous analysis
        fitted_models = pd.read_csv('fitted_models.csv')
        return fitted_models
    except:
        # If file doesn't exist, create sample data
        print("Fitted models file not found. Creating sample data.")
        # Create sample fitted models
        symbols = [f'STOCK_{i}' for i in range(1, 31)]
        
        # Generate random models
        models = []
        for symbol in symbols:
            if symbol in ['STOCK_1', 'STOCK_2', 'STOCK_3', 'STOCK_4', 'STOCK_5', 'STOCK_6', 'STOCK_20', 'STOCK_27']:
                model = 'nig'
            elif symbol in ['STOCK_7', 'STOCK_8', 'STOCK_9']:
                model = 'generalized_t'
            else:
                model = 'normal'
            
            models.append({
                'Symbol': symbol,
                'Model': model
            })
        
        fitted_models = pd.DataFrame(models)
        return fitted_models

def calculate_es(returns, weights, alpha=0.01):
    """Calculate Expected Shortfall (ES) for a portfolio"""
    # Calculate portfolio returns
    portfolio_returns = np.sum(returns * weights, axis=1)
    
    # Calculate VaR
    var = np.percentile(portfolio_returns, alpha * 100)
    
    # Calculate ES
    es = np.mean(portfolio_returns[portfolio_returns <= var])
    
    return es

def calculate_risk_contribution(returns, weights, alpha=0.01):
    """Calculate risk contribution of each asset to the portfolio ES"""
    # Calculate portfolio returns
    portfolio_returns = np.sum(returns * weights, axis=1)
    
    # Calculate VaR
    var = np.percentile(portfolio_returns, alpha * 100)
    
    # Calculate ES
    es = np.mean(portfolio_returns[portfolio_returns <= var])
    
    # Calculate risk contribution
    risk_contribution = np.zeros(len(weights))
    for i in range(len(weights)):
        # Calculate marginal contribution to ES
        mces = np.mean(returns[:, i][portfolio_returns <= var])
        
        # Calculate risk contribution
        risk_contribution[i] = weights[i] * mces / es
    
    return risk_contribution

def risk_parity_objective(weights, returns, alpha=0.01):
    """Objective function for risk parity optimization"""
    # Calculate risk contribution
    risk_contribution = calculate_risk_contribution(returns, weights, alpha)
    
    # Calculate target risk contribution (equal for all assets)
    target_risk = 1.0 / len(weights)
    
    # Calculate sum of squared differences
    sum_squared_diff = np.sum((risk_contribution - target_risk) ** 2)
    
    return sum_squared_diff

def optimize_risk_parity_portfolio(returns, initial_weights=None, alpha=0.01):
    """Optimize a risk parity portfolio using ES as the risk metric"""
    n_assets = returns.shape[1]
    
    # Set initial weights if not provided
    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets
    
    # Set constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
    ]
    
    # Set bounds
    bounds = [(0, 1) for _ in range(n_assets)]  # Weights between 0 and 1
    
    # Optimize
    result = minimize(
        risk_parity_objective,
        initial_weights,
        args=(returns, alpha),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def calculate_portfolio_returns(returns, weights):
    """Calculate portfolio returns"""
    return np.sum(returns * weights, axis=1)

def calculate_portfolio_attribution(returns, weights, betas, market_returns):
    """Calculate portfolio attribution"""
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    
    # Calculate market returns (if not provided)
    if market_returns is None:
        market_returns = np.mean(returns, axis=1)
    
    # Calculate systematic returns
    systematic_returns = np.zeros_like(portfolio_returns)
    for i, beta in enumerate(betas):
        systematic_returns += weights[i] * beta * market_returns
    
    # Calculate idiosyncratic returns
    idiosyncratic_returns = portfolio_returns - systematic_returns
    
    # Calculate total return
    total_return = np.mean(portfolio_returns) * 252  # Annualized
    
    # Calculate systematic return
    systematic_return = np.mean(systematic_returns) * 252  # Annualized
    
    # Calculate idiosyncratic return
    idiosyncratic_return = np.mean(idiosyncratic_returns) * 252  # Annualized
    
    # Calculate total risk
    total_risk = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
    
    # Calculate systematic risk
    systematic_risk = np.std(systematic_returns) * np.sqrt(252)  # Annualized
    
    # Calculate idiosyncratic risk
    idiosyncratic_risk = np.std(idiosyncratic_returns) * np.sqrt(252)  # Annualized
    
    return {
        'Total Return': total_return,
        'Systematic Return': systematic_return,
        'Idiosyncratic Return': idiosyncratic_return,
        'Total Risk': total_risk,
        'Systematic Risk': systematic_risk,
        'Idiosyncratic Risk': idiosyncratic_risk
    }

def main():
    # Read data
    stock_returns, portfolio_returns = read_data()
    portfolio_weights = read_portfolio_weights()
    capm_betas = read_capm_betas()
    fitted_models = read_fitted_models()
    
    # Convert stock returns to numpy array
    returns_array = stock_returns.values
    
    # Extract unique portfolios
    portfolios = portfolio_weights['Portfolio'].unique()
    
    # Initialize results
    risk_parity_weights = {}
    attribution_results = {}
    
    # Calculate risk parity portfolios for each sub-portfolio
    print("\nCalculating risk parity portfolios:")
    print("-" * 80)
    
    for portfolio in portfolios:
        print(f"\nPortfolio {portfolio}:")
        
        # Get stocks in this portfolio
        portfolio_stocks = portfolio_weights[portfolio_weights['Portfolio'] == portfolio]['Symbol'].values
        
        # Get returns for these stocks
        portfolio_returns_data = stock_returns[portfolio_stocks].values
        
        # Get initial weights
        initial_weights = portfolio_weights[portfolio_weights['Portfolio'] == portfolio]['Weight'].values
        
        # Optimize risk parity portfolio
        optimal_weights = optimize_risk_parity_portfolio(portfolio_returns_data, initial_weights)
        
        # Store weights
        risk_parity_weights[portfolio] = dict(zip(portfolio_stocks, optimal_weights))
        
        # Print weights
        for stock, weight in risk_parity_weights[portfolio].items():
            print(f"{stock}: {weight:.4f}")
        
        # Calculate ES
        es = calculate_es(portfolio_returns_data, optimal_weights)
        print(f"Portfolio ES (1%): {es:.4f}")
        
        # Calculate risk contribution
        risk_contribution = calculate_risk_contribution(portfolio_returns_data, optimal_weights)
        
        # Print risk contribution
        print("Risk Contribution:")
        for i, stock in enumerate(portfolio_stocks):
            print(f"{stock}: {risk_contribution[i]:.4f}")
    
    # Calculate attribution for each portfolio
    print("\nCalculating attribution for each portfolio:")
    print("-" * 80)
    
    # Calculate market returns (average of all stocks)
    market_returns = np.mean(returns_array, axis=1)
    
    for portfolio in portfolios:
        print(f"\nPortfolio {portfolio}:")
        
        # Get stocks in this portfolio
        portfolio_stocks = list(risk_parity_weights[portfolio].keys())
        
        # Get weights
        weights = np.array([risk_parity_weights[portfolio][stock] for stock in portfolio_stocks])
        
        # Get returns for these stocks
        portfolio_returns_data = stock_returns[portfolio_stocks].values
        
        # Get betas for these stocks
        portfolio_betas = np.array([capm_betas[capm_betas['Symbol'] == stock]['Beta'].values[0] for stock in portfolio_stocks])
        
        # Calculate attribution
        attribution = calculate_portfolio_attribution(portfolio_returns_data, weights, portfolio_betas, market_returns)
        
        # Store results
        attribution_results[portfolio] = attribution
        
        # Print results
        print(f"Total Return: {attribution['Total Return']:.4f}")
        print(f"Systematic Return: {attribution['Systematic Return']:.4f}")
        print(f"Idiosyncratic Return: {attribution['Idiosyncratic Return']:.4f}")
        print(f"Total Risk: {attribution['Total Risk']:.4f}")
        print(f"Systematic Risk: {attribution['Systematic Risk']:.4f}")
        print(f"Idiosyncratic Risk: {attribution['Idiosyncratic Risk']:.4f}")
    
    # Compare with Part 1 and Part 2 results
    print("\nComparison with Part 1 and Part 2 results:")
    print("-" * 80)
    
    # Read Part 1 and Part 2 results if available
    try:
        part1_results = pd.read_csv('part1_results.csv')
        part2_results = pd.read_csv('part2_results.csv')
        
        for portfolio in portfolios:
            print(f"\nPortfolio {portfolio}:")
            
            # Get Part 1 results
            part1_result = part1_results[part1_results['Portfolio'] == portfolio].iloc[0]
            
            # Get Part 2 results
            part2_result = part2_results[part2_results['Portfolio'] == portfolio].iloc[0]
            
            # Get Part 5 results
            part5_result = attribution_results[portfolio]
            
            # Print comparison
            print("Part 1 (Initial Portfolio):")
            print(f"Total Return: {part1_result['Total Return']:.4f}")
            print(f"Systematic Return: {part1_result['Systematic Return']:.4f}")
            print(f"Idiosyncratic Return: {part1_result['Idiosyncratic Return']:.4f}")
            print(f"Total Risk: {part1_result['Total Risk']:.4f}")
            print(f"Systematic Risk: {part1_result['Systematic Risk']:.4f}")
            print(f"Idiosyncratic Risk: {part1_result['Idiosyncratic Risk']:.4f}")
            
            print("\nPart 2 (Maximum Sharpe Ratio Portfolio):")
            print(f"Total Return: {part2_result['Total Return']:.4f}")
            print(f"Systematic Return: {part2_result['Systematic Return']:.4f}")
            print(f"Idiosyncratic Return: {part2_result['Idiosyncratic Return']:.4f}")
            print(f"Total Risk: {part2_result['Total Risk']:.4f}")
            print(f"Systematic Risk: {part2_result['Systematic Risk']:.4f}")
            print(f"Idiosyncratic Risk: {part2_result['Idiosyncratic Risk']:.4f}")
            
            print("\nPart 5 (Risk Parity Portfolio):")
            print(f"Total Return: {part5_result['Total Return']:.4f}")
            print(f"Systematic Return: {part5_result['Systematic Return']:.4f}")
            print(f"Idiosyncratic Return: {part5_result['Idiosyncratic Return']:.4f}")
            print(f"Total Risk: {part5_result['Total Risk']:.4f}")
            print(f"Systematic Risk: {part5_result['Systematic Risk']:.4f}")
            print(f"Idiosyncratic Risk: {part5_result['Idiosyncratic Risk']:.4f}")
    except:
        print("Part 1 and Part 2 results not available. Using sample data for comparison.")
        
        # Create sample Part 1 and Part 2 results
        part1_results = {
            'A': {
                'Total Return': 0.3066,
                'Systematic Return': 0.6033,
                'Idiosyncratic Return': -1.2967,
                'Total Risk': 0.0002,
                'Systematic Risk': 0.0000,
                'Idiosyncratic Risk': 0.0001
            },
            'B': {
                'Total Return': 0.4179,
                'Systematic Return': 0.4628,
                'Idiosyncratic Return': -0.0449,
                'Total Risk': 0.0002,
                'Systematic Risk': 0.0000,
                'Idiosyncratic Risk': 0.0001
            },
            'C': {
                'Total Return': 0.6146,
                'Systematic Return': 0.8987,
                'Idiosyncratic Return': -0.6961,
                'Total Risk': 0.0001,
                'Systematic Risk': 0.0000,
                'Idiosyncratic Risk': 0.0001
            }
        }
        
        part2_results = {
            'A': {
                'Total Return': 0.1468,
                'Systematic Return': 0.3447,
                'Idiosyncratic Return': -1.8055,
                'Total Risk': 0.0002,
                'Systematic Risk': 0.0000,
                'Idiosyncratic Risk': 0.0001
            },
            'B': {
                'Total Return': 0.4128,
                'Systematic Return': 0.4023,
                'Idiosyncratic Return': -0.5453,
                'Total Risk': 0.0002,
                'Systematic Risk': 0.0000,
                'Idiosyncratic Risk': 0.0001
            },
            'C': {
                'Total Return': 0.3800,
                'Systematic Return': 0.3634,
                'Idiosyncratic Return': 2.8786,
                'Total Risk': 0.0001,
                'Systematic Risk': 0.0000,
                'Idiosyncratic Risk': 0.0001
            }
        }
        
        for portfolio in portfolios:
            print(f"\nPortfolio {portfolio}:")
            
            # Get Part 1 results
            part1_result = part1_results[portfolio]
            
            # Get Part 2 results
            part2_result = part2_results[portfolio]
            
            # Get Part 5 results
            part5_result = attribution_results[portfolio]
            
            # Print comparison
            print("Part 1 (Initial Portfolio):")
            print(f"Total Return: {part1_result['Total Return']:.4f}")
            print(f"Systematic Return: {part1_result['Systematic Return']:.4f}")
            print(f"Idiosyncratic Return: {part1_result['Idiosyncratic Return']:.4f}")
            print(f"Total Risk: {part1_result['Total Risk']:.4f}")
            print(f"Systematic Risk: {part1_result['Systematic Risk']:.4f}")
            print(f"Idiosyncratic Risk: {part1_result['Idiosyncratic Risk']:.4f}")
            
            print("\nPart 2 (Maximum Sharpe Ratio Portfolio):")
            print(f"Total Return: {part2_result['Total Return']:.4f}")
            print(f"Systematic Return: {part2_result['Systematic Return']:.4f}")
            print(f"Idiosyncratic Return: {part2_result['Idiosyncratic Return']:.4f}")
            print(f"Total Risk: {part2_result['Total Risk']:.4f}")
            print(f"Systematic Risk: {part2_result['Systematic Risk']:.4f}")
            print(f"Idiosyncratic Risk: {part2_result['Idiosyncratic Risk']:.4f}")
            
            print("\nPart 5 (Risk Parity Portfolio):")
            print(f"Total Return: {part5_result['Total Return']:.4f}")
            print(f"Systematic Return: {part5_result['Systematic Return']:.4f}")
            print(f"Idiosyncratic Return: {part5_result['Idiosyncratic Return']:.4f}")
            print(f"Total Risk: {part5_result['Total Risk']:.4f}")
            print(f"Systematic Risk: {part5_result['Systematic Risk']:.4f}")
            print(f"Idiosyncratic Risk: {part5_result['Idiosyncratic Risk']:.4f}")
    
    print("\nAnalysis complete. The results show the differences between the initial portfolios, maximum Sharpe ratio portfolios, and risk parity portfolios.")

if __name__ == "__main__":
    main()

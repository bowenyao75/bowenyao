import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Set seaborn style
sns.set_theme(style="whitegrid")

def read_initial_portfolio():
    """Read and display initial portfolio data"""
    portfolio_df = pd.read_csv('initial_portfolio.csv')
    print("\nInitial Portfolio Data Structure:")
    print(portfolio_df.info())
    print("\nInitial Portfolio Preview:")
    print(portfolio_df.head())
    return portfolio_df

def read_risk_free_rate():
    """Read and display risk-free rate data"""
    rf_df = pd.read_csv('rf.csv')
    print("\nRisk-Free Rate Data Structure:")
    print(rf_df.info())
    print("\nRisk-Free Rate Preview:")
    print(rf_df.head())
    return rf_df

def read_daily_prices():
    """Read and display daily price data"""
    prices_df = pd.read_csv('DailyPrices.csv')
    print("\nDaily Prices Data Structure:")
    print(prices_df.info())
    print("\nDaily Prices Preview:")
    print(prices_df.head())
    return prices_df

def calculate_returns(prices_df):
    """Calculate daily returns for all assets"""
    # Convert Date to datetime and set as index
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    prices_df = prices_df.set_index('Date')
    
    # Calculate returns for all columns
    returns_df = prices_df.pct_change()
    returns_df = returns_df.fillna(0)  # Handle first day NA values
    return returns_df

def run_capm_analysis(returns_df, rf_df, portfolio_df, end_date='2023-12-31'):
    """Perform CAPM analysis for all stocks in portfolios using data up to end_date"""
    # Prepare market returns and risk-free rate
    market_returns = returns_df['SPY']
    rf_df['Date'] = pd.to_datetime(rf_df['Date'])
    rf_df = rf_df.set_index('Date')
    rf_rates = rf_df['rf']
    
    # Filter data up to end_date
    returns_df_filtered = returns_df[returns_df.index <= end_date]
    market_returns_filtered = market_returns[market_returns.index <= end_date]
    rf_rates_filtered = rf_rates[rf_rates.index <= end_date]
    
    # Align dates between returns and risk-free rates
    aligned_dates = returns_df_filtered.index.intersection(rf_rates_filtered.index)
    market_returns_filtered = market_returns_filtered[aligned_dates]
    rf_rates_filtered = rf_rates_filtered[aligned_dates]
    
    # Calculate excess returns
    excess_returns = returns_df_filtered.loc[aligned_dates].sub(rf_rates_filtered, axis=0)
    market_excess_returns = market_returns_filtered - rf_rates_filtered
    
    # Calculate expected market return (average of historical market returns)
    expected_market_return = market_returns_filtered.mean()
    expected_rf_rate = rf_rates_filtered.mean()
    
    capm_results = {}
    for portfolio in portfolio_df['Portfolio'].unique():
        portfolio_stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]
        stock_results = {}
        
        for _, row in portfolio_stocks.iterrows():
            symbol = row['Symbol']
            if symbol in returns_df.columns:
                # Run regression
                X = market_excess_returns.values.reshape(-1, 1)
                y = excess_returns[symbol].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
                
                # Calculate idiosyncratic risk
                y_pred = slope * X.flatten() + intercept
                residuals = y - y_pred
                idiosyncratic_risk = np.var(residuals)
                
                stock_results[symbol] = {
                    'beta': slope,
                    'alpha': intercept,
                    'r_squared': r_value**2,
                    'holding': row['Holding'],
                    'idiosyncratic_risk': idiosyncratic_risk,
                    'expected_return': expected_rf_rate + slope * (expected_market_return - expected_rf_rate)
                }
        
        capm_results[portfolio] = stock_results
    
    return capm_results, expected_market_return, expected_rf_rate

def create_optimal_portfolios(capm_results, portfolio_df):
    """Create optimal maximum Sharpe ratio portfolios for each sub-portfolio"""
    optimal_portfolios = {}
    
    for portfolio, stock_results in capm_results.items():
        portfolio_stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]
        symbols = [row['Symbol'] for _, row in portfolio_stocks.iterrows() if row['Symbol'] in stock_results]
        
        if not symbols:
            continue
        
        # Extract expected returns and betas
        expected_returns = np.array([stock_results[symbol]['expected_return'] for symbol in symbols])
        betas = np.array([stock_results[symbol]['beta'] for symbol in symbols])
        idiosyncratic_risks = np.array([stock_results[symbol]['idiosyncratic_risk'] for symbol in symbols])
        
        # Create covariance matrix (simplified using CAPM)
        market_variance = 0.0001  # Approximate market variance
        cov_matrix = np.zeros((len(symbols), len(symbols)))
        
        for i in range(len(symbols)):
            for j in range(len(symbols)):
                if i == j:
                    # Diagonal elements include idiosyncratic risk
                    cov_matrix[i, j] = (betas[i] ** 2) * market_variance + idiosyncratic_risks[i]
                else:
                    # Off-diagonal elements only include systematic risk
                    cov_matrix[i, j] = betas[i] * betas[j] * market_variance
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_risk
            return -sharpe_ratio
        
        # Constraints
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        )
        bounds = tuple((0, 1) for _ in range(len(symbols)))  # Weights between 0 and 1
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/len(symbols)] * len(symbols))
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            optimal_portfolios[portfolio] = dict(zip(symbols, optimal_weights))
        else:
            print(f"Optimization failed for portfolio {portfolio}")
            optimal_portfolios[portfolio] = dict(zip(symbols, initial_weights))
    
    return optimal_portfolios

def calculate_portfolio_values(prices_df, portfolio_df, optimal_weights):
    """Calculate daily portfolio values for each portfolio using optimal weights"""
    portfolio_values = {}
    
    for portfolio in portfolio_df['Portfolio'].unique():
        portfolio_stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]
        daily_values = pd.DataFrame(index=prices_df.index)
        
        if portfolio in optimal_weights:
            for symbol, weight in optimal_weights[portfolio].items():
                if symbol in prices_df.columns:
                    # Calculate value based on weight
                    portfolio_value = 1000000  # Assume $1M initial investment
                    holding = weight * portfolio_value / prices_df[symbol].iloc[0]
                    daily_values[symbol] = prices_df[symbol] * holding
            
            portfolio_values[portfolio] = daily_values.sum(axis=1)
    
    return portfolio_values

def calculate_risk_attribution(returns_df, capm_results, portfolio_df, portfolio_values, optimal_weights):
    """Calculate risk attribution for each portfolio using optimal weights"""
    # Get market data
    market_returns = returns_df['SPY']
    market_var = np.var(market_returns)
    
    # Calculate market return for the period
    market_start = market_returns.iloc[1]  # Use second day to avoid first day return of 0
    market_end = market_returns.iloc[-1]
    market_period_return = (market_end - market_start) / market_start
    
    attribution_results = {}
    for portfolio in capm_results.keys():
        if portfolio not in optimal_weights:
            continue
            
        weights = optimal_weights[portfolio]
        systematic_risk = 0
        idiosyncratic_risk = 0
        systematic_return = 0
        idiosyncratic_return = 0
        
        # Calculate total portfolio return
        initial_value = portfolio_values[portfolio].iloc[1]  # Use second day
        final_value = portfolio_values[portfolio].iloc[-1]
        portfolio_return = (final_value - initial_value) / initial_value
        
        for symbol, weight in weights.items():
            if symbol in capm_results[portfolio]:
                results = capm_results[portfolio][symbol]
                beta = results['beta']
                
                # Calculate risks
                stock_systematic_risk = (beta ** 2) * market_var
                stock_returns = returns_df[symbol]
                stock_total_var = np.var(stock_returns)
                stock_idiosyncratic_risk = stock_total_var - stock_systematic_risk
                
                systematic_risk += weight * stock_systematic_risk
                idiosyncratic_risk += weight * stock_idiosyncratic_risk
                
                # Calculate returns
                stock_start = returns_df[symbol].iloc[1]  # Use second day
                stock_end = returns_df[symbol].iloc[-1]
                stock_return = (stock_end - stock_start) / stock_start
                stock_systematic_return = beta * market_period_return
                stock_idiosyncratic_return = stock_return - stock_systematic_return
                
                systematic_return += weight * stock_systematic_return
                idiosyncratic_return += weight * stock_idiosyncratic_return
        
        attribution_results[portfolio] = {
            'total_return': portfolio_return,
            'systematic_return': systematic_return,
            'idiosyncratic_return': idiosyncratic_return,
            'systematic_risk': systematic_risk,
            'idiosyncratic_risk': idiosyncratic_risk,
            'total_risk': systematic_risk + idiosyncratic_risk
        }
    
    return attribution_results

def compare_expected_vs_realized_idiosyncratic_risk(capm_results, returns_df, optimal_weights):
    """Compare expected vs realized idiosyncratic risk for each stock"""
    comparison_results = {}
    
    for portfolio, stock_results in capm_results.items():
        if portfolio not in optimal_weights:
            continue
            
        portfolio_comparison = {}
        
        for symbol, weight in optimal_weights[portfolio].items():
            if symbol in stock_results and weight > 0.01:  # Only include stocks with significant weights
                expected_idiosyncratic_risk = stock_results[symbol]['idiosyncratic_risk']
                beta = stock_results[symbol]['beta']
                
                # Calculate realized idiosyncratic risk
                market_returns = returns_df['SPY']
                rf_df = pd.read_csv('rf.csv')
                rf_df['Date'] = pd.to_datetime(rf_df['Date'])
                rf_df = rf_df.set_index('Date')
                rf_rates = rf_df['rf']
                
                # Align dates
                aligned_dates = returns_df.index.intersection(rf_rates.index)
                market_returns_aligned = market_returns[aligned_dates]
                rf_rates_aligned = rf_rates[aligned_dates]
                stock_returns_aligned = returns_df[symbol][aligned_dates]
                
                # Calculate excess returns
                market_excess_returns = market_returns_aligned - rf_rates_aligned
                stock_excess_returns = stock_returns_aligned - rf_rates_aligned
                
                # Calculate predicted returns
                predicted_returns = beta * market_excess_returns
                
                # Calculate residuals (realized idiosyncratic returns)
                residuals = stock_excess_returns - predicted_returns
                
                # Calculate realized idiosyncratic risk
                realized_idiosyncratic_risk = np.var(residuals)
                
                portfolio_comparison[symbol] = {
                    'expected_idiosyncratic_risk': expected_idiosyncratic_risk,
                    'realized_idiosyncratic_risk': realized_idiosyncratic_risk,
                    'weight': weight,
                    'beta': beta
                }
        
        comparison_results[portfolio] = portfolio_comparison
    
    return comparison_results

def visualize_results(attribution_results, comparison_results):
    """Create visualizations for the analysis results"""
    # Set the style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Prepare data for plotting
    portfolios = list(attribution_results.keys())
    
    # Returns visualization data
    returns_data = {
        'Portfolio': [],
        'Return Type': [],
        'Value': []
    }
    
    for portfolio in portfolios:
        returns_data['Portfolio'].extend([portfolio] * 3)
        returns_data['Return Type'].extend(['Total', 'Systematic', 'Idiosyncratic'])
        returns_data['Value'].extend([
            attribution_results[portfolio]['total_return'] * 100,
            attribution_results[portfolio]['systematic_return'] * 100,
            attribution_results[portfolio]['idiosyncratic_return'] * 100
        ])
    
    returns_df = pd.DataFrame(returns_data)
    
    # Risk visualization data
    risk_data = {
        'Portfolio': [],
        'Risk Type': [],
        'Value': []
    }
    
    for portfolio in portfolios:
        risk_data['Portfolio'].extend([portfolio] * 3)
        risk_data['Risk Type'].extend(['Total', 'Systematic', 'Idiosyncratic'])
        risk_data['Value'].extend([
            attribution_results[portfolio]['total_risk'] * 100,
            attribution_results[portfolio]['systematic_risk'] * 100,
            attribution_results[portfolio]['idiosyncratic_risk'] * 100
        ])
    
    risk_df = pd.DataFrame(risk_data)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Returns plot
    plt.subplot(2, 1, 1)
    returns_plot = sns.barplot(
        data=returns_df,
        x='Portfolio',
        y='Value',
        hue='Return Type',
        palette='Set2'
    )
    returns_plot.set_title('Optimal Portfolio Returns Attribution', pad=20, fontsize=14)
    returns_plot.set_ylabel('Return (%)', fontsize=12)
    returns_plot.set_xlabel('Portfolio', fontsize=12)
    returns_plot.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Add value labels on the bars
    for container in returns_plot.containers:
        returns_plot.bar_label(container, fmt='%.1f%%', padding=3)
    
    # Adjust legend
    plt.legend(title='Return Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Risk plot
    plt.subplot(2, 1, 2)
    risk_plot = sns.barplot(
        data=risk_df,
        x='Portfolio',
        y='Value',
        hue='Risk Type',
        palette='Set3'
    )
    risk_plot.set_title('Optimal Portfolio Risk Attribution', pad=20, fontsize=14)
    risk_plot.set_ylabel('Risk (%)', fontsize=12)
    risk_plot.set_xlabel('Portfolio', fontsize=12)
    
    # Add value labels on the bars
    for container in risk_plot.containers:
        risk_plot.bar_label(container, fmt='%.2f%%', padding=3)
    
    # Adjust legend
    plt.legend(title='Risk Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a title for the entire figure
    fig.suptitle('Optimal Portfolio Analysis Results', fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    # Create additional visualizations
    
    # 1. Risk-Return Scatter Plot
    risk_return_data = pd.DataFrame({
        'Portfolio': portfolios,
        'Total Return (%)': [attribution_results[p]['total_return'] * 100 for p in portfolios],
        'Total Risk (%)': [attribution_results[p]['total_risk'] * 100 for p in portfolios]
    })
    
    plt.figure(figsize=(10, 6))
    scatter_plot = sns.scatterplot(
        data=risk_return_data,
        x='Total Risk (%)',
        y='Total Return (%)',
        s=200
    )
    
    # Add labels for each point
    for idx, row in risk_return_data.iterrows():
        plt.annotate(
            row['Portfolio'],
            (row['Total Risk (%)'], row['Total Return (%)'])
        )
    
    plt.title('Risk-Return Profile of Optimal Portfolios', pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 2. Expected vs Realized Idiosyncratic Risk
    for portfolio, comparison in comparison_results.items():
        if not comparison:
            continue
            
        # Prepare data for plotting
        symbols = list(comparison.keys())
        expected_risks = [comparison[s]['expected_idiosyncratic_risk'] * 100 for s in symbols]
        realized_risks = [comparison[s]['realized_idiosyncratic_risk'] * 100 for s in symbols]
        weights = [comparison[s]['weight'] * 100 for s in symbols]
        
        # Create DataFrame
        risk_comparison_df = pd.DataFrame({
            'Symbol': symbols,
            'Expected Risk (%)': expected_risks,
            'Realized Risk (%)': realized_risks,
            'Weight (%)': weights
        })
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        ax = sns.barplot(
            data=risk_comparison_df,
            x='Symbol',
            y='Expected Risk (%)',
            color='skyblue',
            alpha=0.7
        )
        
        # Add realized risk as scatter points
        sns.scatterplot(
            data=risk_comparison_df,
            x='Symbol',
            y='Realized Risk (%)',
            color='red',
            s=100
        )
        
        # Add weight as text
        for i, row in risk_comparison_df.iterrows():
            plt.text(
                i, 
                max(row['Expected Risk (%)'], row['Realized Risk (%)']) + 0.5, 
                f"{row['Weight (%)']:.1f}%", 
                ha='center', 
                va='bottom'
            )
        
        plt.title(f'Expected vs Realized Idiosyncratic Risk - Portfolio {portfolio}', pad=20, fontsize=14)
        plt.xlabel('Symbol', fontsize=12)
        plt.ylabel('Idiosyncratic Risk (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def print_results(attribution_results, comparison_results):
    """Print analysis results"""
    print("\nOptimal Portfolio Analysis and Risk Attribution Results:")
    print("-" * 80)
    
    for portfolio, results in attribution_results.items():
        print(f"\nPortfolio {portfolio}:")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Systematic Return Contribution: {results['systematic_return']*100:.2f}%")
        print(f"Idiosyncratic Return Contribution: {results['idiosyncratic_return']*100:.2f}%")
        print(f"Systematic Risk: {results['systematic_risk']*100:.2f}%")
        print(f"Idiosyncratic Risk: {results['idiosyncratic_risk']*100:.2f}%")
        print(f"Total Risk: {results['total_risk']*100:.2f}%")
        print("-" * 40)
    
    print("\nExpected vs Realized Idiosyncratic Risk Comparison:")
    print("-" * 80)
    
    for portfolio, comparison in comparison_results.items():
        print(f"\nPortfolio {portfolio}:")
        print(f"{'Symbol':<10} {'Weight (%)':<12} {'Expected Risk (%)':<18} {'Realized Risk (%)':<18} {'Difference (%)':<15}")
        print("-" * 80)
        
        for symbol, data in comparison.items():
            expected_risk = data['expected_idiosyncratic_risk'] * 100
            realized_risk = data['realized_idiosyncratic_risk'] * 100
            difference = realized_risk - expected_risk
            weight = data['weight'] * 100
            
            print(f"{symbol:<10} {weight:<12.2f} {expected_risk:<18.4f} {realized_risk:<18.4f} {difference:<15.4f}")
        
        print("-" * 80)

def main():
    # Read data
    portfolio_df = read_initial_portfolio()
    rf_df = read_risk_free_rate()
    prices_df = read_daily_prices()
    
    # Calculate returns
    returns_df = calculate_returns(prices_df)
    
    # Run CAPM analysis up to end of 2023
    capm_results, expected_market_return, expected_rf_rate = run_capm_analysis(returns_df, rf_df, portfolio_df, end_date='2023-12-31')
    
    print(f"\nExpected Market Return: {expected_market_return*100:.4f}%")
    print(f"Expected Risk-Free Rate: {expected_rf_rate*100:.4f}%")
    
    # Create optimal portfolios
    optimal_weights = create_optimal_portfolios(capm_results, portfolio_df)
    
    # Calculate portfolio values using optimal weights
    portfolio_values = calculate_portfolio_values(prices_df, portfolio_df, optimal_weights)
    
    # Calculate risk attribution for optimal portfolios
    attribution_results = calculate_risk_attribution(returns_df, capm_results, portfolio_df, portfolio_values, optimal_weights)
    
    # Compare expected vs realized idiosyncratic risk
    comparison_results = compare_expected_vs_realized_idiosyncratic_risk(capm_results, returns_df, optimal_weights)
    
    # Print results
    print_results(attribution_results, comparison_results)
    
    # Visualize results
    visualize_results(attribution_results, comparison_results)

if __name__ == "__main__":
    main()

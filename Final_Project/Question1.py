import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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

def calculate_portfolio_values(prices_df, portfolio_df):
    """Calculate daily portfolio values for each portfolio"""
    portfolio_values = {}
    prices_df_with_date = prices_df.reset_index()
    
    for portfolio in portfolio_df['Portfolio'].unique():
        portfolio_stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]
        daily_values = pd.DataFrame(index=prices_df.index)
        
        for _, row in portfolio_stocks.iterrows():
            symbol = row['Symbol']
            holding = row['Holding']
            if symbol in prices_df.columns:
                daily_values[symbol] = prices_df[symbol] * holding
        
        portfolio_values[portfolio] = daily_values.sum(axis=1)
    
    return portfolio_values

def calculate_portfolio_weights(prices_df, portfolio_df):
    """Calculate portfolio weights for each stock"""
    portfolio_weights = {}
    
    for portfolio in portfolio_df['Portfolio'].unique():
        portfolio_stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]
        weights = {}
        total_value = 0
        
        # First calculate total portfolio value
        for _, row in portfolio_stocks.iterrows():
            symbol = row['Symbol']
            holding = row['Holding']
            if symbol in prices_df.columns:
                price = prices_df[symbol].iloc[0]
                value = price * holding
                total_value += value
        
        # Then calculate weights
        if total_value > 0:  # Avoid division by zero
            for _, row in portfolio_stocks.iterrows():
                symbol = row['Symbol']
                holding = row['Holding']
                if symbol in prices_df.columns:
                    price = prices_df[symbol].iloc[0]
                    value = price * holding
                    weights[symbol] = value / total_value
        
        portfolio_weights[portfolio] = weights
    
    return portfolio_weights

def run_capm_analysis(returns_df, rf_df, portfolio_df):
    """Perform CAPM analysis for all stocks in portfolios"""
    # Prepare market returns and risk-free rate
    market_returns = returns_df['SPY']
    rf_df['Date'] = pd.to_datetime(rf_df['Date'])
    rf_df = rf_df.set_index('Date')
    rf_rates = rf_df['rf']
    
    # Align dates between returns and risk-free rates
    aligned_dates = returns_df.index.intersection(rf_rates.index)
    market_returns = market_returns[aligned_dates]
    rf_rates = rf_rates[aligned_dates]
    
    # Calculate excess returns
    excess_returns = returns_df.loc[aligned_dates].sub(rf_rates, axis=0)
    market_excess_returns = market_returns - rf_rates
    
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
                
                stock_results[symbol] = {
                    'beta': slope,
                    'alpha': intercept,
                    'r_squared': r_value**2,
                    'holding': row['Holding']
                }
        
        capm_results[portfolio] = stock_results
    
    return capm_results

def calculate_risk_attribution_option1(returns_df, capm_results, portfolio_df, portfolio_values):
    """
    Calculate risk attribution for each portfolio - Option 1
    Option 1: Risk-free rate is left in the systematic bucket
    """
    # Get market data
    market_returns = returns_df['SPY']
    market_var = np.var(market_returns)
    
    # Calculate market return for the period
    market_start = returns_df['SPY'].iloc[1]  # Use second day to avoid first day return of 0
    market_end = returns_df['SPY'].iloc[-1]
    market_period_return = (market_end - market_start) / market_start
    
    # Calculate portfolio weights
    first_prices = returns_df.iloc[1].to_frame().T
    portfolio_weights = calculate_portfolio_weights(first_prices, portfolio_df)
    
    attribution_results = {}
    for portfolio in capm_results.keys():
        weights = portfolio_weights[portfolio]
        systematic_risk = 0
        idiosyncratic_risk = 0
        systematic_return = 0
        idiosyncratic_return = 0
        
        # Calculate total portfolio return
        initial_value = portfolio_values[portfolio].iloc[1]  # Use second day
        final_value = portfolio_values[portfolio].iloc[-1]
        portfolio_return = (final_value - initial_value) / initial_value
        
        for symbol, results in capm_results[portfolio].items():
            if symbol in weights:
                weight = weights[symbol]
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
                
                # Option 1: Risk-free rate in systematic bucket
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

def calculate_risk_attribution_option2(returns_df, capm_results, portfolio_df, portfolio_values, rf_df):
    """
    Calculate risk attribution for each portfolio - Option 2
    Option 2: Risk-free rate is put in the idiosyncratic bucket
    """
    # Get market data
    market_returns = returns_df['SPY']
    market_var = np.var(market_returns)
    
    # Calculate market return for the period
    market_start = returns_df['SPY'].iloc[1]  # Use second day to avoid first day return of 0
    market_end = returns_df['SPY'].iloc[-1]
    market_period_return = (market_end - market_start) / market_start
    
    # Calculate average risk-free rate for the period
    rf_df['Date'] = pd.to_datetime(rf_df['Date'])
    rf_df = rf_df.set_index('Date')
    aligned_dates = returns_df.index.intersection(rf_df.index)
    avg_rf_rate = rf_df.loc[aligned_dates, 'rf'].mean()
    
    # Calculate market excess return
    market_excess_return = market_period_return - avg_rf_rate
    
    # Calculate portfolio weights
    first_prices = returns_df.iloc[1].to_frame().T
    portfolio_weights = calculate_portfolio_weights(first_prices, portfolio_df)
    
    attribution_results = {}
    for portfolio in capm_results.keys():
        weights = portfolio_weights[portfolio]
        systematic_risk = 0
        idiosyncratic_risk = 0
        systematic_return = 0
        idiosyncratic_return = 0
        
        # Calculate total portfolio return
        initial_value = portfolio_values[portfolio].iloc[1]  # Use second day
        final_value = portfolio_values[portfolio].iloc[-1]
        portfolio_return = (final_value - initial_value) / initial_value
        
        for symbol, results in capm_results[portfolio].items():
            if symbol in weights:
                weight = weights[symbol]
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
                
                # Option 2: Risk-free rate in idiosyncratic bucket
                stock_systematic_return = beta * market_excess_return
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

def calculate_risk_attribution_option3(returns_df, capm_results, portfolio_df, portfolio_values, rf_df):
    """
    Calculate risk attribution for each portfolio - Option 3
    Option 3: Separate risk-free rate component
    """
    # Get market data
    market_returns = returns_df['SPY']
    market_var = np.var(market_returns)
    
    # Calculate market return for the period
    market_start = returns_df['SPY'].iloc[1]  # Use second day to avoid first day return of 0
    market_end = returns_df['SPY'].iloc[-1]
    market_period_return = (market_end - market_start) / market_start
    
    # Calculate average risk-free rate for the period
    rf_df['Date'] = pd.to_datetime(rf_df['Date'])
    rf_df = rf_df.set_index('Date')
    aligned_dates = returns_df.index.intersection(rf_df.index)
    avg_rf_rate = rf_df.loc[aligned_dates, 'rf'].mean()
    
    # Calculate market excess return
    market_excess_return = market_period_return - avg_rf_rate
    
    # Calculate portfolio weights
    first_prices = returns_df.iloc[1].to_frame().T
    portfolio_weights = calculate_portfolio_weights(first_prices, portfolio_df)
    
    attribution_results = {}
    for portfolio in capm_results.keys():
        weights = portfolio_weights[portfolio]
        systematic_risk = 0
        idiosyncratic_risk = 0
        systematic_return = 0
        idiosyncratic_return = 0
        rf_return = 0  # New component for risk-free rate
        
        # Calculate total portfolio return
        initial_value = portfolio_values[portfolio].iloc[1]  # Use second day
        final_value = portfolio_values[portfolio].iloc[-1]
        portfolio_return = (final_value - initial_value) / initial_value
        
        for symbol, results in capm_results[portfolio].items():
            if symbol in weights:
                weight = weights[symbol]
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
                
                # Option 3: Separate risk-free rate component
                stock_systematic_return = beta * market_excess_return
                stock_idiosyncratic_return = stock_return - stock_systematic_return - avg_rf_rate
                stock_rf_return = avg_rf_rate
                
                systematic_return += weight * stock_systematic_return
                idiosyncratic_return += weight * stock_idiosyncratic_return
                rf_return += weight * stock_rf_return
        
        attribution_results[portfolio] = {
            'total_return': portfolio_return,
            'systematic_return': systematic_return,
            'idiosyncratic_return': idiosyncratic_return,
            'rf_return': rf_return,  # New field for risk-free attribution
            'systematic_risk': systematic_risk,
            'idiosyncratic_risk': idiosyncratic_risk,
            'total_risk': systematic_risk + idiosyncratic_risk
        }
    
    return attribution_results

def print_results(attribution_results):
    """Print analysis results"""
    print("\nCAPM Analysis and Risk Attribution Results:")
    print("-" * 80)
    
    # Check if Option 3 is being used (risk-free component exists)
    is_option3 = 'rf_return' in attribution_results[list(attribution_results.keys())[0]]
    
    for portfolio, results in attribution_results.items():
        print(f"\nPortfolio {portfolio}:")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        
        if is_option3:
            print(f"Risk-Free Rate Contribution: {results['rf_return']*100:.2f}%")
            
        print(f"Systematic Return Contribution: {results['systematic_return']*100:.2f}%")
        print(f"Idiosyncratic Return Contribution: {results['idiosyncratic_return']*100:.2f}%")
        print(f"Systematic Risk: {results['systematic_risk']*100:.2f}%")
        print(f"Idiosyncratic Risk: {results['idiosyncratic_risk']*100:.2f}%")
        print(f"Total Risk: {results['total_risk']*100:.2f}%")
        print("-" * 40)

def visualize_results(attribution_results):
    """Create visualizations for the analysis results using seaborn"""
    # Set the style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Prepare data for plotting
    portfolios = list(attribution_results.keys())
    
    # Check if Option 3 is being used (risk-free component exists)
    is_option3 = 'rf_return' in attribution_results[portfolios[0]]
    
    # Returns visualization data
    returns_data = {
        'Portfolio': [],
        'Return Type': [],
        'Value': []
    }
    
    for portfolio in portfolios:
        if is_option3:
            returns_data['Portfolio'].extend([portfolio] * 4)
            returns_data['Return Type'].extend(['Total', 'Risk-Free', 'Systematic', 'Idiosyncratic'])
            returns_data['Value'].extend([
                attribution_results[portfolio]['total_return'] * 100,
                attribution_results[portfolio]['rf_return'] * 100,
                attribution_results[portfolio]['systematic_return'] * 100,
                attribution_results[portfolio]['idiosyncratic_return'] * 100
            ])
        else:
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
    returns_plot.set_title('Portfolio Returns Attribution', pad=20, fontsize=14)
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
    risk_plot.set_title('Portfolio Risk Attribution', pad=20, fontsize=14)
    risk_plot.set_ylabel('Risk (%)', fontsize=12)
    risk_plot.set_xlabel('Portfolio', fontsize=12)
    
    # Add value labels on the bars
    for container in risk_plot.containers:
        risk_plot.bar_label(container, fmt='%.2f%%', padding=3)
    
    # Adjust legend
    plt.legend(title='Risk Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a title for the entire figure
    fig.suptitle('Portfolio Analysis Results', fontsize=16, y=1.02)
    
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
    
    plt.title('Risk-Return Profile of Portfolios', pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 2. Return Components Distribution
    if is_option3:
        return_components = pd.DataFrame({
            'Portfolio': portfolios,
            'Risk-Free': [attribution_results[p]['rf_return'] * 100 for p in portfolios],
            'Systematic': [attribution_results[p]['systematic_return'] * 100 for p in portfolios],
            'Idiosyncratic': [attribution_results[p]['idiosyncratic_return'] * 100 for p in portfolios]
        })
    else:
        return_components = pd.DataFrame({
            'Portfolio': portfolios,
            'Systematic': [attribution_results[p]['systematic_return'] * 100 for p in portfolios],
            'Idiosyncratic': [attribution_results[p]['idiosyncratic_return'] * 100 for p in portfolios]
        })
    
    return_components_melted = pd.melt(
        return_components,
        id_vars=['Portfolio'],
        var_name='Component',
        value_name='Return (%)'
    )
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=return_components_melted,
        x='Portfolio',
        y='Return (%)',
        hue='Component',
        split=True,
        inner='box',
        palette='Set2'
    )
    plt.title('Distribution of Return Components', pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the analysis"""
    print("Portfolio Analysis with CAPM Risk Attribution")
    print("=" * 50)
    
    # Read data
    portfolio_df = read_initial_portfolio()
    rf_df = read_risk_free_rate()
    prices_df = read_daily_prices()
    
    # Calculate returns
    returns_df = calculate_returns(prices_df)
    
    # Calculate portfolio values
    portfolio_values = calculate_portfolio_values(prices_df, portfolio_df)
    
    # Perform CAPM analysis
    capm_results = run_capm_analysis(returns_df, rf_df, portfolio_df)
    
    # Choose which risk attribution option to use
    # Uncomment the option you want to use
    
    # Option 1: Risk-free rate in systematic bucket (original implementation)
    print("\nUsing Option 1: Risk-free rate in systematic bucket")
    attribution_results = calculate_risk_attribution_option1(returns_df, capm_results, portfolio_df, portfolio_values)
    
    # Option 2: Risk-free rate in idiosyncratic bucket
    print("\nUsing Option 2: Risk-free rate in idiosyncratic bucket")
    attribution_results = calculate_risk_attribution_option2(returns_df, capm_results, portfolio_df, portfolio_values, rf_df)
    
    # Option 3: Separate risk-free rate component
    print("\nUsing Option 3: Separate risk-free rate component")
    attribution_results = calculate_risk_attribution_option3(returns_df, capm_results, portfolio_df, portfolio_values, rf_df)
    
    # Print results
    print_results(attribution_results)
    
    # Visualize results
    visualize_results(attribution_results)

if __name__ == "__main__":
    main()
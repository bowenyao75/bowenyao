import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


# Black-Scholes-Merton call option pricing function
def bsm_call(S, K, r, T, sigma):
    """
    Calculate the price of a European call option using Black-Scholes-Merton formula.
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free interest rate (annualized)
    T : float
        Time to expiration (in years)
    sigma : float
        Volatility of the underlying asset (annualized)
    Returns:
    --------
    float
        Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call


# Black-Scholes-Merton put option pricing function
def bsm_put(S, K, r, T, sigma):
    """
    Calculate the price of a European put option using Black-Scholes-Merton formula.
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free interest rate (annualized)
    T : float
        Time to expiration (in years)
    sigma : float
        Volatility of the underlying asset (annualized)
    Returns:
    --------
    float
        Put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put


# Calculate option Greeks
def calculate_greeks(S, K, r, T, sigma):
    """
    Calculate the Greeks (Delta, Vega, Theta) for a European call option.
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free interest rate (annualized)
    T : float
        Time to expiration (in years)
    sigma : float
        Volatility of the underlying asset (annualized)
    Returns:
    --------
    dict
        Dictionary containing Delta, Vega, and Theta
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Delta - sensitivity to changes in the underlying asset price
    delta = norm.cdf(d1)
    # Vega - sensitivity to changes in volatility (scaled for 1% change)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    # Theta - sensitivity to the passage of time (daily)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
             r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    return {'delta': delta, 'vega': vega, 'theta': theta}


# Find the implied volatility using Newton-Raphson method
def find_implied_volatility(S, K, r, T, market_price, initial_guess=0.2, 
                           tolerance=1e-8, max_iterations=100):
    """
    Find the implied volatility using Newton-Raphson method.
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free interest rate (annualized)
    T : float
        Time to expiration (in years)
    market_price : float
        Market price of the option
    initial_guess : float, optional
        Initial guess for volatility, by default 0.2
    tolerance : float, optional
        Convergence tolerance, by default 1e-8
    max_iterations : int, optional
        Maximum number of iterations, by default 100
    Returns:
    --------
    float
        Implied volatility
    """
    sigma = initial_guess
    iteration = 0
    while iteration < max_iterations:
        # Calculate option price with current sigma
        option_price = bsm_call(S, K, r, T, sigma)
        price_diff = option_price - market_price
        # Check if we've converged
        if abs(price_diff) < tolerance:
            break
        # Calculate vega (sensitivity to volatility)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        # Update sigma using Newton-Raphson
        sigma = sigma - price_diff / vega
        # Ensure sigma stays positive
        if sigma <= 0:
            sigma = 0.001
        iteration += 1
    if iteration == max_iterations:
        print("Warning: Maximum iterations reached, convergence may not be achieved")
    return sigma


def main():
    # Given parameters
    S = 31      # Stock price
    K = 30      # Strike price
    r = 0.10    # Risk-free rate
    T = 0.25    # Time to maturity (3 months)
    C = 3.00    # Call option price
    # A. Calculate implied volatility
    implied_vol = find_implied_volatility(S, K, r, T, C)
    print(f"A. Implied Volatility: {implied_vol * 100:.2f}%")
    # B. Calculate Greeks
    greeks = calculate_greeks(S, K, r, T, implied_vol)
    print("\nB. Option Greeks:")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Vega (per 1% change in volatility): {greeks['vega']:.4f}")
    print(f"Theta (daily): {greeks['theta']:.4f}")
    # Calculate price change for a 1% increase in volatility
    volatility_increase = implied_vol + 0.01
    price_after_increase = bsm_call(S, K, r, T, volatility_increase)
    price_change = price_after_increase - C
    print("\nPrice change for a 1% increase in volatility:")
    print(f"New option price: {price_after_increase:.4f}")
    print(f"Actual price change: {price_change:.4f}")
    print(f"Expected change based on Vega: {greeks['vega']:.4f}")
    # Create a chart showing the impact of volatility on option price
    volatilities = np.linspace(implied_vol - 0.05, implied_vol + 0.05, 100)
    prices = [bsm_call(S, K, r, T, sigma) for sigma in volatilities]
    plt.figure(figsize=(10, 6))
    plt.plot(volatilities * 100, prices, 'b-', linewidth=2)
    plt.plot([implied_vol * 100], [C], 'ro', markersize=8)
    plt.xlabel('Volatility (%)')
    plt.ylabel('Option Price ($)')
    plt.title('Impact of Volatility on Call Option Price')
    plt.grid(True)
    plt.axvline(x=implied_vol * 100, color='gray', linestyle='--')
    plt.axhline(y=C, color='gray', linestyle='--')
    plt.savefig('volatility_impact.png', dpi=300, bbox_inches='tight')
    # C. Calculate put price using BSM and check Put-Call Parity
    put_price = bsm_put(S, K, r, T, implied_vol)
    parity_value = S - K * np.exp(-r * T)
    parity_check = C - put_price
    print("\nC. Put Price and Put-Call Parity:")
    print(f"Put Price: {put_price:.4f}")
    print(f"Put-Call Parity LHS (C - P): {parity_check:.4f}")
    print(f"Put-Call Parity RHS (S - K*e^(-rT)): {parity_value:.4f}")
    print(f"Difference: {parity_check - parity_value:.8f}")
    print(f"Put-Call Parity Holds: {abs(parity_check - parity_value) < 1e-6}")
    # Create visualizations for Greeks
    # Delta vs. Stock Price
    stock_prices = np.linspace(K * 0.8, K * 1.2, 100)
    deltas = [calculate_greeks(s, K, r, T, implied_vol)['delta'] for s in stock_prices]
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, deltas, 'g-', linewidth=2)
    plt.plot([S], [greeks['delta']], 'ro', markersize=8)
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Delta')
    plt.title('Delta vs. Stock Price')
    plt.grid(True)
    plt.axvline(x=S, color='gray', linestyle='--')
    plt.savefig('delta_curve.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()


def bsm_call(S, K, r, T, sigma):
    """
    Calculate Black-Scholes-Merton call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)    
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call


def bsm_put(S, K, r, T, sigma):
    """
    Calculate Black-Scholes-Merton put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put


def calculate_greeks(S, K, r, T, sigma, option_type='call'):
    """
    Calculate option Greeks
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    results = {}
    # Delta
    if option_type == 'call':
        results['delta'] = norm.cdf(d1)
    else:  # put
        results['delta'] = norm.cdf(d1) - 1
    # Gamma (same for call and put)
    results['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    # Vega (same for call and put, but per 1% change)
    results['vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100
    # Theta (annualized, then converted to daily)
    if option_type == 'call':
        results['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                           r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:  # put
        results['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                           r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return results


def main():
    # Given parameters
    S0 = 31        # Current stock price
    K = 30         # Strike price
    r = 0.10       # Risk-free rate
    T_original = 0.25  # Original time to maturity (3 months)
    implied_vol = 0.3351  # Implied volatility from previous calculation
    # New parameters
    annual_vol = 0.25   # Annual stock volatility
    annual_return = 0   # Expected annual return
    trading_days = 255  # Trading days in a year
    holding_period = 20  # Trading days holding period
    alpha = 0.05        # Confidence level for VaR/ES
    # Calculate time decay for the holding period (in years)
    time_decay = holding_period / trading_days
    T_new = T_original - time_decay
    # Current option values
    current_call = bsm_call(S0, K, r, T_original, implied_vol)
    current_put = bsm_put(S0, K, r, T_original, implied_vol)
    # Current portfolio value
    current_portfolio = current_call + current_put + S0
    print("Initial Values:")
    print(f"Stock Price: ${S0:.2f}")
    print(f"Call Option: ${current_call:.4f}")
    print(f"Put Option: ${current_put:.4f}")
    print(f"Portfolio Value: ${current_portfolio:.4f}")
    # Calculate Greeks
    call_greeks = calculate_greeks(S0, K, r, T_original, implied_vol, 'call')
    put_greeks = calculate_greeks(S0, K, r, T_original, implied_vol, 'put')
    # Portfolio Greeks
    portfolio_delta = call_greeks['delta'] + put_greeks['delta'] + 1  # +1 for stock
    portfolio_gamma = call_greeks['gamma'] + put_greeks['gamma']
    portfolio_theta = call_greeks['theta'] + put_greeks['theta']
    print("\nGreeks:")
    print(f"Call Delta: {call_greeks['delta']:.4f}")
    print(f"Put Delta: {put_greeks['delta']:.4f}")
    print(f"Portfolio Delta: {portfolio_delta:.4f}")
    print(f"Portfolio Gamma: {portfolio_gamma:.6f}")
    print(f"Portfolio Theta (daily): ${portfolio_theta:.4f}")
    # D. Delta Normal Approximation
    # Convert annual volatility to the holding period
    holding_period_vol = annual_vol * np.sqrt(holding_period / trading_days)
    # Delta-Normal VaR calculation
    z_score_5pct = norm.ppf(alpha)  # Z-score for 5% quantile
    portfolio_value_std = abs(portfolio_delta) * S0 * holding_period_vol
    # Add in the time decay effect
    time_decay_effect = portfolio_theta * holding_period
    # VaR calculation with time decay
    delta_normal_var = -(z_score_5pct * portfolio_value_std + time_decay_effect)
    # ES calculation for delta-normal method
    # For normal distribution, ES = mean - std * pdf(z_alpha) / alpha
    es_factor = norm.pdf(z_score_5pct) / alpha
    delta_normal_es = -(portfolio_value_std * es_factor + time_decay_effect)
    print("\nD. Delta Normal Approximation:")
    print(f"Holding Period Volatility: {holding_period_vol*100:.2f}%")
    print(f"Portfolio Value Std Dev: ${portfolio_value_std:.4f}")
    print(f"Time Decay Effect: ${time_decay_effect:.4f}")
    print(f"VaR (5%): ${delta_normal_var:.4f}")
    print(f"ES (5%): ${delta_normal_es:.4f}")
    # E. Monte Carlo Simulation
    np.random.seed(42)  # For reproducibility
    num_simulations = 10000
    # Generate random stock returns
    stock_returns = np.random.normal(
        annual_return * (holding_period/trading_days),
        annual_vol * np.sqrt(holding_period/trading_days),
        num_simulations
    )
    # Calculate simulated stock prices
    stock_prices_sim = S0 * np.exp(stock_returns)
    # Calculate portfolio values
    portfolio_values = np.zeros(num_simulations)
    for i in range(num_simulations):
        S_sim = stock_prices_sim[i]
        call_value = bsm_call(S_sim, K, r, T_new, implied_vol)
        put_value = bsm_put(S_sim, K, r, T_new, implied_vol)
        portfolio_values[i] = call_value + put_value + S_sim
    # Calculate portfolio P&L
    portfolio_pnl = portfolio_values - current_portfolio
    # Calculate VaR and ES
    sorted_pnl = np.sort(portfolio_pnl)
    var_index = int(alpha * num_simulations)
    mc_var = -sorted_pnl[var_index]
    mc_es = -np.mean(sorted_pnl[:var_index+1])
    print("\nE. Monte Carlo Simulation:")
    print(f"Number of Simulations: {num_simulations}")
    print(f"VaR (5%): ${mc_var:.4f}")
    print(f"ES (5%): ${mc_es:.4f}")
    # Comparison
    print("\nComparison between methods:")
    print(f"Delta-Normal VaR: ${delta_normal_var:.4f}")
    print(f"Monte Carlo VaR: ${mc_var:.4f}")
    print(f"Difference: ${abs(delta_normal_var - mc_var):.4f}")
    print(f"Delta-Normal ES: ${delta_normal_es:.4f}")
    print(f"Monte Carlo ES: ${mc_es:.4f}")
    print(f"Difference: ${abs(delta_normal_es - mc_es):.4f}")
    # Generate portfolio value curve
    stock_range = np.linspace(20, 42, 100)
    portfolio_values_curve = np.zeros_like(stock_range)
    delta_approx_values = np.zeros_like(stock_range)
    for i, S in enumerate(stock_range):
        # Exact portfolio value
        call_value = bsm_call(S, K, r, T_new, implied_vol)
        put_value = bsm_put(S, K, r, T_new, implied_vol)
        portfolio_values_curve[i] = call_value + put_value + S
        # Delta approximation
        delta_approx_values[i] = current_portfolio + portfolio_delta * (S - S0)
    # Create visualization of portfolio value vs stock price
    plt.figure(figsize=(12, 8))
    plt.plot(stock_range, portfolio_values_curve, 'b-', linewidth=2, label='Actual Portfolio Value')
    plt.plot(stock_range, delta_approx_values, 'r--', linewidth=2, label='Delta Approximation')
    plt.axvline(x=S0, color='g', linestyle='--', label=f'Current Stock Price (${S0})')
    # Mark the VaR points
    var_stock_price_dn = S0 * np.exp(z_score_5pct * holding_period_vol)
    plt.scatter([var_stock_price_dn], 
                [current_portfolio - delta_normal_var], 
                color='orange', s=100, marker='o', label='Delta-Normal VaR Point')
    # Add percentiles from Monte Carlo
    percentile_5th = np.percentile(stock_prices_sim, 5)
    plt.axvline(x=percentile_5th, color='purple', linestyle=':', 
                label=f'5th Percentile Stock Price (${percentile_5th:.2f})')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value vs. Stock Price')
    plt.grid(True)
    plt.legend()
    plt.savefig('portfolio_value_curve.png', dpi=300, bbox_inches='tight')
    # Create histogram of Monte Carlo results
    plt.figure(figsize=(12, 8))
    plt.hist(portfolio_pnl, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=-mc_var, color='r', linestyle='--', 
                label=f'Monte Carlo VaR (${mc_var:.2f})')
    plt.axvline(x=-delta_normal_var, color='g', linestyle='--', 
                label=f'Delta-Normal VaR (${delta_normal_var:.2f})')
    plt.axvline(x=-mc_es, color='purple', linestyle=':', 
                label=f'Monte Carlo ES (${mc_es:.2f})')
    plt.axvline(x=-delta_normal_es, color='orange', linestyle=':', 
                label=f'Delta-Normal ES (${delta_normal_es:.2f})')
    plt.xlabel('Portfolio Profit/Loss ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Portfolio P&L (Monte Carlo Simulation)')
    plt.grid(True)
    plt.legend()
    plt.savefig('portfolio_pnl_distribution.png', dpi=300, bbox_inches='tight')
    # Create another visualization showing the delta effect vs the true non-linear relationship
    fig, ax1 = plt.subplots(figsize=(12, 8))
    # True portfolio value change
    true_change = portfolio_values_curve - current_portfolio
    ax1.plot(stock_range, true_change, 'b-', linewidth=2, label='Actual Value Change')
    # Delta approximation
    delta_change = portfolio_delta * (stock_range - S0)
    ax1.plot(stock_range, delta_change, 'r--', linewidth=2, label='Delta Approximation')
    # Delta + Gamma approximation (second-order)
    gamma_change = delta_change + 0.5 * portfolio_gamma * (stock_range - S0)**2
    ax1.plot(stock_range, gamma_change, 'g-.', linewidth=2, label='Delta-Gamma Approximation')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=S0, color='k', linestyle='--', alpha=0.3, label=f'Current Stock Price (${S0})')
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Portfolio Value Change ($)')
    ax1.set_title('Non-Linearity of Portfolio Value Change')
    ax1.grid(True)
    ax1.legend()
    # Second axis showing approximation error
    ax2 = ax1.twinx()
    ax2.plot(stock_range, true_change - delta_change, 'k:', label='Delta Approximation Error')
    ax2.set_ylabel('Approximation Error ($)')
    plt.savefig('delta_approximation_error.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
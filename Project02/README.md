# Quantitative Risk Management - Project 2

## Overview
This repository contains three Python scripts that analyze financial risk metrics, portfolio value, and option pricing for the Quantitative Risk Management assignment. The project involves calculating stock returns, portfolio risk measures, and options pricing using various risk assessment methods.

## Files
### 1. `question1.py` - Stock Returns Analysis
This script focuses on analyzing **demeaned arithmetic and logarithmic returns** for three stocks: **SPY (S&P 500 ETF), AAPL (Apple Inc.), and EQIX (Equinix Inc.)**. The script performs the following:
- Computes **arithmetic and log returns** using historical price data.
- Demeans the returns by subtracting their means.
- Calculates **standard deviations** to assess volatility.
- Compares the arithmetic and log return statistics to determine if their volatility measures are similar.

### 2. `question2.py` - Portfolio Value and Risk Assessment
This script calculates **portfolio value and risk measures** for a portfolio consisting of:
- **100 shares of SPY**
- **200 shares of AAPL**
- **150 shares of EQIX**

The script performs:
- **Portfolio valuation** based on stock prices on January 3, 2025.
- **Portfolio weight calculation** to determine the influence of each stock.
- **Value at Risk (VaR) and Expected Shortfall (ES)** calculations using:
  - **Normal Distribution with EWMA (λ=0.97)**
  - **T-Distribution using a Gaussian Copula**
  - **Historical Simulation**
- **Comparison of VaR and ES estimates** across different methodologies.
- **Analysis of diversification benefits**, showing how risk is reduced when stocks are combined in a portfolio.

### 3. `question3.py` - European Option Pricing and Portfolio Risk  
This script evaluates a **European call and put option** for a stock with the following characteristics:
- **Stock Price:** $31
- **Strike Price:** $30
- **Time to Maturity:** 3 months (0.25 years)
- **Risk-Free Rate:** 10%
- **Market Price of Call Option:** $3.00

The script includes:
- **Implied Volatility Calculation** using the Newton-Raphson method with the **Black-Scholes-Merton model**.
- **Computation of Option Greeks** (Delta, Vega, Theta) to assess price sensitivity.
- **Verification of Put-Call Parity** to ensure arbitrage-free pricing.
- **Portfolio Risk Calculation** for a portfolio including:
  - 1 Call Option
  - 1 Put Option
  - 1 Stock Share
- **VaR and ES Calculations** using:
  - **Delta Normal Approximation**
  - **Monte Carlo Simulation**
- **Comparison of Risk Methods**, highlighting how Monte Carlo better captures the non-linear payoff of options compared to the Delta Normal approximation.

## Summary
Each script contributes to understanding different aspects of risk management:
- **`question1.py`**: Stock return analysis.
- **`question2.py`**: Portfolio valuation and risk measurement.
- **`question3.py`**: Options pricing and risk assessment.

These scripts provide key insights into **financial risk, volatility, and portfolio diversification** using theoretical models and empirical data.

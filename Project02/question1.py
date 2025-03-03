import pandas as pd
import numpy as np


def calculate_returns(file_path):
    # Read the CSV file 
    df = pd.read_csv(file_path)
    # Convert the Date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    # Select only the columns we need (Date, SPY, AAPL, EQIX)
    selected_df = df[['Date', 'SPY', 'AAPL', 'EQIX']]
    # Set Date as index for easier manipulation
    selected_df.set_index('Date', inplace=True)
    # A. Calculate Arithmetic Returns
    arithmetic_returns = selected_df.pct_change()
    # Drop the first row which will be NaN after the pct_change operation
    arithmetic_returns = arithmetic_returns.dropna()
    # Calculate the mean of each column
    arithmetic_means = arithmetic_returns.mean()
    # Remove the mean from each series
    demeaned_arithmetic_returns = arithmetic_returns.subtract(arithmetic_means)
    # Calculate standard deviation for each stock
    arithmetic_std_devs = demeaned_arithmetic_returns.std()
    # B. Calculate Log Returns
    log_returns = np.log(selected_df / selected_df.shift(1))
    # Drop the first row which will be NaN
    log_returns = log_returns.dropna()
    # Calculate the mean of each column
    log_means = log_returns.mean()
    # Remove the mean from each series
    demeaned_log_returns = log_returns.subtract(log_means)
    # Calculate standard deviation for each stock
    log_std_devs = demeaned_log_returns.std()
    # Reset index to include Date as a column again
    demeaned_arithmetic_returns = demeaned_arithmetic_returns.reset_index()
    demeaned_log_returns = demeaned_log_returns.reset_index()
    return (demeaned_arithmetic_returns, arithmetic_std_devs, 
            demeaned_log_returns, log_std_devs)


def main():
    """
    Main function to execute the analysis and display the results.
    """
    file_path = 'DailyPrices.csv'
    try:
        # Calculate returns
        (demeaned_arithmetic_returns, arithmetic_std_devs, 
         demeaned_log_returns, log_std_devs) = calculate_returns(file_path)
        # Display the results for arithmetic returns
        print("\nA. Arithmetic Returns (Demeaned)")
        print("Last 5 rows:")
        print(demeaned_arithmetic_returns.tail().to_string(index=False))
        print("\nStandard Deviations:")
        for stock, std_dev in arithmetic_std_devs.items():
            print(f"{stock}: {std_dev:.8f}")
        # Display the results for log returns
        print("\nB. Log Returns (Demeaned)")
        print("Last 5 rows:")
        print(demeaned_log_returns.tail().to_string(index=False))
        print("\nStandard Deviations:")
        for stock, std_dev in log_std_devs.items():
            print(f"{stock}: {std_dev:.8f}") 
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

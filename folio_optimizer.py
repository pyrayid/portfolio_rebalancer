import yfinance as yf
import pandas as pd
from datetime import datetime
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import HRPOpt


def calculate_portfolio_metrics(portfolio_values, risk_free_rate):
    """
    Calculates performance metrics for a given portfolio.

    Args:
        portfolio_values (pd.Series): A time series of portfolio values.
        risk_free_rate (float): The risk-free rate.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    monthly_returns = portfolio_values.pct_change().dropna()
    volatility = monthly_returns.std() * (12**0.5) if not monthly_returns.empty else 0.0
    annualized_avg_return = monthly_returns.mean() * 12
    sharpe_ratio = (
        (annualized_avg_return - risk_free_rate) / volatility
        if volatility != 0
        else 0.0
    )

    downside_returns = monthly_returns[monthly_returns < 0]
    downside_deviation = (
        downside_returns.std() * (12**0.5) if not downside_returns.empty else 0.0
    )
    sortino_ratio = (
        (annualized_avg_return - risk_free_rate) / downside_deviation
        if downside_deviation != 0
        else 0.0
    )

    peak = portfolio_values.expanding(min_periods=1).max()
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    return {
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": max_drawdown,
        "Annualized Return": annualized_avg_return,
    }


def find_best_stock_portfolio(
    stock_list,
    initial_investment=10000,
    months=60,
    risk_free_rate=0.04,  # Example: 4% annual risk-free rate
    min_weight_per_stock=0.0,  # New parameter for minimum weight per stock
    max_weight_per_stock=1.0,  # New parameter for maximum weight per stock
):
    """
    Computes the monthly price for the past 'months' for each stock in 'stock_list',
    and then finds the set of stocks that would have given the best performance
    if 'initial_investment' was equally invested in all of them.

    Args:
        stock_list (list): A list of stock tickers (e.g., ['AAPL', 'MSFT']).
        initial_investment (float): The initial amount to invest.
        months (int): The number of past months to consider for performance calculation.

    """
    try:
        # Download historical data for all stocks in one call
        period = f"{months // 12}y" if months % 12 == 0 else f"{months}mo"
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(months=months)

        data = yf.download(
            stock_list,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            print("Could not retrieve data for any of the stocks.")
            return [], 0.0, 0.0

        # Use 'Adj Close' prices for historical analysis
        if "Adj Close" in data.columns:
            monthly_prices = data["Adj Close"].resample("ME").last()
        else:
            print("No 'Adj Close' data found for any of the stocks.")
            return [], 0.0, 0.0

        # Drop rows with any missing data
        monthly_prices = monthly_prices.dropna()

        if monthly_prices.empty:
            print("No valid stock data after cleaning.")
            return [], 0.0, 0.0

        # Calculate monthly returns
        returns = monthly_prices.pct_change().dropna()

        if returns.empty or len(returns.columns) < 2:
            print(
                "Not enough data to calculate returns for optimization (need at least 2 stocks)."
            )
            return [], 0.0, 0.0

        combined_prices = monthly_prices

    except Exception as e:
        print(f"Error downloading or processing data: {e}")
        return [], 0.0, 0.0

    # Estimate expected returns and covariance matrix
    mu = mean_historical_return(combined_prices)
    S = CovarianceShrinkage(combined_prices).ledoit_wolf()

    # Ensure mu and S have aligned indices/columns
    # This is usually handled by PyPortfolioOpt if inputs are from the same DataFrame,
    # but explicit check for robustness.
    common_tickers_for_pypfopt = list(set(mu.index) & set(S.columns))
    if not common_tickers_for_pypfopt or len(common_tickers_for_pypfopt) < 2:
        print(
            "Not enough common tickers found between expected returns and covariance matrix for optimization (need at least 2)."
        )
        return [], 0.0, 0.0

    mu = mu.loc[common_tickers_for_pypfopt]
    S = S.loc[common_tickers_for_pypfopt, common_tickers_for_pypfopt]

    # Initialize EfficientFrontier

    # Define optimization objectives
    optimized_portfolios = []

    # 1. Hierarchical Risk Parity Portfolio
    try:
        hrp = HRPOpt(returns, S)
        hrp.optimize()
        hrp_weights = hrp.clean_weights()

        # Calculate performance for this portfolio using matrix operations
        weights_hrp = pd.Series(hrp_weights)
        available_tickers = combined_prices.columns.intersection(weights_hrp.index)
        if not available_tickers.empty:
            initial_prices = combined_prices[available_tickers].iloc[0]
            shares = (
                initial_investment * weights_hrp[available_tickers] / initial_prices
            )
            portfolio_values_over_time_hrp = (
                combined_prices[available_tickers] * shares
            ).sum(axis=1)
        else:
            portfolio_values_over_time_hrp = pd.Series(0.0, index=combined_prices.index)

        portfolio_values_over_time_hrp = portfolio_values_over_time_hrp[
            portfolio_values_over_time_hrp > 0
        ]

        if not portfolio_values_over_time_hrp.empty:
            final_value_hrp = portfolio_values_over_time_hrp.iloc[-1].item()
            peak_hrp = portfolio_values_over_time_hrp.expanding(min_periods=1).max()
            drawdown_hrp = (portfolio_values_over_time_hrp - peak_hrp) / peak_hrp
            max_drawdown_hrp = drawdown_hrp.min() if not drawdown_hrp.empty else 0.0

            # Calculate portfolio metrics
            metrics_hrp = calculate_portfolio_metrics(
                portfolio_values_over_time_hrp, risk_free_rate
            )

            optimized_portfolios.append(
                {
                    "Portfolio Type": "Hierarchical Risk Parity",
                    "Stocks": [
                        {
                            ticker: round(weight * 100, 2)
                            for ticker, weight in hrp_weights.items()
                            if weight > 0
                        }
                    ],
                    "Final Value": final_value_hrp,
                    "Maximum Drawdown": metrics_hrp["Maximum Drawdown"],
                    "Volatility": metrics_hrp["Volatility"],
                    "Sharpe Ratio": metrics_hrp["Sharpe Ratio"],
                    "Sortino Ratio": metrics_hrp["Sortino Ratio"],
                    "Annualized Return": metrics_hrp["Annualized Return"] if "Annualized Return" in metrics_hrp else 0.0,
                }
            )
        else:
            print("HRP portfolio has no valid values over time.")
    except Exception as e:
        print(f"Error optimizing HRP portfolio: {e}")

    # 2. Maximize Sharpe Ratio Portfolio
    try:
        ef_sharpe = EfficientFrontier(
            mu, S, weight_bounds=(min_weight_per_stock, max_weight_per_stock)
        )
        raw_weights_sharpe = ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights_sharpe = ef_sharpe.clean_weights()

        # Calculate performance for this portfolio using matrix operations
        weights_sharpe = pd.Series(cleaned_weights_sharpe)
        available_tickers = combined_prices.columns.intersection(weights_sharpe.index)
        if not available_tickers.empty:
            initial_prices = combined_prices[available_tickers].iloc[0]
            shares = (
                initial_investment * weights_sharpe[available_tickers] / initial_prices
            )
            portfolio_values_over_time_sharpe = (
                combined_prices[available_tickers] * shares
            ).sum(axis=1)
        else:
            portfolio_values_over_time_sharpe = pd.Series(
                0.0, index=combined_prices.index
            )

        portfolio_values_over_time_sharpe = portfolio_values_over_time_sharpe[
            portfolio_values_over_time_sharpe > 0
        ]

        if not portfolio_values_over_time_sharpe.empty:
            final_value_sharpe = portfolio_values_over_time_sharpe.iloc[-1].item()
            peak_sharpe = portfolio_values_over_time_sharpe.expanding(
                min_periods=1
            ).max()
            drawdown_sharpe = (
                portfolio_values_over_time_sharpe - peak_sharpe
            ) / peak_sharpe
            max_drawdown_sharpe = (
                drawdown_sharpe.min() if not drawdown_sharpe.empty else 0.0
            )

            # Calculate portfolio metrics
            metrics_sharpe = calculate_portfolio_metrics(
                portfolio_values_over_time_sharpe, risk_free_rate
            )

            optimized_portfolios.append(
                {
                    "Portfolio Type": "Max Sharpe Ratio",
                    "Stocks": [
                        {
                            ticker: round(weight * 100, 2)
                            for ticker, weight in cleaned_weights_sharpe.items()
                            if weight > 0
                        }
                    ],
                    "Final Value": final_value_sharpe,
                    "Maximum Drawdown": metrics_sharpe["Maximum Drawdown"],
                    "Volatility": metrics_sharpe["Volatility"],
                    "Sharpe Ratio": metrics_sharpe["Sharpe Ratio"],
                    "Sortino Ratio": metrics_sharpe["Sortino Ratio"],
                    "Annualized Return": metrics_sharpe["Annualized Return"] if "Annualized Return" in metrics_sharpe else 0.0,
                }
            )
        else:
            print("Max Sharpe Ratio portfolio has no valid values over time.")
    except Exception as e:
        print(f"Error optimizing Max Sharpe Ratio portfolio: {e}")

    # 3. Minimum Volatility Portfolio
    try:
        ef_min_vol = EfficientFrontier(
            mu, S, weight_bounds=(min_weight_per_stock, max_weight_per_stock)
        )
        raw_weights_min_vol = ef_min_vol.min_volatility()
        cleaned_weights_min_vol = ef_min_vol.clean_weights()

        # Calculate performance for this portfolio using matrix operations
        weights_min_vol = pd.Series(cleaned_weights_min_vol)
        available_tickers = combined_prices.columns.intersection(weights_min_vol.index)
        if not available_tickers.empty:
            initial_prices = combined_prices[available_tickers].iloc[0]
            shares = (
                initial_investment * weights_min_vol[available_tickers] / initial_prices
            )
            portfolio_values_over_time_min_vol = (
                combined_prices[available_tickers] * shares
            ).sum(axis=1)
        else:
            portfolio_values_over_time_min_vol = pd.Series(
                0.0, index=combined_prices.index
            )

        portfolio_values_over_time_min_vol = portfolio_values_over_time_min_vol[
            portfolio_values_over_time_min_vol > 0
        ]

        if not portfolio_values_over_time_min_vol.empty:
            final_value_min_vol = portfolio_values_over_time_min_vol.iloc[-1].item()
            peak_min_vol = portfolio_values_over_time_min_vol.expanding(
                min_periods=1
            ).max()
            drawdown_min_vol = (
                portfolio_values_over_time_min_vol - peak_min_vol
            ) / peak_min_vol
            max_drawdown_min_vol = (
                drawdown_min_vol.min() if not drawdown_min_vol.empty else 0.0
            )

            # Calculate portfolio metrics
            metrics_min_vol = calculate_portfolio_metrics(
                portfolio_values_over_time_min_vol, risk_free_rate
            )

            optimized_portfolios.append(
                {
                    "Portfolio Type": "Min Volatility",
                    "Stocks": [
                        {
                            ticker: round(weight * 100, 2)
                            for ticker, weight in cleaned_weights_min_vol.items()
                            if weight > 0
                        }
                    ],
                    "Final Value": final_value_min_vol,
                    "Maximum Drawdown": metrics_min_vol["Maximum Drawdown"],
                    "Volatility": metrics_min_vol["Volatility"],
                    "Sharpe Ratio": metrics_min_vol["Sharpe Ratio"],
                    "Sortino Ratio": metrics_min_vol["Sortino Ratio"],
                    "Annualized Return": metrics_min_vol["Annualized Return"] if "Annualized Return" in metrics_min_vol else 0.0,
                }
            )
        else:
            print("Min Volatility portfolio has no valid values over time.")
    except Exception as e:
        print(f"Error optimizing Min Volatility portfolio: {e}")

    # After all optimizations, convert to DataFrame and print
    if optimized_portfolios:
        results_df = pd.DataFrame(optimized_portfolios)
        results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False)
        # print("\n--- Optimized Portfolio Performance Metrics ---")
        # print(
        #     results_df.to_string(
        #         index=False,
        #         formatters={
        #             "Maximum Drawdown": "{:.2%}".format,
        #             "Volatility": "{:.2%}".format,
        #             "Sharpe Ratio": "{:.2f}".format,
        #             "Sortino Ratio": "{:.2f}".format,
        #             "Annualized Return": "{:.2%}".format,
        #         },
        #     )
        # )

        # Save the table to a CSV file
        today = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = f"optimized_portfolio_metrics_{today}.csv"
        results_df.to_csv(output_filename, index=False)
        print(f"\nOptimized portfolio performance metrics saved to {output_filename}")
    else:
        print("No valid optimized portfolios found.")
        return pd.DataFrame()

    # Return the results DataFrame
    return results_df

if __name__ == "__main__":
    # Example Usage:
    # You can replace this with your actual stock list
    stocks_list = [
        "AAPL",
        "ADP",
        "AMZN",
        "ANET",
        "AVGO",
        "BSX",
        "CDNS",
        "ETN",
        "GOOGL",
        "HD",
        "KO",
        "LIN",
        "LOW",
        "MA",
        "META",
        "NVDA",
        "PANW",
        "PEP",
        "PG",
        "SNPS",
        "SYK",
        "TJX",
        "UBER",
        "V",
        "WMT",
        "RSG",
        "MCO",
        "ASML",
        "CTAS",
        "PWR",
        "LLY",
    ]

    print("Analyzing stock portfolios...")

    # Define initial_investment for use in the print statement
    initial_investment_amount = 10000

    # Define a risk-free rate (e.g., current yield on a 3-month Treasury bill)
    # For simplicity, using 0.04 (4%) as an example. Adjust as needed.
    risk_free_rate_annual = 0.04

    results_df = find_best_stock_portfolio(
        stocks_list,
        initial_investment=initial_investment_amount,
        months=60,
        risk_free_rate=risk_free_rate_annual,
        min_weight_per_stock=0.00,  # Example: minimum 1% allocation to any stock
        max_weight_per_stock=0.25,  # Example: maximum 25% allocation to any single stock
    )

    #print stocks from heirarchical risk parity portfolio
    if not results_df.empty:
        hrp_portfolio = results_df[results_df["Portfolio Type"] == "Hierarchical Risk Parity"]
        # splits stocks dictionary into separate columns as a csv
        if not hrp_portfolio.empty:
            hrp_stocks = hrp_portfolio["Stocks"].iloc[0]
            print("\nHierarchical Risk Parity Portfolio:")
            for stock in hrp_stocks:
                for ticker, weight in stock.items():
                    print(f"{ticker},{weight}")
        else:
            print("No Hierarchical Risk Parity portfolio found.")
    else:
        print("No results to display.")

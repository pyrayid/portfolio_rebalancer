import pandas as pd
import numpy as np
import yfinance as yf
import os

# === User inputs ===
date = "Jun-27-2025"
optimization_method = "Hierarchical Risk Parity"  # Options: "Equal Weight", "Hierarchical Risk Parity"
cash_to_invest = 0  # Cash to invest (0 to use pct)
cash_to_invest_pct = 0  # Percentage of cash available to invest
cash_to_withdraw = 10000
# === User inputs END ===

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
    "NVO",
    "WELL",
    "EQIX"
]

# Load the portfolio CSV file
file_path = f"files/Portfolio_Positions_{date}.csv"
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' does not exist.")
    exit(1)

df_positions = pd.read_csv(file_path)[
    ["Symbol", "Description", "Quantity", "Last Price", "Current Value"]
]
df_positions = df_positions.dropna(subset=["Symbol"])

# Clean and convert numeric columns
df_positions["Quantity"] = pd.to_numeric(df_positions["Quantity"], errors="coerce")
df_positions["Current Value"] = pd.to_numeric(
    df_positions["Current Value"].replace(r"[\$,]", "", regex=True), errors="coerce"
)

# Extract cash position (FDRXX**)
if "FDRXX**" in df_positions["Symbol"].values:
    cash_available = df_positions.loc[
        df_positions["Symbol"] == "FDRXX**", "Current Value"
    ].iloc[0]
    # calculate cash to invest
    if cash_to_invest == 0:
        cash_to_invest = cash_available * cash_to_invest_pct
    df_positions = df_positions[df_positions["Symbol"] != "FDRXX**"]
else:
    cash_available = 0.0

# Build a full target DataFrame including missing tickers
df = pd.DataFrame({"Symbol": stocks_list})
df = df.merge(df_positions, on="Symbol", how="left").fillna(
    {"Description": "", "Quantity": 0, "Current Value": 0.0}
)


# Fetch live or existing prices
failed_downloads = 0
def fetch_price(row):
    global failed_downloads
    ticker = row["Symbol"]
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[
            -1
        ]  # Get the latest closing price
        if price is not None and not np.isnan(price):
            print(f"Fetched price for {ticker}: {price}")
            return price
    except Exception as e:
        failed_downloads += 1
        if "Too Many Requests. Rate limited. Try after a while." in str(e):
            print(f"Rate limit exceeded for {ticker}. Using Last Price.")
            return float(row["Last Price"].replace("$", "").replace(",", ""))
        else:
            print(f"Failed to fetch price for {ticker}: {e}")
    return np.nan


df["price"] = df.apply(fetch_price, axis=1)

if failed_downloads >= 1:
    print("Too many failed downloads. Stopping execution.")
    exit(1)

df["Current Value"] = df["Quantity"] * df["price"]

# === Buying logic ===

if optimization_method == "Equal Weight":
    # Equal weight for each ticker
    equal_weight = 1.0 / len(stocks_list)
    df["target_weight"] = equal_weight
elif optimization_method == "Hierarchical Risk Parity":
    # HRP logic from folio_optimizer.py
    from pypfopt.hierarchical_portfolio import HRPOpt
    # from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage

    # Fetch historical data
    # Use 5 years of monthly data for HRP optimization
    data = yf.download(stocks_list, period="1y", interval="1d", progress=False)
    #monthly_prices = data["Close"].resample("ME").last().dropna()
    # returns = monthly_prices.pct_change().dropna()
    daily_prices = data["Close"]
    daily_returns = daily_prices.pct_change().dropna()              # compute daily pct‐change
    # daily_returns = daily_returns.tail(252)                         # keep only the last 252 trading days
    returns = daily_returns 

    # # Estimate expected returns and covariance matrix
    # mu = mean_historical_return(monthly_prices)
    S = CovarianceShrinkage(daily_prices).ledoit_wolf()

    # Hierarchical Risk Parity Portfolio
    hrp = HRPOpt(returns, S)
    hrp.optimize()
    hrp_weights = hrp.clean_weights()

    # Assign HRP weights to the DataFrame
    df["target_weight"] = df["Symbol"].apply(
        lambda x: hrp_weights.get(x, 0)
    )  # Use get to handle missing keys

# Compute current weights
current_value_total = df["Current Value"].sum()
df["current_weight"] = df["Current Value"] / (
    current_value_total if current_value_total > 0 else np.nan
)

# Smart Buys: proportional underweights
df["underweight"] = (df["target_weight"] - df["current_weight"].fillna(0)).clip(lower=0)
total_underweight = df["underweight"].sum()
df["buy_allocation"] = np.where(
    total_underweight > 0, df["underweight"] / total_underweight * cash_to_invest, 0.0
)

# Compute raw share counts (may contain nan/inf)
raw_shares = df["buy_allocation"] / df["price"]
# Replace any NaN or infinite with 0
df["shares_to_buy"] = raw_shares.fillna(0).round(2)  # .apply(np.floor).astype(int)


# === Selling logic ===

# Calculate overweights
df["overweight"] = (df["current_weight"] - df["target_weight"]).clip(lower=0)

# Calculate the cash value of the overweight portion for each stock
df["sell_allocation_overweight"] = df["overweight"] * current_value_total

# Calculate total selling value
total_sell_value = df["sell_allocation_overweight"].sum()

# If total sell value is less than cash_to_withdraw, sell from above-average positions
if total_sell_value < cash_to_withdraw:
    remaining_to_sell = cash_to_withdraw - total_sell_value

    # Calculate target portfolio value after full withdrawal
    target_total_value = current_value_total - cash_to_withdraw

    # Calculate target value for each position based on target weight
    df["final_target_value"] = df["target_weight"] * target_total_value

    # Calculate required selling to reach target value
    df["sell_allocation_final"] = df["Current Value"] - df["final_target_value"]

    # Clean up temporary columns
    # df = df.drop(["value_after_overweight", "final_target_value"], axis=1)

    # Ensure we don't sell more than we own
    df["sell_allocation_final"] = df[["sell_allocation_final", "Current Value"]].min(
        axis=1
    )

    # Determine shares to sell based on live prices
    df["shares_to_sell"] = (
        (df["sell_allocation_final"] / df["price"]).fillna(0).round(2)
    )

else:
    df["shares_to_sell"] = (
        (df["sell_allocation_overweight"] / df["price"]).fillna(0).round(2)
    )



print(f"Cash available from FDRXX**: ${cash_available:,.2f}")
print(f"Total cash to invest:        ${cash_to_invest:,.2f}")
print(f"Total cash to withdraw:      ${cash_to_withdraw:,.2f}")
print("Using optimization method:", optimization_method)
# save to csv
df.to_csv(f"files/Portfolio_Rebalance_{optimization_method}_{date}.csv", index=False)

import pandas as pd
import numpy as np
import yfinance as yf
import os
import fear_and_greed
import matplotlib.pyplot as plt


def fetch_price_history(stock_list, period="1y", interval="1d"):
    """
    Fetch historical close prices for all stocks in stock_list.
    Returns a DataFrame: index=date, columns=symbols.
    """
    data = yf.download(stock_list, period=period, interval=interval, progress=False)
    return data["Close"]


def plot_portfolio_value(
    price_history, stock_list, stock_weights=None, title="Portfolio Value Over Time"
):
    """
    Plot the weighted portfolio value over time.
    """
    if stock_weights is None:
        n = len(stock_list)
        stock_weights = {s: 1.0 / n for s in stock_list}
    aligned = price_history[stock_list].dropna()
    weights = np.array([stock_weights[s] for s in stock_list])
    portfolio_values = aligned.values @ weights
    plt.figure(figsize=(10, 5))
    plt.plot(aligned.index, portfolio_values, label="Portfolio Value")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def drawdown_trigger_invest_amount(
    stock_list,
    cash_available,
    price_history,
    stock_weights=None,
    drawdown_threshold=0.15,
    invest_pct=0.2,
    withdraw_pct=0.2,
):
    """
    Determine invest/withdraw amount based on current drawdown of the portfolio value.
    stock_weights: dict or None. If None, uses equal weights.
    """
    if stock_weights is None:
        n = len(stock_list)
        stock_weights = {s: 1.0 / n for s in stock_list}
    aligned = price_history[stock_list].dropna()
    weights = np.array([stock_weights[s] for s in stock_list])
    portfolio_values = aligned.values @ weights
    roll_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - roll_max) / roll_max
    current_drawdown = drawdown[-1]
    print(f"Current Drawdown: {current_drawdown:.2f}")
    if current_drawdown < -drawdown_threshold:
        return cash_available * invest_pct
    elif current_drawdown > drawdown_threshold:  # Near new highs
        return -cash_available * withdraw_pct
    else:
        return 0


def mean_reversion_zscore_invest_amount(
    stock_list,
    cash_available,
    price_history,
    stock_weights=None,
    zscore_threshold=-1.5,
    invest_pct=0.2,
    withdraw_pct=0.2,
):
    """
    Determine invest/withdraw amount based on Z-score of portfolio value.
    stock_weights: dict or None. If None, uses equal weights.
    """
    if stock_weights is None:
        n = len(stock_list)
        stock_weights = {s: 1.0 / n for s in stock_list}
    aligned = price_history[stock_list].dropna()
    weights = np.array([stock_weights[s] for s in stock_list])
    portfolio_values = aligned.values @ weights
    # Calculate Z-score based on last 100 days
    mean = portfolio_values[-100:].mean()
    std = portfolio_values[-100:].std()
    if std == 0:
        return 0
    z = (portfolio_values[-1] - mean) / std
    print(f"Mean Reversion Z-score: {z:.2f}")
    if z < zscore_threshold:
        return cash_available * invest_pct
    elif z > abs(zscore_threshold):
        return -cash_available * withdraw_pct
    else:
        return 0


def determine_cash_allocation(
    stocks_list,
    cash_available=0.0,
    stocks_weights=None,
):
    # Determine cash settings
    # 1) Fetch the CNN Fear & Greed Index to determine cash allocations
    #    - If using manual settings, skip this step
    #    - If using FGI, set cash to invest/withdraw based on index value
    #           FGI < 25: Extreme Fear (buy for $10,000)
    #           25 ≤ FGI < 50: Fear (buy for $5,000)
    #           50 ≤ FGI < 70: Neutral (no action)
    #           70 ≤ FGI < 80: Greed (sell for $5,000)
    #           FGI ≥ 80: Extreme Greed (sell for $10,000)

    use_manual_cash_settings = False  # Set to True to use manual cash settings
    use_fgn = True  # Set to True to use Fear & Greed Index
    use_drawdown_trigger = True  # Set to True to use drawdown trigger logic
    use_mean_reversion_zscore = True  # Set to True to use mean reversion Z-score logic

    if use_manual_cash_settings:
        cash_to_invest = 0  # Set your manual cash to invest
        cash_to_withdraw = 0  # Set your manual cash to withdraw
        cash_pct = 0.0  # Set your manual cash_pct (e.g., 0.2 for 20%)
        print(
            f"Manual cash settings used: Invest ${cash_to_invest}, Withdraw ${cash_to_withdraw}, Cash % {cash_pct:.2%}"
        )
        return cash_to_invest, cash_to_withdraw, cash_pct

    if use_fgn:
        # Fetch CNN Fear & Greed Index and set cash allocations
        print("Fetching Fear & Greed Index...")
        try:
            fg = fear_and_greed.get()
            fear_greed_value = round(fg.value)
            fear_greed_desc = fg.description.upper()
            print(f"Fear & Greed Index: {fear_greed_value} {fear_greed_desc}")
        except Exception as e:
            print(f"Failed to fetch Fear & Greed Index: {e}")
            fear_greed_value = None
            fear_greed_desc = None

        # Set buy/sell logic based on index
        if fear_greed_value is not None:
            if fear_greed_value < 25:
                cash_to_invest = 10000
                cash_to_withdraw = 0
                cash_pct = 0.0
                print(f"    - BUY for ${cash_to_invest}")
            elif 25 <= fear_greed_value < 50:
                # 25–49: Fear, invest a small amount if cash is available
                cash_to_invest = 5000
                cash_to_withdraw = 0
                cash_pct = 0.2
                print(f"    - BUY for ${cash_to_invest}")
            elif 50 <= fear_greed_value < 70:
                # 50–74: Greed aka Neutral zone, do nothing
                cash_to_invest = 0
                cash_to_withdraw = 0
                cash_pct = 0.0
                print("    - Neutral zone, do nothing.")
            elif 70 <= fear_greed_value < 80:
                # 75–100: Extreme Greed
                cash_to_invest = 0
                cash_to_withdraw = 5000
                cash_pct = 0.3
                print(f"    - SELL for ${cash_to_withdraw}")
            elif fear_greed_value >= 80:  # 75–100: Extreme Greed
                cash_to_invest = 0
                cash_to_withdraw = 10000
                cash_pct = 0.5
                print(f"    - SELL for ${cash_to_withdraw}")
            else:
                cash_to_invest = 0
                cash_to_withdraw = 0
                cash_pct = 0.0
                print("     - Do nothing.")

    if use_drawdown_trigger or use_mean_reversion_zscore:
        price_history = fetch_price_history(stocks_list, period="1y", interval="1d")

        plot_portfolio_value(
            price_history,
            stocks_list,
            stocks_weights,
            title="Portfolio Value Over Time",
        )
        if use_drawdown_trigger:
            cash_change = drawdown_trigger_invest_amount(
                stocks_list, cash_available, price_history, stocks_weights
            )
            print(f"Drawdown trigger based cash settings: {cash_change}")
        if use_mean_reversion_zscore:
            cash_change = mean_reversion_zscore_invest_amount(
                stocks_list, cash_available, price_history, stocks_weights
            )
            print(f"Mean Reversion Z-score based cash settings: {cash_change}")
        cash_to_invest = cash_change if cash_change > 0 else 0
        cash_to_withdraw = -cash_change if cash_change < 0 else 0
        cash_pct = 0.0

    return cash_to_invest, cash_to_withdraw, cash_pct


stocks_list = [
    "AAPL",
    "ADP",
    "AMZN",
    "ANET",
    "ASML",
    "AVGO",
    "BSX",
    "CDNS",
    "CEG",
    "CL",
    "CTAS",
    "CWST",
    "ETN",
    "GOOGL",
    "HD",
    "KMB",
    "KO",
    "LIN",
    "LLY",
    "LOW",
    "MA",
    "MCO",
    "META",
    "NVDA",
    "NVO",
    "PANW",
    "PEP",
    "PG",
    "PWR",
    "RSG",
    "SNPS",
    "SYK",
    "TJX",
    "UBER",
    "V",
    "WCN",
    "WMT",
    "WTS",
    "SGOL",
]

# How this works:
# 1) Determine cash allocations
# 2) Fetch the portfolio file for the specified date
# 3) Rebalance the portfolio using the specified optimization method


# === User inputs START ===
date = "Sep-04-2025"  # Date for the portfolio file
optimization_method = (
    "Hierarchical Risk Parity"  # Options: "Equal Weight", "Hierarchical Risk Parity"
)
rebalancing_mode = True  # Set to True to rebalance by selling shares
# === User inputs END ===


# Load the portfolio CSV file
file_path = os.path.join("files", f"Portfolio_Positions_{date}.csv")
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' does not exist.")
    exit(1)

df_positions = pd.read_csv(file_path)[
    [
        "Symbol",
        "Description",
        "Quantity",
        "Last Price",
        "Current Value",
        "Cost Basis Total",
        "Total Gain/Loss Dollar",
    ]
]
df_positions = df_positions.dropna(subset=["Symbol"])

# Clean and convert numeric columns
df_positions["Quantity"] = pd.to_numeric(df_positions["Quantity"], errors="coerce")
df_positions["Current Value"] = pd.to_numeric(
    df_positions["Current Value"].replace(r"[\$,]", "", regex=True), errors="coerce"
)
df_positions["Cost Basis Total"] = pd.to_numeric(
    df_positions["Cost Basis Total"].replace(r"[\$,]", "", regex=True), errors="coerce"
)
df_positions["Total Gain/Loss Dollar"] = pd.to_numeric(
    df_positions["Total Gain/Loss Dollar"].replace(r"[\$,]", "", regex=True),
    errors="coerce",
)

# Extract cash position (FDRXX** and SPSK)
cash_available = 0.0
if "FDRXX**" in df_positions["Symbol"].values:
    cash_available += df_positions.loc[
        df_positions["Symbol"] == "FDRXX**", "Current Value"
    ].iloc[0]
if "SPSK" in df_positions["Symbol"].values:
    cash_available += df_positions.loc[
        df_positions["Symbol"] == "SPSK", "Current Value"
    ].iloc[0]

# Build a full target DataFrame including missing tickers
df = pd.DataFrame({"Symbol": stocks_list})
df = df.merge(df_positions, on="Symbol", how="left").fillna(
    {"Description": "", "Quantity": 0, "Current Value": 0.0}
)

# Compute current weights
current_value_total = df["Current Value"].sum()
df["current_weight"] = df["Current Value"] / (
    current_value_total if current_value_total > 0 else np.nan
)

# Determine cash allocations
cash_to_invest, cash_to_withdraw, cash_pct = determine_cash_allocation(
    stocks_list,
    cash_available,
    stocks_weights=dict(zip(df["Symbol"], df["current_weight"])),
)


def fetch_price(row, failed_downloads_count):
    ticker = row["Symbol"]
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[
            -1
        ]  # Get the latest closing price
        if price is not None and not np.isnan(price):
            # print(f"Fetched price for {ticker}: {price}")
            return price, failed_downloads_count
    except Exception as e:
        failed_downloads_count += 1
        if "Too Many Requests. Rate limited. Try after a while." in str(e):
            print(f"Rate limit exceeded for {ticker}. Using Last Price.")
            return float(
                row["Last Price"].replace("$", "").replace(",", "")
            ), failed_downloads_count
        else:
            print(f"Failed to fetch price for {ticker}: {e}")
    return np.nan, failed_downloads_count


# Initialize failed_downloads before the loop
failed_downloads = 0
prices = []
for index, row in df.iterrows():
    price, failed_downloads = fetch_price(row, failed_downloads)
    prices.append(price)

df["price"] = prices

if failed_downloads >= 1:
    print("Too many failed downloads. Stopping execution.")
    exit(1)

# update Current Value based on fetched prices
df["Current Value"] = df["Quantity"] * df["price"]
# Compute current weights
current_value_total = df["Current Value"].sum()
df["current_weight"] = df["Current Value"] / (
    current_value_total if current_value_total > 0 else np.nan
)


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
    data = yf.download(stocks_list, period="2y", interval="1d", progress=False)
    daily_prices = data["Close"]
    daily_returns = daily_prices.pct_change(
        fill_method=None
    ).dropna()  # compute daily pct‐change
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

if rebalancing_mode:
    # Rebalance using both current holdings and any cash_to_invest/withdraw
    cash_adjustment = cash_to_invest - cash_to_withdraw
else:
    cash_adjustment = cash_to_invest - cash_to_withdraw

new_total_value = current_value_total + cash_adjustment

# target dollar value per ticker on the adjusted total
df["target_value"] = df["target_weight"] * new_total_value

# Re-calculate current_value_total after price updates
current_value_total = df["Current Value"].sum()

# Calculate adjusted total value for target allocation
adjusted_total = current_value_total + cash_adjustment


# Calculate total portfolio value including current cash
total_portfolio_value = current_value_total + cash_available

# Calculate raw (un-clipped) trade dollars if rebalancing fully
raw_trade = df["target_weight"] * adjusted_total - df["Current Value"]

# Calculate target cash based on cash_pct of total portfolio value (excluding cash_adjustment)
target_cash = cash_pct * total_portfolio_value

# Initialize trade-related variables
available_for_buy = 0.0
allowed_sell = 0.0

# Decide if we're only buying or only selling today
if rebalancing_mode:
    # Rebalance: ignore cash flows, trade to target weights
    df["trade_value"] = df["target_value"] - df["Current Value"]
elif cash_adjustment > 0:
    # Only allocate buys
    buys = raw_trade.clip(lower=0)
    # Only buy if cash available after trade stays above target_cash
    available_for_buy = min(cash_adjustment, max(0, cash_available - target_cash))
    if buys.sum() > 0 and available_for_buy > 0:
        df["trade_value"] = buys * (available_for_buy / buys.sum())
    else:
        df["trade_value"] = 0

elif cash_adjustment < 0:
    # Only allocate sells where profit > 0
    df["profit"] = df["Current Value"] - df["Cost Basis Total"]
    sell_mask = (raw_trade < 0) & (df["profit"] > 0)
    sells = raw_trade.where(sell_mask, 0)

    # Calculate max cash we can add from selling without dropping below target_cash
    # The amount to sell is primarily driven by the withdrawal request.
    # We also consider selling enough to get back to our target cash reserve if we are below it.
    amount_to_raise_by_selling = abs(cash_adjustment) + max(
        0, target_cash - cash_available
    )
    allowed_sell = amount_to_raise_by_selling

    total_sells = -sells.sum() if sells.sum() < 0 else 0

    if total_sells > 0 and allowed_sell > 0:
        df["trade_value"] = sells * (allowed_sell / total_sells)
    elif sells.sum() != 0:  # Fallback if allowed_sell is 0 but there are sells
        df["trade_value"] = sells * (
            abs(cash_adjustment) / total_sells
        )  # Use abs(cash_adjustment) here
    else:
        df["trade_value"] = 0

else:
    # No cash change → no trades
    df["trade_value"] = 0

# 4) convert to shares
df["shares_to_trade"] = (df["trade_value"] / df["price"]).round(2)
df["shares_to_buy"] = df["shares_to_trade"].clip(lower=0)
df["shares_to_sell"] = (-df["shares_to_trade"]).clip(lower=0)


print("=" * 50)
print("Optimization method:", optimization_method)
print("Rebalancing mode:", rebalancing_mode)
print(
    f"Total portfolio + cash value:     ${(current_value_total + cash_available):,.2f}"
)
print(f"Total Gain/Loss value:            ${df['Total Gain/Loss Dollar'].sum():,.2f} ({(df['Total Gain/Loss Dollar'].sum() / (current_value_total + cash_available) if current_value_total > 0 else np.nan):.2%})")

print(
    f"Cash available (FDRXX** & SPSK):  ${cash_available:,.2f} ({(cash_available / (current_value_total + cash_available)):.2%})"
)
print(f"Target cash ({cash_pct:.2%}):              ${target_cash:,.2f}")
print(f"Total cash to invest:             ${cash_to_invest:,.2f}")
print(f"Total cash to withdraw:           ${cash_to_withdraw:,.2f}")
if cash_adjustment > 0:
    print(f"Cash available for buys:          ${available_for_buy:,.2f}")
    print(
        f"Cash after buys (est.):           ${(cash_available - available_for_buy):,.2f}"
    )
elif cash_adjustment < 0:
    print(
        f"Max cash allowed from sells:      ${allowed_sell if 'allowed_sell' in locals() else 0:,.2f}"
    )
    print(
        f"Cash after sells (est.):          ${(cash_available + (allowed_sell if 'allowed_sell' in locals() else 0)):,.2f}"
    )
print("=" * 50)

# save to csv
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = os.path.join(
    "files", f"Portfolio_Rebalance_{optimization_method}_{date}_{timestamp}.csv"
)
# sort by shares_to_sell descending, then shares_to_buy descending
df = df.sort_values(by=["shares_to_sell", "shares_to_buy"], ascending=[False, False])
df.to_csv(output_filename, index=False)

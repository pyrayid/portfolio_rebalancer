import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date
import plotly.express as px

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

st.title("Portfolio Rebalancer")
st.markdown(
    "Rebalance your portfolio with this tool: <br>1. Enter the amount of cash you want to invest or withdraw. <br>2. Current portfolio is prepopulated with a balanced portfolio of 30+ stocks selected for their growth potential and stability. You can update this with your own portfolio and click calculate. <br><br>The chart will show how your portfolio would have grown over the past 60 months compared to the balanced portfolio. The rebalancing recommendations will suggest how to invest or withdraw cash based on stock prices today and the optimization method selected.",
    unsafe_allow_html=True,
)

# User Input
st.header("Current Portfolio")

portfolio_input_method = st.radio(
    "Choose portfolio input method:", ("Manual Entry", "Upload CSV"), horizontal=True
)

if portfolio_input_method == "Manual Entry":
    # Create a DataFrame for portfolio input
    portfolio_df = pd.DataFrame(
        {"Symbol": stocks_list, "Shares": [0] * len(stocks_list)}
    )
    portfolio_df = st.data_editor(
        portfolio_df, num_rows="dynamic", hide_index=False
    )  # , num_rows="fixed", disabled=["Symbol"])
elif portfolio_input_method == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload portfolio CSV(Symbol,Shares)", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            # Ensure the uploaded CSV has Symbol and Shares columns
            if "Symbol" in uploaded_df.columns and "Shares" in uploaded_df.columns:
                # Create a DataFrame for portfolio input
                portfolio_df = pd.DataFrame(
                    {"Symbol": stocks_list, "Shares": [0] * len(stocks_list)}
                )
                # Iterate through the uploaded data and update the Shares values in portfolio_df
                for index, row in uploaded_df.iterrows():
                    symbol = row["Symbol"]
                    shares = row["Shares"]
                    # Find the matching row in portfolio_df
                    match = portfolio_df["Symbol"] == symbol
                    if any(match):  # Check if there's at least one match
                        portfolio_df.loc[match, "Shares"] = shares
                portfolio_df = st.data_editor(
                    portfolio_df,
                    num_rows="fixed",
                    disabled=["Symbol"],
                    hide_index=False,
                )
            else:
                st.error("Uploaded CSV must contain 'Symbol' and 'Shares' columns.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
else:
    portfolio_df = pd.DataFrame(
        {"Symbol": stocks_list, "Shares": [0] * len(stocks_list)}
    )
    portfolio_df = st.data_editor(portfolio_df, num_rows="dynamic", hide_index=False)


# Sidebar Input
with st.sidebar:
    st.header("Investment/Withdrawal")
    investment_type = st.selectbox("Select action", ["Invest", "Withdraw"])
    amount = st.number_input(
        f"Amount to {investment_type.lower()}", value=0.0, step=0.01
    )
    # amount_type = st.selectbox("Amount type", ["Amount", "Percentage"])
    optimization_method = st.selectbox(
        "Select optimization method", ["Equal Weight", "Hierarchical Risk Parity"]
    )

    if investment_type == "Invest":
        cash_to_invest = amount
        cash_to_withdraw = 0.0
    else:
        cash_to_invest = 0.0
        cash_to_withdraw = amount

    st.write(f"Total cash to invest:        ${cash_to_invest:,.2f}")
    st.write(f"Total cash to withdraw:      ${cash_to_withdraw:,.2f}")


def calculate_portfolio_rebalance(
    portfolio_df, amount, cash_to_invest, cash_to_withdraw, optimization_method
):
    # Create DataFrame from user input
    df_positions = portfolio_df.rename(columns={"Shares": "Quantity"})

    df = pd.DataFrame({"Symbol": portfolio_df["Symbol"].tolist()})
    df = df.merge(df_positions, on="Symbol", how="left").fillna(
        {"Description": "", "Quantity": 0, "Current Value": 0.0}
    )

    # Fetch live or existing prices
    def fetch_price(row):
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
            if "Too Many Requests. Rate limited. Try after a while." in str(e):
                print(f"Rate limit exceeded for {ticker}. Using Last Price.")
                # return float(row["Last Price"].replace("$", "").replace(",", "")) # no last price
                return np.nan
            else:
                print(f"Failed to fetch price for {ticker}: {e}")
        return np.nan

    df["price"] = df.apply(fetch_price, axis=1)
    df["Current Value"] = df["Quantity"] * df["price"]

    # === Buying logic ===
    if optimization_method == "Equal Weight":
        # Equal weight for each ticker
        equal_weight = 1.0 / len(stocks_list)
        df["target_weight"] = equal_weight
    elif optimization_method == "Hierarchical Risk Parity":
        # HRP logic from folio_optimizer.py
        from pypfopt import HRPOpt
        from pypfopt.expected_returns import mean_historical_return
        from pypfopt.risk_models import CovarianceShrinkage

        # Fetch historical data
        tickers = portfolio_df["Symbol"].tolist()
        data = yf.download(tickers, period="144mo", progress=False)
        monthly_prices = data["Close"].resample("ME").last().dropna()
        returns = monthly_prices.pct_change().dropna()

        # Estimate expected returns and covariance matrix
        mu = mean_historical_return(monthly_prices)
        S = CovarianceShrinkage(monthly_prices).ledoit_wolf()

        # Hierarchical Risk Parity Portfolio
        hrp = HRPOpt(returns, S)
        hrp.optimize()
        hrp_weights = hrp.clean_weights()

        # Assign HRP weights to the DataFrame
        df["target_weight"] = df["Symbol"].apply(
            lambda x: hrp_weights.get(x, 0)
        )  # Use get to handle missing keys
    else:
        st.error("Invalid optimization method selected.")
        return

    # Compute current weights
    current_value_total = df["Current Value"].sum()
    df["current_weight"] = df["Current Value"] / (
        current_value_total if current_value_total > 0 else np.nan
    )

    # Smart Buys: proportional underweights
    df["underweight"] = (df["target_weight"] - df["current_weight"].fillna(0)).clip(
        lower=0
    )
    total_underweight = df["underweight"].sum()
    df["buy_allocation"] = np.where(
        total_underweight > 0,
        df["underweight"] / total_underweight * cash_to_invest,
        0.0,
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
        df["sell_allocation_final"] = df[
            ["sell_allocation_final", "Current Value"]
        ].min(axis=1)

        # Determine shares to sell based on live prices
        df["shares_to_sell"] = (
            (df["sell_allocation_final"] / df["price"]).fillna(0).round(2)
        )

        # rename df["sell_allocation_final"] to "sell_allocation"
        df.rename(columns={"sell_allocation_final": "sell_allocation"}, inplace=True)

    else:
        df["shares_to_sell"] = (
            (df["sell_allocation_overweight"] / df["price"]).fillna(0).round(2)
        )
        # rename df["sell_allocation_overweight"] to "sell_allocation"
        df.rename(
            columns={
                "sell_allocation_overweight": "Withdrawal Amount"}, inplace=True,
        )

        df.rename(
            columns={
                "buy_allocation": "Investment Amount",
                "shares_to_buy": "Shares to Buy",
                "shares_to_sell": "Shares to Sell",
            },
            inplace=True,
        )

    # Display results
    st.header("Rebalance Recommendations")

    invest_columns = [
        "Symbol",
        "price",
        "Quantity",
        "Current Value",
        "Investment Amount",
        "Shares to Buy",
    ]

    withdraw_columns = [
        "Symbol",
        "price",
        "Quantity",
        "Current Value",
        "Withdrawal Amount",
        "Shares to Sell",
    ]

    if investment_type == "Invest":
        columns_to_display = invest_columns
        st.dataframe(df[columns_to_display])
    else:
        columns_to_display = withdraw_columns
        st.dataframe(df[columns_to_display])

    # Download CSV
    csv = df[columns_to_display].to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"Portfolio_Rebalance_{date.today()}.csv",
        mime="text/csv",
    )


def plot_portfolio_growth(
    balanced_stocks, portfolio_stocks, initial_investment=10000, months=60
):
    """
    Plots the portfolio growth for the past 60 months, starting with an initial investment.
    """
    end_date = date.today()
    start_date = pd.to_datetime(end_date) - pd.DateOffset(months=months)

    # Fetch monthly price data for the stocks and SPX
    data = yf.download(
        balanced_stocks + ["^GSPC"], start=start_date, end=end_date, interval="1mo"
    )

    # Calculate the portfolio value for each month
    portfolio_value = pd.DataFrame(index=data.index)
    portfolio_value["Balanced Portfolio"] = 0

    for i, row in data.iterrows():
        # Calculate the portfolio value for the current month
        stock_prices = row["Close"][balanced_stocks]
        portfolio_value.loc[i, "Balanced Portfolio"] = (
            stock_prices * (initial_investment / len(balanced_stocks))
        ).sum()

    # Normalize the portfolio value and SPX value to start at initial_investment
    portfolio_value["Balanced Portfolio"] = (
        portfolio_value["Balanced Portfolio"]
        / portfolio_value["Balanced Portfolio"].iloc[0]
        * initial_investment
    )
    data_spx = data["Close"]["^GSPC"]
    data_spx = data_spx / data_spx.iloc[0] * initial_investment

    # Create a Pandas DataFrame to store the portfolio value and SPX value over time
    df = pd.DataFrame(
        {"Balanced Portfolio": portfolio_value["Balanced Portfolio"], "SPX": data_spx}
    )

    # Add "Current Portfolio" if portfolio_stocks is different from stocks
    current_portfolio_better = False
    if balanced_stocks != portfolio_stocks:
        data_current = yf.download(
            portfolio_stocks + ["^GSPC"], start=start_date, end=end_date, interval="1mo"
        )
        portfolio_value["Current Portfolio"] = 0
        for i, row in data_current.iterrows():
            stock_prices = row["Close"][portfolio_stocks]
            portfolio_value.loc[i, "Current Portfolio"] = (
                stock_prices * (initial_investment / len(portfolio_stocks))
            ).sum()
        portfolio_value["Current Portfolio"] = (
            portfolio_value["Current Portfolio"]
            / portfolio_value["Current Portfolio"].iloc[0]
            * initial_investment
        )
        df["Current Portfolio"] = portfolio_value["Current Portfolio"]

    df.index.name = "Date"
    df = df.reset_index()

    # Plot the portfolio growth and SPX growth using streamlit
    y_columns = ["Balanced Portfolio", "SPX"]
    if "Current Portfolio" in df.columns:
        y_columns.append("Current Portfolio")
        # if the last value of "Current Portfolio" is greater than the last value of "Balanced Portfolio", set current_portfolio_better to True
        if df["Current Portfolio"].iloc[-1] > df["Balanced Portfolio"].iloc[-1]:
            current_portfolio_better = True
        else:
            current_portfolio_better = False

    fig = px.line(
        df,
        x="Date",
        y=y_columns,
        title="Portfolio Growth vs SPX for 10k invested 5 years ago",
    )
    st.plotly_chart(fig)

    return current_portfolio_better


if st.button("Calculate"):
    if amount == 0:
        st.error("Amount to invest/withdraw cannot be zero.")
    else:
        current_portfolio_better = plot_portfolio_growth(
            stocks_list, portfolio_df["Symbol"].tolist()
        )
        if not current_portfolio_better and cash_to_invest > 0:
            # Invest all cash in the balanced portfolio
            st.success(
                "Balaced portfolio performed better! Rebalancing recommendations will be for this portfolio."
            )
            balanced_portfolio_df = pd.DataFrame(
                {"Symbol": stocks_list, "Shares": [0] * len(stocks_list)}
            )
            calculate_portfolio_rebalance(
                balanced_portfolio_df,
                amount,
                cash_to_invest,
                cash_to_withdraw,
                optimization_method,
            )
        else:
            if cash_to_invest > 0:
                st.success(
                    "Your current portfolio performed better! Rebalancing recommendations will be for this portfolio."
                )
            calculate_portfolio_rebalance(
                portfolio_df,
                amount,
                cash_to_invest,
                cash_to_withdraw,
                optimization_method,
            )

# if st.button("Plot Portfolio Growth"):
#     plot_portfolio_growth(stocks_list)

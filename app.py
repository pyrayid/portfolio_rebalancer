import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date

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
    portfolio_df = st.data_editor(portfolio_df, num_rows="fixed", disabled=["Symbol"])
elif portfolio_input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload portfolio CSV(Symbol,Shares)", type=["csv"])

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
                    portfolio_df, num_rows="fixed", disabled=["Symbol"]
                )
            else:
                st.error("Uploaded CSV must contain 'Symbol' and 'Shares' columns.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
else:
    portfolio_df = pd.DataFrame(
        {"Symbol": stocks_list, "Shares": [0] * len(stocks_list)}
    )


# Sidebar Input
with st.sidebar:
    st.header("Investment/Withdrawal")
    investment_type = st.selectbox("Select action", ["Invest", "Withdraw"])
    amount = st.number_input(
        f"Amount to {investment_type.lower()}", value=0.0, step=0.01
    )
    # amount_type = st.selectbox("Amount type", ["Amount", "Percentage"])
    # time_period = st.number_input("Time period (days)", value=30, step=1)

    if investment_type == "Invest":
        cash_to_invest = amount
        cash_to_withdraw = 0.0
    else:
        cash_to_invest = 0.0
        cash_to_withdraw = amount

    st.write(f"Total cash to invest:        ${cash_to_invest:,.2f}")
    st.write(f"Total cash to withdraw:      ${cash_to_withdraw:,.2f}")

def calculate_portfolio_rebalance(portfolio_df, amount, cash_to_invest, cash_to_withdraw):
    # Create DataFrame from user input
    df_positions = portfolio_df.rename(columns={"Shares": "Quantity"})

    if amount == 0:
        st.error("Amount to invest/withdraw cannot be zero.")
        return
    
    df = pd.DataFrame({"Symbol": stocks_list})
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

    # Equal weight for each ticker
    equal_weight = 1.0 / len(stocks_list)
    df["target_weight"] = equal_weight

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
            columns={"sell_allocation_overweight": "sell_allocation"}, inplace=True
        )

    # Display results
    st.header("Portfolio Rebalance Recommendations")

    invest_columns = [
        "Symbol",
        "price",
        "Quantity",
        "Current Value",
        "buy_allocation",
        "shares_to_buy",
    ]

    withdraw_columns = [
        "Symbol",
        "price",
        "Quantity",
        "Current Value",
        "sell_allocation",
        "shares_to_sell",
    ]

    if investment_type == "Invest":
        columns_to_display = invest_columns
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

if st.button("Calculate"):
    calculate_portfolio_rebalance(portfolio_df, amount, cash_to_invest, cash_to_withdraw)

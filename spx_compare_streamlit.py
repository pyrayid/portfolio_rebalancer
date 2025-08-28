import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("SPX 2008 vs Recent Chart Comparison")

# Define periods to compare
st.sidebar.header("Select Comparison Periods")
period_1_start = st.sidebar.date_input("2008 Start", pd.to_datetime("2005-01-01"))
period_2_start = st.sidebar.date_input("Recent Start", pd.to_datetime("2022-04-01"))

# Download SPX data
spx = yf.Ticker("^GSPC")
# Use fixed end date: today
end_date = pd.to_datetime("today")
df = spx.history(start=min(period_1_start, period_2_start), end=end_date)


# Slice periods
# Use all data from start to today for each period
# Find index of start dates
def ensure_utc(dt):
    dt = pd.to_datetime(dt)
    if dt.tzinfo is None:
        return dt.tz_localize("UTC")
    else:
        return dt.tz_convert("UTC")


idx_2008 = df.index.searchsorted(ensure_utc(period_1_start))
idx_recent = df.index.searchsorted(ensure_utc(period_2_start))
df_2008 = df.iloc[idx_2008:].copy()
df_recent = df.iloc[idx_recent:].copy()

# Normalize both periods to start at 100 for easier visual comparison
df_2008["Norm"] = df_2008["Close"] / df_2008["Close"].iloc[0] * 100
df_recent["Norm"] = df_recent["Close"] / df_recent["Close"].iloc[0] * 100

# Reset index for both to align x-axis (days since start)
df_2008 = df_2008.reset_index()
df_recent = df_recent.reset_index()
df_2008["Day"] = range(len(df_2008))
df_recent["Day"] = range(len(df_recent))

# User sets number of days to plot
st.sidebar.header("Chart Settings")
days_default = min(1000, min(len(df_2008), len(df_recent)))
days_to_plot = st.sidebar.number_input(
    "Days Since Period Start",
    min_value=1,
    max_value=5000,
    value=1000,
)

# Slice to user-selected number of days, pad with NaN if needed
plot_2008 = df_2008.iloc[:days_to_plot].copy()
plot_recent = df_recent.iloc[:days_to_plot].copy()

# Pad with NaN if less than days_to_plot
if len(plot_2008) < days_to_plot:
    pad_len = days_to_plot - len(plot_2008)
    pad_df = pd.DataFrame(
        {
            "Day": range(len(plot_2008), days_to_plot),
            "Norm": [float("nan")] * pad_len,
        }
    )
    plot_2008 = pd.concat([plot_2008, pad_df], ignore_index=True)
if len(plot_recent) < days_to_plot:
    pad_len = days_to_plot - len(plot_recent)
    pad_df = pd.DataFrame(
        {
            "Day": range(len(plot_recent), days_to_plot),
            "Norm": [float("nan")] * pad_len,
        }
    )
    plot_recent = pd.concat([plot_recent, pad_df], ignore_index=True)

# Plot with aligned x-axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(plot_2008["Day"], plot_2008["Norm"], label=f"2008 ({period_1_start} onward)")
ax.plot(
    plot_recent["Day"], plot_recent["Norm"], label=f"Recent ({period_2_start} onward)"
)
ax.set_xlabel("Days Since Period Start")
ax.set_ylabel("Normalized SPX Close (Start=100)")
ax.set_title("SPX Price Action: 2008 vs Recent (Aligned Start)")
ax.legend()
st.pyplot(fig)

st.write(
    f"Showing the first {days_to_plot} days from each period. Extra days are left blank if data is not available. Adjust the slider in the sidebar to change the window."
)

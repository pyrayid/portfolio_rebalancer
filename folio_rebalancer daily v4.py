"""
Portfolio Rebalancer - Modular Version
=====================================

A comprehensive portfolio rebalancing tool that supports multiple optimization methods
and cash allocation strategies. This module provides functionality for:

- Portfolio data loading and processing
- Multiple cash allocation strategies (Fear & Greed, drawdown trigger, mean reversion)
- Portfolio optimization methods (Equal Weight, Hierarchical Risk Parity)
- Automated rebalancing with buy/sell recommendations

Author: Portfolio Management System
Date: 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import fear_and_greed
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.risk_models import CovarianceShrinkage


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================


class Config:
    """Configuration constants for the portfolio rebalancer."""

    # Default stock list
    DEFAULT_STOCKS = [
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

    # Cash allocation settings
    CASH_SYMBOLS = ["FDRXX**", "SPSK"]

    # Fear & Greed Index thresholds
    FGI_EXTREME_FEAR = 25
    FGI_FEAR = 50
    FGI_NEUTRAL = 70
    FGI_GREED = 80

    # Investment amounts for FGI
    FGI_EXTREME_FEAR_AMOUNT = 10000
    FGI_FEAR_AMOUNT = 5000
    FGI_GREED_AMOUNT = 5000
    FGI_EXTREME_GREED_AMOUNT = 10000

    # Cash percentages for FGI
    FGI_FEAR_CASH_PCT = 0.2
    FGI_GREED_CASH_PCT = 0.3
    FGI_EXTREME_GREED_CASH_PCT = 0.5

    # Drawdown and mean reversion settings
    DEFAULT_DRAWDOWN_THRESHOLD = 0.15
    DEFAULT_INVEST_PCT = 0.2
    DEFAULT_WITHDRAW_PCT = 0.2
    DEFAULT_ZSCORE_THRESHOLD = -1.5

    # Data fetching settings
    PRICE_HISTORY_PERIOD = "1y"
    PRICE_HISTORY_INTERVAL = "1d"
    HRP_DATA_PERIOD = "2y"
    HRP_DATA_INTERVAL = "1d"
    ZSCORE_LOOKBACK_DAYS = 100


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


class DataFetcher:
    """Handles data fetching operations for stocks and market indicators."""

    @staticmethod
    def fetch_price_history(
        stock_list: List[str], period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical close prices for all stocks in stock_list.

        Args:
            stock_list: List of stock symbols
            period: Time period for data (default: "1y")
            interval: Data interval (default: "1d")

        Returns:
            DataFrame with index=date, columns=symbols
        """
        data = yf.download(stock_list, period=period, interval=interval, progress=False)
        return data["Close"]

    @staticmethod
    def fetch_current_price(
        ticker: str, fallback_price: Optional[float] = None
    ) -> Tuple[float, bool]:
        """
        Fetch current price for a single ticker.

        Args:
            ticker: Stock symbol
            fallback_price: Price to use if fetch fails

        Returns:
            Tuple of (price, success_flag)
        """
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")["Close"].iloc[-1]
            if price is not None and not np.isnan(price):
                return float(price), True
        except Exception as e:
            if "Too Many Requests. Rate limited. Try after a while." in str(e):
                print(f"Rate limit exceeded for {ticker}. Using fallback price.")
                if fallback_price is not None:
                    return float(fallback_price), False
            else:
                print(f"Failed to fetch price for {ticker}: {e}")

        return np.nan, False

    @staticmethod
    def fetch_fear_greed_index() -> Tuple[Optional[int], Optional[str]]:
        """
        Fetch CNN Fear & Greed Index.

        Returns:
            Tuple of (index_value, description)
        """
        try:
            fg = fear_and_greed.get()
            return round(fg.value), fg.description.upper()
        except Exception as e:
            print(f"Failed to fetch Fear & Greed Index: {e}")
            return None, None


class PortfolioVisualizer:
    """Handles portfolio visualization and plotting."""

    @staticmethod
    def plot_portfolio_value(
        price_history: pd.DataFrame,
        stock_list: List[str],
        stock_weights: Optional[Dict[str, float]] = None,
        title: str = "Portfolio Value Over Time",
    ) -> None:
        """
        Plot the weighted portfolio value over time.

        Args:
            price_history: DataFrame with historical prices
            stock_list: List of stock symbols
            stock_weights: Dictionary of stock weights (default: equal weights)
            title: Plot title
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


# =============================================================================
# CASH ALLOCATION STRATEGIES
# =============================================================================


class CashAllocationStrategy:
    """Base class for cash allocation strategies."""

    def calculate_allocation(
        self,
        stocks_list: List[str],
        cash_available: float,
        stocks_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate cash allocation.

        Returns:
            Tuple of (cash_to_invest, cash_to_withdraw, cash_pct)
        """
        raise NotImplementedError


class FearGreedStrategy(CashAllocationStrategy):
    """Cash allocation based on CNN Fear & Greed Index."""

    def calculate_allocation(
        self,
        stocks_list: List[str],
        cash_available: float,
        stocks_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """Calculate allocation based on Fear & Greed Index."""
        print("Fetching Fear & Greed Index...")

        fear_greed_value, fear_greed_desc = DataFetcher.fetch_fear_greed_index()

        if fear_greed_value is None:
            print("Failed to fetch Fear & Greed Index. Using neutral allocation.")
            return 0.0, 0.0, 0.0

        print(f"Fear & Greed Index: {fear_greed_value} {fear_greed_desc}")

        if fear_greed_value < Config.FGI_EXTREME_FEAR:
            print(f"    - BUY for ${Config.FGI_EXTREME_FEAR_AMOUNT}")
            return Config.FGI_EXTREME_FEAR_AMOUNT, 0.0, 0.0
        elif Config.FGI_EXTREME_FEAR <= fear_greed_value < Config.FGI_FEAR:
            print(f"    - BUY for ${Config.FGI_FEAR_AMOUNT}")
            return Config.FGI_FEAR_AMOUNT, 0.0, Config.FGI_FEAR_CASH_PCT
        elif Config.FGI_FEAR <= fear_greed_value < Config.FGI_NEUTRAL:
            print("    - Neutral zone, do nothing.")
            return 0.0, 0.0, 0.0
        elif Config.FGI_NEUTRAL <= fear_greed_value < Config.FGI_GREED:
            print(f"    - SELL for ${Config.FGI_GREED_AMOUNT}")
            return 0.0, Config.FGI_GREED_AMOUNT, Config.FGI_GREED_CASH_PCT
        elif fear_greed_value >= Config.FGI_GREED:
            print(f"    - SELL for ${Config.FGI_EXTREME_GREED_AMOUNT}")
            return (
                0.0,
                Config.FGI_EXTREME_GREED_AMOUNT,
                Config.FGI_EXTREME_GREED_CASH_PCT,
            )
        else:
            print("     - Do nothing.")
            return 0.0, 0.0, 0.0


class DrawdownTriggerStrategy(CashAllocationStrategy):
    """Cash allocation based on portfolio drawdown."""

    def __init__(
        self,
        drawdown_threshold: float = Config.DEFAULT_DRAWDOWN_THRESHOLD,
        invest_pct: float = Config.DEFAULT_INVEST_PCT,
        withdraw_pct: float = Config.DEFAULT_WITHDRAW_PCT,
    ):
        self.drawdown_threshold = drawdown_threshold
        self.invest_pct = invest_pct
        self.withdraw_pct = withdraw_pct

    def calculate_allocation(
        self,
        stocks_list: List[str],
        cash_available: float,
        stocks_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """Calculate allocation based on portfolio drawdown."""
        if stocks_weights is None:
            n = len(stocks_list)
            stocks_weights = {s: 1.0 / n for s in stocks_list}

        price_history = DataFetcher.fetch_price_history(
            stocks_list, Config.PRICE_HISTORY_PERIOD, Config.PRICE_HISTORY_INTERVAL
        )

        aligned = price_history[stocks_list].dropna()
        weights = np.array([stocks_weights[s] for s in stocks_list])
        portfolio_values = aligned.values @ weights

        roll_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - roll_max) / roll_max
        current_drawdown = drawdown[-1]

        print(f"Current Drawdown: {current_drawdown:.2f}")

        if current_drawdown < -self.drawdown_threshold:
            cash_change = cash_available * self.invest_pct
        elif current_drawdown > self.drawdown_threshold:
            cash_change = -cash_available * self.withdraw_pct
        else:
            cash_change = 0

        print(f"Drawdown trigger based cash settings: {cash_change}")

        cash_to_invest = cash_change if cash_change > 0 else 0
        cash_to_withdraw = -cash_change if cash_change < 0 else 0

        return cash_to_invest, cash_to_withdraw, 0.0


class MeanReversionStrategy(CashAllocationStrategy):
    """Cash allocation based on mean reversion Z-score."""

    def __init__(
        self,
        zscore_threshold: float = Config.DEFAULT_ZSCORE_THRESHOLD,
        invest_pct: float = Config.DEFAULT_INVEST_PCT,
        withdraw_pct: float = Config.DEFAULT_WITHDRAW_PCT,
    ):
        self.zscore_threshold = zscore_threshold
        self.invest_pct = invest_pct
        self.withdraw_pct = withdraw_pct

    def calculate_allocation(
        self,
        stocks_list: List[str],
        cash_available: float,
        stocks_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """Calculate allocation based on mean reversion Z-score."""
        if stocks_weights is None:
            n = len(stocks_list)
            stocks_weights = {s: 1.0 / n for s in stocks_list}

        price_history = DataFetcher.fetch_price_history(
            stocks_list, Config.PRICE_HISTORY_PERIOD, Config.PRICE_HISTORY_INTERVAL
        )

        aligned = price_history[stocks_list].dropna()
        weights = np.array([stocks_weights[s] for s in stocks_list])
        portfolio_values = aligned.values @ weights

        # Calculate Z-score based on last N days
        lookback_data = portfolio_values[-Config.ZSCORE_LOOKBACK_DAYS :]
        mean = lookback_data.mean()
        std = lookback_data.std()

        if std == 0:
            return 0.0, 0.0, 0.0

        z = (portfolio_values[-1] - mean) / std
        print(f"Mean Reversion Z-score: {z:.2f}")

        if z < self.zscore_threshold:
            cash_change = cash_available * self.invest_pct
        elif z > abs(self.zscore_threshold):
            cash_change = -cash_available * self.withdraw_pct
        else:
            cash_change = 0

        print(f"Mean Reversion Z-score based cash settings: {cash_change}")

        cash_to_invest = cash_change if cash_change > 0 else 0
        cash_to_withdraw = -cash_change if cash_change < 0 else 0

        return cash_to_invest, cash_to_withdraw, 0.0


class ManualCashStrategy(CashAllocationStrategy):
    """Manual cash allocation strategy."""

    def __init__(
        self,
        cash_to_invest: float = 0,
        cash_to_withdraw: float = 0,
        cash_pct: float = 0.0,
    ):
        self.cash_to_invest = cash_to_invest
        self.cash_to_withdraw = cash_to_withdraw
        self.cash_pct = cash_pct

    def calculate_allocation(
        self,
        stocks_list: List[str],
        cash_available: float,
        stocks_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """Return manual cash allocation settings."""
        print(
            f"Manual cash settings used: Invest ${self.cash_to_invest}, "
            f"Withdraw ${self.cash_to_withdraw}, Cash % {self.cash_pct:.2%}"
        )
        return self.cash_to_invest, self.cash_to_withdraw, self.cash_pct


class CashAllocationManager:
    """Manages multiple cash allocation strategies."""

    def __init__(
        self,
        use_manual: bool = True,
        use_fear_greed: bool = True,
        use_drawdown: bool = True,
        use_mean_reversion: bool = True,
        manual_settings: Optional[Dict[str, float]] = None,
    ):
        self.use_manual = use_manual
        self.use_fear_greed = use_fear_greed
        self.use_drawdown = use_drawdown
        self.use_mean_reversion = use_mean_reversion

        # Initialize strategies
        self.manual_strategy = ManualCashStrategy(
            manual_settings.get("cash_to_invest", 0) if manual_settings else 0,
            manual_settings.get("cash_to_withdraw", 0) if manual_settings else 0,
            manual_settings.get("cash_pct", 0.0) if manual_settings else 0.0,
        )
        self.fear_greed_strategy = FearGreedStrategy()
        self.drawdown_strategy = DrawdownTriggerStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()

    def determine_cash_allocation(
        self,
        stocks_list: List[str],
        cash_available: float,
        stocks_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Determine cash allocation using the configured strategies.

        Returns:
            Tuple of (cash_to_invest, cash_to_withdraw, cash_pct)
        """
        if self.use_manual:
            return self.manual_strategy.calculate_allocation(
                stocks_list, cash_available, stocks_weights
            )

        if self.use_fear_greed:
            return self.fear_greed_strategy.calculate_allocation(
                stocks_list, cash_available, stocks_weights
            )

        # For drawdown and mean reversion strategies, we need to plot the portfolio
        if self.use_drawdown or self.use_mean_reversion:
            price_history = DataFetcher.fetch_price_history(
                stocks_list, Config.PRICE_HISTORY_PERIOD, Config.PRICE_HISTORY_INTERVAL
            )
            PortfolioVisualizer.plot_portfolio_value(
                price_history, stocks_list, stocks_weights, "Portfolio Value Over Time"
            )

        cash_to_invest = 0.0
        cash_to_withdraw = 0.0
        cash_pct = 0.0

        if self.use_drawdown:
            invest, withdraw, pct = self.drawdown_strategy.calculate_allocation(
                stocks_list, cash_available, stocks_weights
            )
            cash_to_invest = invest
            cash_to_withdraw = withdraw
            cash_pct = pct

        if self.use_mean_reversion:
            invest, withdraw, pct = self.mean_reversion_strategy.calculate_allocation(
                stocks_list, cash_available, stocks_weights
            )
            cash_to_invest = invest
            cash_to_withdraw = withdraw
            cash_pct = pct

        return cash_to_invest, cash_to_withdraw, cash_pct


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================


class PortfolioOptimizer:
    """Handles portfolio optimization methods."""

    @staticmethod
    def equal_weight_optimization(stocks_list: List[str]) -> Dict[str, float]:
        """
        Calculate equal weights for all stocks.

        Args:
            stocks_list: List of stock symbols

        Returns:
            Dictionary of stock weights
        """
        equal_weight = 1.0 / len(stocks_list)
        return {stock: equal_weight for stock in stocks_list}

    @staticmethod
    def hierarchical_risk_parity_optimization(
        stocks_list: List[str],
    ) -> Dict[str, float]:
        """
        Calculate weights using Hierarchical Risk Parity optimization.

        Args:
            stocks_list: List of stock symbols

        Returns:
            Dictionary of stock weights
        """
        # Fetch historical data
        data = yf.download(
            stocks_list,
            period=Config.HRP_DATA_PERIOD,
            interval=Config.HRP_DATA_INTERVAL,
            progress=False,
        )
        daily_prices = data["Close"]
        daily_returns = daily_prices.pct_change(fill_method=None).dropna()

        # Estimate covariance matrix
        S = CovarianceShrinkage(daily_prices).ledoit_wolf()

        # Hierarchical Risk Parity Portfolio
        hrp = HRPOpt(daily_returns, S)
        hrp.optimize()
        hrp_weights = hrp.clean_weights()

        return hrp_weights


# =============================================================================
# PORTFOLIO DATA MANAGEMENT
# =============================================================================


class PortfolioDataManager:
    """Handles portfolio data loading and processing."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df_positions = None
        self.df = None
        self.cash_available = 0.0

    def load_portfolio_data(self) -> bool:
        """
        Load portfolio data from CSV file.

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.file_path):
            print(f"Error: File '{self.file_path}' does not exist.")
            return False

        try:
            self.df_positions = pd.read_csv(self.file_path)[
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
            self.df_positions = self.df_positions.dropna(subset=["Symbol"])
            return True
        except Exception as e:
            print(f"Error loading portfolio data: {e}")
            return False

    def process_portfolio_data(self, stocks_list: List[str]) -> bool:
        """
        Process and clean portfolio data.

        Args:
            stocks_list: List of all stocks to include

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean and convert numeric columns
            self.df_positions["Quantity"] = pd.to_numeric(
                self.df_positions["Quantity"], errors="coerce"
            )
            self.df_positions["Current Value"] = pd.to_numeric(
                self.df_positions["Current Value"].replace(r"[\$,]", "", regex=True),
                errors="coerce",
            )
            self.df_positions["Cost Basis Total"] = pd.to_numeric(
                self.df_positions["Cost Basis Total"].replace(r"[\$,]", "", regex=True),
                errors="coerce",
            )
            self.df_positions["Total Gain/Loss Dollar"] = pd.to_numeric(
                self.df_positions["Total Gain/Loss Dollar"].replace(
                    r"[\$,]", "", regex=True
                ),
                errors="coerce",
            )

            # Extract cash position
            self._extract_cash_position()

            # Build full target DataFrame including missing tickers
            self.df = pd.DataFrame({"Symbol": stocks_list})
            self.df = self.df.merge(self.df_positions, on="Symbol", how="left").fillna(
                {"Description": "", "Quantity": 0, "Current Value": 0.0}
            )

            # Compute current weights
            current_value_total = self.df["Current Value"].sum()
            self.df["current_weight"] = self.df["Current Value"] / (
                current_value_total if current_value_total > 0 else np.nan
            )

            return True
        except Exception as e:
            print(f"Error processing portfolio data: {e}")
            return False

    def _extract_cash_position(self) -> None:
        """Extract cash available from cash symbols."""
        self.cash_available = 0.0
        for cash_symbol in Config.CASH_SYMBOLS:
            if cash_symbol in self.df_positions["Symbol"].values:
                cash_value = self.df_positions.loc[
                    self.df_positions["Symbol"] == cash_symbol, "Current Value"
                ].iloc[0]
                self.cash_available += cash_value

    def update_prices(self) -> bool:
        """
        Update current prices for all stocks.

        Returns:
            True if successful, False otherwise
        """
        try:
            failed_downloads = 0
            prices = []

            for index, row in self.df.iterrows():
                fallback_price = float(
                    row["Last Price"].replace("$", "").replace(",", "")
                )
                price, success = DataFetcher.fetch_current_price(
                    row["Symbol"], fallback_price
                )
                prices.append(price)
                if not success:
                    failed_downloads += 1

            self.df["price"] = prices

            if failed_downloads >= 1:
                print("Too many failed downloads. Stopping execution.")
                return False

            # Update Current Value based on fetched prices
            self.df["Current Value"] = self.df["Quantity"] * self.df["price"]

            # Recompute current weights
            current_value_total = self.df["Current Value"].sum()
            self.df["current_weight"] = self.df["Current Value"] / (
                current_value_total if current_value_total > 0 else np.nan
            )

            return True
        except Exception as e:
            print(f"Error updating prices: {e}")
            return False

    def get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        return dict(zip(self.df["Symbol"], self.df["current_weight"]))

    def get_total_value(self) -> float:
        """Get total portfolio value."""
        return self.df["Current Value"].sum()


# =============================================================================
# PORTFOLIO REBALANCER
# =============================================================================


class PortfolioRebalancer:
    """Main portfolio rebalancing engine."""

    def __init__(
        self,
        optimization_method: str = "Equal Weight",
        rebalancing_mode: bool = True,
        cash_allocation_manager: Optional[CashAllocationManager] = None,
    ):
        self.optimization_method = optimization_method
        self.rebalancing_mode = rebalancing_mode
        self.cash_allocation_manager = (
            cash_allocation_manager or CashAllocationManager()
        )

    def rebalance_portfolio(
        self, data_manager: PortfolioDataManager, stocks_list: List[str]
    ) -> pd.DataFrame:
        """
        Perform portfolio rebalancing.

        Args:
            data_manager: PortfolioDataManager instance
            stocks_list: List of stock symbols

        Returns:
            DataFrame with rebalancing recommendations
        """
        # Determine cash allocations
        current_weights = data_manager.get_current_weights()
        cash_to_invest, cash_to_withdraw, cash_pct = (
            self.cash_allocation_manager.determine_cash_allocation(
                stocks_list, data_manager.cash_available, current_weights
            )
        )

        # Calculate target weights
        if self.optimization_method == "Equal Weight":
            target_weights = PortfolioOptimizer.equal_weight_optimization(stocks_list)
        elif self.optimization_method == "Hierarchical Risk Parity":
            target_weights = PortfolioOptimizer.hierarchical_risk_parity_optimization(
                stocks_list
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

        # Assign target weights to DataFrame
        data_manager.df["target_weight"] = data_manager.df["Symbol"].apply(
            lambda x: target_weights.get(x, 0)
        )

        # Calculate rebalancing trades
        self._calculate_trades(data_manager, cash_to_invest, cash_to_withdraw, cash_pct)

        return data_manager.df

    def _calculate_trades(
        self,
        data_manager: PortfolioDataManager,
        cash_to_invest: float,
        cash_to_withdraw: float,
        cash_pct: float,
    ) -> None:
        """Calculate trade recommendations."""
        current_value_total = data_manager.get_total_value()
        cash_adjustment = cash_to_invest - cash_to_withdraw
        new_total_value = current_value_total + cash_adjustment

        # Target dollar value per ticker on the adjusted total
        data_manager.df["target_value"] = (
            data_manager.df["target_weight"] * new_total_value
        )

        # Calculate raw trade dollars
        raw_trade = (
            data_manager.df["target_weight"] * new_total_value
            - data_manager.df["Current Value"]
        )

        # Calculate target cash
        total_portfolio_value = current_value_total + data_manager.cash_available
        target_cash = cash_pct * total_portfolio_value

        # Calculate trades based on mode and cash adjustment
        if self.rebalancing_mode:
            # Rebalance: ignore cash flows, trade to target weights
            data_manager.df["trade_value"] = (
                data_manager.df["target_value"] - data_manager.df["Current Value"]
            )
        elif cash_adjustment > 0:
            # Only allocate buys
            self._calculate_buy_trades(
                data_manager, raw_trade, cash_adjustment, target_cash
            )
        elif cash_adjustment < 0:
            # Only allocate sells
            self._calculate_sell_trades(
                data_manager, raw_trade, cash_adjustment, target_cash
            )
        else:
            # No cash change → no trades
            data_manager.df["trade_value"] = 0

        # Convert to shares
        data_manager.df["shares_to_trade"] = (
            data_manager.df["trade_value"] / data_manager.df["price"]
        ).round(2)
        data_manager.df["shares_to_buy"] = data_manager.df["shares_to_trade"].clip(
            lower=0
        )
        data_manager.df["shares_to_sell"] = (-data_manager.df["shares_to_trade"]).clip(
            lower=0
        )

    def _calculate_buy_trades(
        self,
        data_manager: PortfolioDataManager,
        raw_trade: pd.Series,
        cash_adjustment: float,
        target_cash: float,
    ) -> None:
        """Calculate buy trades only."""
        buys = raw_trade.clip(lower=0)
        available_for_buy = min(
            cash_adjustment, max(0, data_manager.cash_available - target_cash)
        )

        if buys.sum() > 0 and available_for_buy > 0:
            data_manager.df["trade_value"] = buys * (available_for_buy / buys.sum())
        else:
            data_manager.df["trade_value"] = 0

    def _calculate_sell_trades(
        self,
        data_manager: PortfolioDataManager,
        raw_trade: pd.Series,
        cash_adjustment: float,
        target_cash: float,
    ) -> None:
        """Calculate sell trades only."""
        data_manager.df["profit"] = (
            data_manager.df["Current Value"] - data_manager.df["Cost Basis Total"]
        )
        sell_mask = (raw_trade < 0) & (data_manager.df["profit"] > 0)
        sells = raw_trade.where(sell_mask, 0)

        # Calculate max cash we can add from selling
        amount_to_raise_by_selling = abs(cash_adjustment) + max(
            0, target_cash - data_manager.cash_available
        )
        allowed_sell = amount_to_raise_by_selling

        total_sells = -sells.sum() if sells.sum() < 0 else 0

        if total_sells > 0 and allowed_sell > 0:
            data_manager.df["trade_value"] = sells * (allowed_sell / total_sells)
        elif sells.sum() != 0:
            data_manager.df["trade_value"] = sells * (
                abs(cash_adjustment) / total_sells
            )
        else:
            data_manager.df["trade_value"] = 0


# =============================================================================
# REPORTING AND OUTPUT
# =============================================================================


class PortfolioReporter:
    """Handles portfolio reporting and output generation."""

    @staticmethod
    def print_summary(
        data_manager: PortfolioDataManager,
        optimization_method: str,
        rebalancing_mode: bool,
        cash_to_invest: float,
        cash_to_withdraw: float,
        cash_pct: float,
    ) -> None:
        """Print portfolio summary."""
        current_value_total = data_manager.get_total_value()
        total_portfolio_value = current_value_total + data_manager.cash_available
        target_cash = cash_pct * total_portfolio_value

        print("=" * 50)
        print("Optimization method:", optimization_method)
        print("Rebalancing mode:", rebalancing_mode)
        print(f"Total portfolio + cash value:     ${total_portfolio_value:,.2f}")

        total_gain_loss = data_manager.df["Total Gain/Loss Dollar"].sum()
        gain_loss_pct = (
            (total_gain_loss / total_portfolio_value)
            if total_portfolio_value > 0
            else np.nan
        )
        print(
            f"Total Gain/Loss value:            ${total_gain_loss:,.2f} ({gain_loss_pct:.2%})"
        )

        cash_pct_display = data_manager.cash_available / total_portfolio_value
        print(
            f"Cash available (FDRXX** & SPSK):  ${data_manager.cash_available:,.2f} ({cash_pct_display:.2%})"
        )
        print(f"Target cash ({cash_pct:.2%}):              ${target_cash:,.2f}")
        print(f"Total cash to invest:             ${cash_to_invest:,.2f}")
        print(f"Total cash to withdraw:           ${cash_to_withdraw:,.2f}")
        print("=" * 50)

    @staticmethod
    def save_results(
        df: pd.DataFrame, optimization_method: str, date: str, output_dir: str = "files"
    ) -> str:
        """
        Save rebalancing results to CSV.

        Args:
            df: DataFrame with rebalancing results
            optimization_method: Optimization method used
            date: Date string for filename
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(
            output_dir,
            f"Portfolio_Rebalance_{optimization_method}_{date}_{timestamp}.csv",
        )

        # Sort by shares_to_sell descending, then shares_to_buy descending
        df_sorted = df.sort_values(
            by=["shares_to_sell", "shares_to_buy"], ascending=[False, False]
        )
        df_sorted.to_csv(output_filename, index=False)

        return output_filename


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main execution function."""
    # === User Configuration ===
    date = "Sep-04-2025"
    optimization_method = "Hierarchical Risk Parity"  # Options: "Equal Weight", "Hierarchical Risk Parity"
    rebalancing_mode = True  # Set to True to rebalance by selling shares

    # Cash allocation settings
    use_manual_cash_settings = True
    use_fgn = True
    use_drawdown_trigger = True
    use_mean_reversion_zscore = True

    # Manual cash settings (used when use_manual_cash_settings = True)
    manual_cash_to_invest = 0
    manual_cash_to_withdraw = 0
    manual_cash_pct = 0.0

    # === Initialize Components ===
    stocks_list = Config.DEFAULT_STOCKS

    # Load portfolio data
    file_path = os.path.join("files", f"Portfolio_Positions_{date}.csv")
    data_manager = PortfolioDataManager(file_path)

    if not data_manager.load_portfolio_data():
        return

    if not data_manager.process_portfolio_data(stocks_list):
        return

    # Update prices
    if not data_manager.update_prices():
        return

    # Initialize cash allocation manager
    manual_settings = {
        "cash_to_invest": manual_cash_to_invest,
        "cash_to_withdraw": manual_cash_to_withdraw,
        "cash_pct": manual_cash_pct,
    }

    cash_allocation_manager = CashAllocationManager(
        use_manual=use_manual_cash_settings,
        use_fear_greed=use_fgn,
        use_drawdown=use_drawdown_trigger,
        use_mean_reversion=use_mean_reversion_zscore,
        manual_settings=manual_settings if use_manual_cash_settings else None,
    )

    # Initialize rebalancer
    rebalancer = PortfolioRebalancer(
        optimization_method=optimization_method,
        rebalancing_mode=rebalancing_mode,
        cash_allocation_manager=cash_allocation_manager,
    )

    # Perform rebalancing
    result_df = rebalancer.rebalance_portfolio(data_manager, stocks_list)

    # Get cash allocation for reporting
    current_weights = data_manager.get_current_weights()
    cash_to_invest, cash_to_withdraw, cash_pct = (
        cash_allocation_manager.determine_cash_allocation(
            stocks_list, data_manager.cash_available, current_weights
        )
    )

    # Print summary and save results
    PortfolioReporter.print_summary(
        data_manager,
        optimization_method,
        rebalancing_mode,
        cash_to_invest,
        cash_to_withdraw,
        cash_pct,
    )

    output_file = PortfolioReporter.save_results(result_df, optimization_method, date)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

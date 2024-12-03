import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class AssetAssumption:
    """
    AssetAssumption is a utility class for calculating financial assumptions necessary
    for portfolio optimization. The class provides methods for computing expected returns, 
    CAPM-based expected returns, and covariance matrices using historical price data.

    Attributes:
        returns_window (int): Rolling window size for expected return calculations.
        covariance_window (int): Rolling window size for covariance matrix calculations.

    Methods:
        calculate_expected_return(price_data):
            Calculates the expected returns for each asset using a rolling window.

        calculate_capm_expected_return(price_data, risk_free_rate):
            Computes expected returns for each asset based on the Capital Asset Pricing Model (CAPM).

        calculate_covariance(price_data):
            Computes the covariance matrix for asset returns using historical data.
    """

    def __init__(self, returns_window: int = 52, covariance_window: int = 52):
        self.returns_window = returns_window
        self.covariance_window = covariance_window

    def calculate_expected_return(self, price_data: pd.DataFrame) -> pd.Series:

        weekly_prices = price_data.resample('W').last()
        weekly_returns = weekly_prices.pct_change().astype(np.float32)

        weekly_returns = weekly_returns.to_numpy()

        if weekly_returns.shape[0] == 0 or weekly_returns.shape[0] < self.returns_window:
            if np.all(np.isnan(weekly_returns)):
                expected_returns = np.full(price_data.shape[1], -99999)
            else:
                expected_returns = np.nanmean(weekly_returns, axis=0)

            expected_returns = np.where(np.isnan(expected_returns), -99999, expected_returns)
            return pd.Series(expected_returns, index=price_data.columns)

        rolling_means = np.lib.stride_tricks.sliding_window_view(
            weekly_returns, self.returns_window, axis=0
        ).mean(axis=2)
        expected_returns = rolling_means[-1]

        valid_data_counts = np.sum(~np.isnan(weekly_returns), axis=0)
        fallback_means = []
        for col_idx in range(weekly_returns.shape[1]):
            if valid_data_counts[col_idx] == 0:
                fallback_means.append(-99999)
            else:
                fallback_means.append(np.nanmean(weekly_returns[:, col_idx]))
        fallback_means = np.array(fallback_means)

        final_returns = np.where(valid_data_counts >= self.returns_window, expected_returns, fallback_means)
        final_returns = np.nan_to_num(final_returns, nan=-99999)

        return pd.Series(final_returns, index=price_data.columns)

    def calculate_capm_expected_return(self, price_data: pd.DataFrame, risk_free_rate: float = 0.01) -> pd.Series:
        weekly_prices = price_data.resample('W').last()
        weekly_returns = weekly_prices.pct_change().astype(np.float32)

        market_returns = price_data.iloc[:, 0].resample('W').last().pct_change().astype(np.float32)

        betas = []
        for col in weekly_returns.columns:
            valid_data = weekly_returns[col].dropna()
            valid_market = market_returns.loc[valid_data.index].dropna()

            if len(valid_market) < 2:
                betas.append(-99999)
                continue

            beta = np.cov(valid_data, valid_market)[0, 1] / np.var(valid_market)
            betas.append(beta)

        betas = np.array(betas)
        capm_returns = risk_free_rate + betas * (market_returns.mean() - risk_free_rate)

        capm_returns = np.nan_to_num(capm_returns, nan=-99999)
        return pd.Series(capm_returns, index=price_data.columns)

    def calculate_covariance(self, price_data: pd.DataFrame) -> pd.DataFrame:
        weekly_prices = price_data.resample('W').last()
        weekly_returns = weekly_prices.pct_change().astype(np.float32)

        weekly_returns = weekly_returns.to_numpy()

        num_assets = weekly_returns.shape[1]
        covariance_matrix = np.zeros((num_assets, num_assets))

        for i in range(num_assets):
            for j in range(i, num_assets):
                valid_data = weekly_returns[:, [i, j]]
                valid_data = valid_data[~np.isnan(valid_data).any(axis=1)]
                if len(valid_data) >= self.covariance_window:
                    cov_value = np.cov(valid_data[-self.covariance_window:], rowvar=False)[0, 1]
                elif len(valid_data) > 1:
                    cov_value = np.cov(valid_data, rowvar=False)[0, 1]
                else:
                    cov_value = 0
                covariance_matrix[i, j] = cov_value
                if i != j:
                    covariance_matrix[j, i] = cov_value

        return pd.DataFrame(covariance_matrix, index=price_data.columns, columns=price_data.columns)

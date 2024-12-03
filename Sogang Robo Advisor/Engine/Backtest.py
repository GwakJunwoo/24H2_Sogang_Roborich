import matplotlib.pyplot as plt
from .Evaluation import show
from tqdm import tqdm
import pandas as pd
import numpy as np


class Backtest:
    """
    The Backtest class simulates portfolio performance over a specified time frame
    with periodic rebalancing and tracks key performance metrics.

    Attributes:
        pipeline: The Pipeline object that defines the optimization process.
        price_data (pd.DataFrame): Historical price data used for backtesting.
        rebalance_dates (list): Dates on which portfolio rebalancing occurs.
        trading_days (list): List of valid trading days to adjust rebalance dates if needed.
        allocations (list): Stores portfolio allocations at each rebalance date.
        portfolio_values (pd.DataFrame): Tracks the portfolio value over time.

    Methods:
        get_previous_trading_day(date):
            Returns the nearest trading day before or equal to the given date.

        handle_missing_data(data, current_date):
            Handles missing price data by forward-filling and filtering up to the current date.

        rebalance(current_date):
            Performs portfolio rebalancing using the pipeline and returns the allocation.

        run_backtest(initial_value):
            Runs the backtest simulation over the defined rebalance dates.

        calculate_performance():
            Calculates key performance metrics: cumulative return, maximum drawdown, and Sharpe ratio.

        evaluation(allocation_dict):
            Compares the portfolio performance with a benchmark and returns evaluation metrics.

        visualize_performance():
            Generates plots for cumulative returns and maximum drawdown over the backtest period.
    """

    def __init__(self, pipeline, price_data, rebalance_dates, trading_days):
        self.pipeline = pipeline
        self.price_data = price_data.ffill()
        self.rebalance_dates = rebalance_dates
        self.trading_days = trading_days
        self.allocations = []
        self.portfolio_values = pd.DataFrame(index=price_data.loc[str(rebalance_dates[0]):].index,
                                             columns=["Portfolio Value"])

    def get_previous_trading_day(self, date):
        return max([d for d in self.trading_days if d <= date])

    def handle_missing_data(self, data, current_date):
        return data.ffill().loc[:current_date]

    def rebalance(self, current_date):
        prev_trading_day = self.get_previous_trading_day(current_date)
        price_data_until_now = self.price_data.loc[:prev_trading_day]

        clean_price_data = self.handle_missing_data(price_data_until_now, prev_trading_day)

        allocation = self.pipeline.run(clean_price_data)

        leaf_nodes = self.pipeline.universe.get_leaf_nodes()

        final_allocation = {k: v for k, v in allocation.items() if k in leaf_nodes}
        return final_allocation

    def run_backtest(self, initial_value=1000000):
        current_value = initial_value
        for i, date in tqdm(enumerate(self.rebalance_dates)):
            allocation = self.rebalance(date)

            self.allocations.append((date, allocation))

            if i < len(self.rebalance_dates) - 1:
                next_rebalance_date = self.rebalance_dates[i + 1]
            else:
                next_rebalance_date = self.price_data.index[-1]

            asset_returns = self.price_data.loc[date:next_rebalance_date].pct_change()
            for day in asset_returns.index:
                valid_allocation = {k: v for k, v in allocation.items() if v > 0}
                valid_asset_returns = asset_returns.loc[day][valid_allocation.keys()]

                valid_asset_returns = valid_asset_returns.dropna()
                valid_allocation_series = pd.Series(valid_allocation).loc[valid_asset_returns.index]

                day_return = sum(valid_asset_returns * valid_allocation_series)
                current_value *= (1 + day_return)
                self.portfolio_values.loc[day] = current_value
        self.portfolio_values.iloc[0] = initial_value

    def calculate_performance(self):
        cumulative_return = self.portfolio_values["Portfolio Value"].iloc[-1] / \
                            self.portfolio_values["Portfolio Value"].iloc[0] - 1

        running_max = self.portfolio_values["Portfolio Value"].cummax()
        drawdown = (self.portfolio_values["Portfolio Value"] - running_max) / running_max
        mdd = drawdown.min()

        daily_return = self.portfolio_values["Portfolio Value"].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_return.mean() / daily_return.std()

        return cumulative_return, mdd, sharpe_ratio

    def evaluation(self, allocation_dict):
        model = self.portfolio_values['Portfolio Value'].astype(np.int64) / 100
        bench = self.price_data[['069500']].squeeze()

        model = model.pct_change().dropna()
        bench = bench.pct_change().dropna()

        common_index = model.index.intersection(bench.index)

        model_filtered = model.loc[common_index]
        bench_filtered = bench.loc[common_index]

        evaluation_target = pd.DataFrame({
            'Benchmark': bench_filtered,
            'Model': model_filtered
        })

        evaluation_metrics = show(allocation_dict, evaluation_target)
        return evaluation_metrics

    def visualize_performance(self):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        self.portfolio_values["Cumulative Return"] = self.portfolio_values["Portfolio Value"] / \
                                                     self.portfolio_values["Portfolio Value"].iloc[0]
        ax[0].plot(self.portfolio_values.index, self.portfolio_values["Cumulative Return"], label="Cumulative Return")
        ax[0].set_title("Cumulative Return")
        ax[0].set_ylabel("Return")
        ax[0].legend()

        running_max = self.portfolio_values["Portfolio Value"].cummax()
        drawdown = (self.portfolio_values["Portfolio Value"] - running_max) / running_max
        ax[1].plot(self.portfolio_values.index, drawdown, label="Drawdown", color='red')
        ax[1].set_title("Maximum Drawdown (MDD)")
        ax[1].set_ylabel("Drawdown")
        ax[1].legend()

        plt.tight_layout()
        plt.show()

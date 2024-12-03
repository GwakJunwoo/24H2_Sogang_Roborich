![Sogang Robo Logo](Sogang%20Robo%20Advisor/Logo/sogang-robo-logo-professional.svg)

**SGRA(Sogang Robo Advisor)** is a Python-based portfolio optimization and backtesting engine designed to support
hierarchical asset allocation strategies and various optimization techniques. The tool provides flexibility in
integrating multiple optimizers, calculating key performance metrics, and visualizing portfolio evaluations.

By combining modules for **assumption modeling**, **optimization**, and **backtesting**, SGRA enables seamless
end-to-end workflows for portfolio construction and evaluation:

- Use the `AssetAssumption` module to calculate expected returns and covariance matrices from historical price data.
- Define a hierarchical asset tree with the `Tree` class and apply modular optimizers to different levels of the
  hierarchy.
- Execute dynamic rebalancing and evaluate portfolio performance over time with the `Backtest` engine.
- Compare strategy results with benchmarks and visualize outcomes for deeper insights.

This integrated approach simplifies portfolio management tasks, providing both flexibility and robust analytics in a
single framework.

## Dependencies

```shell
pip install numpy pandas matplotlib plotly tqdm cvxpy pillow finance-datareader
```

## Features

### 1. **Hierarchical Asset Allocation**

- **Strategic Asset Allocation (SAA)**:
    - Long-term allocation focusing on maintaining the overall portfolio's target risk and return characteristics.
    - Optimizes high-level asset classes (e.g., equities, bonds, alternatives) to reflect strategic objectives.
- **Tactical Asset Allocation (TAA)**:
    - Shorter-term allocation to take advantage of market inefficiencies or opportunities.
    - Dynamically adjusts weights within sub-asset classes while adhering to the constraints set by SAA.

- **Building Block Approach**:
    - **Tree-Based Hierarchical Optimization**:
        - Assets are structured hierarchically using parent-child relationships.
        - Each level of the hierarchy can apply a different optimization method.
    - **Modular Optimizers**:
        - Multiple optimization techniques can be applied at different levels of the tree.
        - Supports integration of custom optimizers for specific strategies.
    - For more detailed information, refer to the following resource:
        - [KB Securities Research Report](https://rdata.kbsec.com/pdf_data/20220103100504197K.pdf)

### 2. **Assumption Modeling**

The `AssetAssumption` class calculates:

- **Expected Returns**:
    - Simple historical expected returns based on a rolling window.
    - Expected returns using the **CAPM (Capital Asset Pricing Model)**.
- **Covariance Matrix**:
    - Asset return covariances calculated from historical data.
    - Supports rolling window calculations to focus on recent data trends.

### 3. **Supported Optimizers**

1. **Mean-Variance Optimizer**:
    - Balances risk and return using covariance matrices.
    - Requires expected returns and covariance matrix as inputs.
2. **Equal Weight Optimizer**:
    - Assigns equal weights to all assets within the group.
3. **Dynamic Risk Optimizer**:
    - Allocates weights dynamically based on risk tolerance and investment horizon.
4. **Risk Parity Optimizer**:
    - Balances risk contribution equally among assets.
5. **Goal-Based Optimizer**:
    - Focuses on achieving specific investment goals using Monte Carlo simulations.

### 4. **Backtesting**

The `Backtest` class simulates portfolio performance over a specified time frame. It integrates with the `Pipeline`
class to dynamically rebalance portfolios and evaluate performance metrics.

- **Dynamic Rebalancing**:
    - Rebalances the portfolio at specified dates based on optimization outputs from the `Pipeline`.
    - Handles missing or incomplete data by forward-filling values to ensure continuity in calculations.
    - Tracks portfolio value changes over time, allowing for detailed performance evaluation.

### 5. **Performance Evaluation**

- Calculates comprehensive investment metrics, including:
    - **Cumulative Return**: Total return over the evaluation period.
    - **CAGR** (Compound Annual Growth Rate): Annualized portfolio growth rate.
    - **Sharpe Ratio**: Risk-adjusted return measurement.
    - **Sortino Ratio**: Focused risk-adjusted return using downside risk.
    - **Max Drawdown**: Largest peak-to-trough decline during the evaluation period.
    - **Annualized Volatility**: Yearly volatility of returns.
    - **Calmar Ratio**: Return-to-risk ratio using maximum drawdown.
    - **Skewness**: Asymmetry of return distribution.
    - **Kurtosis**: "Fat-tailedness" of the return distribution.
    - **Expected Daily/Monthly/Yearly Returns**: Anticipated return values over different time horizons.
    - **Kelly Criterion**: Optimal betting fraction for reinvestment.
    - **VaR (Value at Risk)**: Expected loss under adverse market conditions.
    - **CVaR (Conditional VaR)**: Expected loss beyond the VaR threshold.

## Investment Types and Goals

| **Risk Level**    | **Code** | **Investment Goal**      | **Code** |
|-------------------|----------|--------------------------|----------|
| Conservative      | 1        | Marriage Fund Planning   | 1        |
| Risk-Averse       | 2        | Retirement Fund Planning | 2        |
| Risk-Neutral      | 3        | Long-Term Wealth Growth  | 3        |
| Aggressive        | 4        | Saving for a Large Goal  | 4        |
| Highly Aggressive | 5        |                          |          |

## Goal-Specific Strategies

| Investment Goal              | Code | Characteristics                                                         | Methodology     |
|------------------------------|------|-------------------------------------------------------------------------|-----------------|
| **Marriage Fund Planning**   | 1    | Balances stability and returns for medium-term asset accumulation.      | DRA → MVO → MVO |
| **Retirement Fund Planning** | 2    | Aims for long-term stability with pension-like returns.                 | RPO → GBI → MVO |
| **Long-Term Wealth Growth**  | 3    | Targets high returns over the long term while managing volatility.      | RPO → MVO → MVO |
| **Saving for a Large Goal**  | 4    | Seeks stable returns and risk management over the short to medium term. | DRA → GBI → MVO |

### Abbreviations

- **RPO**: Risk Parity Optimization
- **MVO**: Mean-Variance Optimization
- **DRA**: Dynamic Risk Allocation
- **GBI**: Goal-Based Investing

## Installation and Usage

1. **Install Dependencies**:
   Run the following to install required libraries:
   ```bash
   pip install numpy pandas matplotlib cvxpy tqdm

2. **Usage Example**:
   Run the following Example:
   ```bash
   main(codes=['069500','139260','161510','273130','439870','251340','114260'], risk_level=5, investor_goal=4)

## Project Origin

This project was developed as part of the **Big Data Capstone Design(BDS4010)** course at Sogang University.  
For more details and updates on the project, please refer to the
related [Sogang Wiki](http://cscp2.sogang.ac.kr/BDS4010/index.php/3%ED%8C%80:_%EB%A1%9C%EB%B3%B4%EB%A6%AC%EC%B9%98).

## License

This project is licensed under the **MIT License**. For more details, please refer to the LICENSE file.

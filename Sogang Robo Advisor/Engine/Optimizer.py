from typing import Tuple, Union

from .BaseOptimizer import *

"""
This module provides a collection of portfolio optimization techniques 
designed to allocate asset weights based on various objectives and constraints.

Functions:
    portfolio_variance(weights, covariance_matrix):
        Calculates the portfolio variance given weights and a covariance matrix.

    mean_return(weights, expected_returns):
        Calculates the weighted mean return of a portfolio.

    is_positive_semidefinite(matrix):
        Checks if a given matrix is positive semi-definite.

    make_positive_semidefinite(matrix):
        Adjusts a matrix to make it positive semi-definite by modifying eigenvalues.

    mean_variance_optimizer(nodes, covariance_matrix, expected_returns, weight_bounds, risk_aversion):
        Optimizes weights to balance risk and return based on the mean-variance framework.

    equal_weight_optimizer(nodes, weight_bounds):
        Assigns equal weights to all assets in the portfolio.

    dynamic_risk_optimizer(nodes, covariance_matrix, risk_tolerance, goal_period, weight_bounds):
        Dynamically allocates weights based on risk tolerance and investment horizon.

    risk_parity_optimizer(nodes, covariance_matrix, risk_aversion, weight_bounds):
        Allocates weights to achieve risk parity among all assets.

    goal_based_optimizer(nodes, covariance_matrix, expected_returns, weight_bounds, risk_aversion, goal_amount, goal_period, simulations):
        Optimizes portfolio weights to achieve a specific financial goal using Monte Carlo simulations.
"""


def portfolio_variance(weights, covariance_matrix):
    return cp.quad_form(weights, cp.Constant(covariance_matrix))


def mean_return(weights, expected_returns):
    return cp.matmul(weights, expected_returns)


def is_positive_semidefinite(matrix: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(matrix) >= 0)


def make_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    min_eigenvalue = np.min(np.linalg.eigvals(matrix))
    if min_eigenvalue < 0:
        matrix += np.eye(matrix.shape[0]) * (-min_eigenvalue + 1e-6)
    return matrix


def mean_variance_optimizer(
        nodes: List[Any],
        covariance_matrix: np.ndarray,
        expected_returns: np.ndarray,
        weight_bounds: Union[List[Tuple], Tuple] = (0, 1),
        risk_aversion: float = 0.5
) -> List[float]:
    def mean_variance_objective(w, cov_matrix, exp_returns, risk_aversion):
        return risk_aversion * portfolio_variance(w, cov_matrix) - mean_return(w, exp_returns)

    n_assets = len(nodes)
    tickers = [node.name for node in nodes]

    if covariance_matrix.shape[0] != n_assets or covariance_matrix.shape[1] != n_assets:
        raise ValueError(
            f"Covariance matrix dimensions {covariance_matrix.shape} do not match the number of assets ({n_assets}).")

    if not is_positive_semidefinite(covariance_matrix):
        covariance_matrix = make_positive_semidefinite(covariance_matrix)

    if n_assets == 1:
        return [1.0]
    optimizer = BaseConvexOptimizer(n_assets, tickers=tickers, weight_bounds=weight_bounds)
    optimizer.convex_objective(
        lambda w: mean_variance_objective(w, covariance_matrix, expected_returns, risk_aversion),
        weights_sum_to_one=True
    )

    return list(optimizer.clean_weights().values())


def equal_weight_optimizer(
        nodes: List[Any],
        weight_bounds: Union[List[Tuple], Tuple] = (0, 1)
) -> List[float]:
    n = len(nodes)
    if n == 0:
        return []
    return [1.0 / n] * n


def dynamic_risk_optimizer(
        nodes: List[Any],
        covariance_matrix: np.ndarray,
        risk_tolerance: float = 0.5,
        goal_period: int = 10,
        weight_bounds: Union[List[Tuple], Tuple] = (0, 1)
) -> List[float]:
    n_assets = len(nodes)
    if n_assets == 0:
        return []

    stability_factor = goal_period / 10
    adjusted_volatility = np.diag(covariance_matrix) ** (risk_tolerance / stability_factor)

    inverse_volatility = 1 / adjusted_volatility
    weights = inverse_volatility / np.sum(inverse_volatility)

    return list(weights)


def risk_parity_optimizer(
        nodes: List[Any],
        covariance_matrix: np.ndarray,
        risk_aversion: float = 0.5,
        weight_bounds: Union[List[Tuple], Tuple] = (0, 1)
) -> List[float]:
    n_assets = len(nodes)
    if n_assets == 0:
        return []

    volatilities = np.sqrt(np.diag(covariance_matrix))
    adjusted_volatility = volatilities ** risk_aversion
    inverse_volatility = 1 / adjusted_volatility
    weights = inverse_volatility / np.sum(inverse_volatility)

    return list(weights)


def goal_based_optimizer(
        nodes: List[Any],
        covariance_matrix: np.ndarray,
        expected_returns: np.ndarray,
        weight_bounds: Union[List[Tuple], Tuple] = (0, 1),
        risk_aversion: float = 0.5,
        goal_amount: float = 1000000,
        goal_period: int = 10,
        simulations: int = 1000,
) -> List[float]:
    n_assets = len(nodes)
    tickers = [node.name for node in nodes]

    if covariance_matrix.shape[0] != n_assets or covariance_matrix.shape[1] != n_assets:
        raise ValueError(
            f"Covariance matrix dimensions {covariance_matrix.shape} do not match the number of assets ({n_assets}).")

    if not is_positive_semidefinite(covariance_matrix):
        covariance_matrix = make_positive_semidefinite(covariance_matrix)

    if n_assets == 1:
        return [1.0]

    np.random.seed(42)
    ending_values = []
    for _ in range(simulations):
        weights = np.random.dirichlet(np.ones(n_assets), size=1).flatten()
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        yearly_returns = np.random.normal(portfolio_return, portfolio_volatility, goal_period)
        ending_value = np.prod(1 + yearly_returns) * goal_amount
        ending_values.append(ending_value)

    failure_prob = np.mean(np.array(ending_values) < goal_amount)
    success_prob = 1 - failure_prob

    adjusted_risk_aversion = risk_aversion * (1 + success_prob)

    def gbi_objective(w, cov_matrix, exp_returns, adjusted_risk_aversion):
        portfolio_var = portfolio_variance(w, cov_matrix)
        portfolio_ret = mean_return(w, exp_returns)
        return adjusted_risk_aversion * portfolio_var - portfolio_ret

    optimizer = BaseConvexOptimizer(n_assets, tickers=tickers, weight_bounds=weight_bounds)
    optimizer.convex_objective(
        lambda w: gbi_objective(w, covariance_matrix, expected_returns, adjusted_risk_aversion),
        weights_sum_to_one=True
    )

    return list(optimizer.clean_weights().values())

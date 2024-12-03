from typing import Any, List, Callable, Dict
import numpy as np
import cvxpy as cp
import collections


class BaseOptimizer:
    """
    BaseOptimizer is a foundational class for portfolio optimization. It provides 
    basic functionalities for managing asset weights and generating cleaned outputs.

    Attributes:
        n_assets (int): Number of assets in the portfolio.
        tickers (list): List of asset tickers. Defaults to numerical indices if not provided.
        weights (np.ndarray): Computed portfolio weights.

    Methods:
        set_weights(input_weights):
            Sets the portfolio weights manually.
        clean_weights(cutoff, rounding):
            Cleans and formats weights by removing near-zero values and applying rounding.
        _make_output_weights(weights):
            Generates an ordered dictionary of weights with asset tickers as keys.
    """

    def __init__(self, n_assets, tickers=None):
        self.n_assets = n_assets
        self.tickers = tickers if tickers else list(range(n_assets))
        self.weights = None

    def set_weights(self, input_weights: Dict[str, float]) -> None:
        self.weights = np.array([input_weights[ticker] for ticker in self.tickers])

    def clean_weights(self, cutoff=1e-4, rounding=None) -> Dict[str, float]:
        if self.weights is None:
            raise AttributeError("Weights not yet computed")

        self.weights[np.abs(self.weights) < cutoff] = 0
        if rounding is not None:
            self.weights = np.round(self.weights, rounding)

        return self._make_output_weights()

    def _make_output_weights(self, weights=None) -> Dict[str, float]:
        return collections.OrderedDict(zip(self.tickers, weights if weights is not None else self.weights))


class BaseConvexOptimizer(BaseOptimizer):
    """
    BaseConvexOptimizer is a convex optimization class designed for portfolio optimization problems.
    It extends the BaseOptimizer class by providing functionalities to define custom convex 
    optimization objectives and constraints.

    Attributes:
        n_assets (int): Number of assets in the portfolio.
        tickers (list): List of asset tickers. Defaults to numerical indices if not provided.
        weight_bounds (tuple or list): Weight bounds for the assets, either as a single tuple (global) 
                                       or a list of tuples (per asset).
        _solver (str): Solver to use for solving the optimization problem (e.g., 'ECOS', 'SCS').
        _verbose (bool): Whether to display solver output.
        _w (cp.Variable): CVXPY variable representing asset weights.
        _objective (cp.Expression): The optimization objective function.
        _constraints (list): List of constraints for the optimization problem.

    Methods:
        add_weight_bounds():
            Adds constraints for asset weight bounds based on the provided `weight_bounds` parameter.

        add_constraint(constraint_function):
            Adds a custom constraint to the optimization problem.

        convex_objective(custom_objective, weights_sum_to_one=True, **kwargs):
            Sets the optimization objective and solves the convex problem. Optionally enforces 
            the weights to sum to one.

        _solve_cvxpy_opt_problem():
            Solves the defined convex optimization problem using CVXPY and returns the portfolio weights.
    """

    def __init__(self, n_assets, tickers=None, weight_bounds=(0, 1), solver=None, verbose=False):
        super().__init__(n_assets, tickers)
        self.weight_bounds = weight_bounds
        self._solver = solver
        self._verbose = verbose

        self._w = cp.Variable(n_assets)
        self._objective = None
        self._constraints = []

        self.add_weight_bounds()

    def add_weight_bounds(self):
        if isinstance(self.weight_bounds, tuple):
            lower_bound, upper_bound = self.weight_bounds
            self._constraints.append(self._w >= lower_bound)
            self._constraints.append(self._w <= upper_bound)
        elif isinstance(self.weight_bounds, list):
            for i, (lower_bound, upper_bound) in enumerate(self.weight_bounds):
                self._constraints.append(self._w[i] >= lower_bound)
                self._constraints.append(self._w[i] <= upper_bound)

    def add_constraint(self, constraint_function: Any) -> None:
        self._constraints.append(constraint_function(self._w))

    def _solve_cvxpy_opt_problem(self) -> Dict[str, float]:
        problem = cp.Problem(cp.Minimize(self._objective), self._constraints)
        problem.solve(solver=self._solver, verbose=self._verbose)
        self.weights = self._w.value
        return self._make_output_weights()

    def convex_objective(self, custom_objective: Callable, weights_sum_to_one=True, **kwargs) -> Dict[str, float]:
        self._objective = custom_objective(self._w, **kwargs)
        if weights_sum_to_one:
            self.add_constraint(lambda w: cp.sum(w) == 1)

        return self._solve_cvxpy_opt_problem()

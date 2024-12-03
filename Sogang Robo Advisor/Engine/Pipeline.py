from typing import List, Tuple

from .Assumption import *
from .Optimizer import *
from .Tree import *

optimizer_inputs = {
    'mean_variance_optimizer': ['expected_returns', 'covariance_matrix'],
    'equal_weight_optimizer': [],
    'dynamic_risk_optimizer': ['covariance_matrix'],
    'risk_parity_optimizer': ['covariance_matrix'],
    'goal_based_optimizer': ['expected_returns', 'covariance_matrix'],

}


class Pipeline:
    """
    Pipeline is a central orchestration class for portfolio optimization. 
    It manages the execution of multiple optimization steps across a hierarchical 
    tree of assets, leveraging user-defined optimizers and calculated assumptions.

    Attributes:
        steps (List[Tuple[str, Callable]]): A list of optimization steps, where each step specifies an optimizer function.
        universe (Tree): A tree structure representing the asset hierarchy.
        assumption (AssetAssumption): An object for calculating expected returns and covariance matrices.

    Methods:
        run(price_data):
            Executes the pipeline for the provided price data, returning portfolio allocations.
            
        _optimize_node(node, depth, allocations, expected_returns, covariance_matrix, parent_weight):
            Recursively applies optimizations to nodes in the asset hierarchy.

        _get_nodes_bounds(nodes):
            Retrieves weight bounds for child nodes based on node parameters.
    """

    def __init__(self, steps: List[Tuple[str, Callable]], universe: Tree, assumption: AssetAssumption):
        self.steps = steps
        self.universe = universe
        self.assumption = assumption
        self.saa_memory = None

    def run(self, price_data: pd.DataFrame) -> Dict[str, float]:
        expected_returns = self.assumption.calculate_expected_return(price_data)
        covariance_matrix = self.assumption.calculate_covariance(price_data)

        month = price_data.index[-1].month

        valid_assets = expected_returns[expected_returns > -99999].index
        filtered_expected_returns = expected_returns.loc[valid_assets]
        filtered_covariance_matrix = covariance_matrix.loc[valid_assets, valid_assets]

        allocations = {}
        root_node = self.universe.root
        self._optimize_node(root_node, 1, allocations, filtered_expected_returns, filtered_covariance_matrix,
                            saa_rebalance_month=month)
        return allocations

    def _optimize_node(
            self,
            node: Node,
            depth: int,
            allocations: Dict[str, float],
            expected_returns: pd.Series,
            covariance_matrix: pd.DataFrame,
            parent_weight: float = 1.0,
            saa_rebalance_month=12,
    ) -> None:
        if depth <= len(self.steps):
            optimizer_name, optimizer_func = self.steps[depth - 1]

            required_inputs = optimizer_inputs.get(optimizer_func.__name__, [])
            input_args = {}

            child_names = [child.name for child in node.children]

            valid_child_names = [name for name in child_names if name in expected_returns.index]

            if len(valid_child_names) == 0:
                for child_node in node.children:
                    allocations[child_node.name] = 0.0
                return

            if 'expected_returns' in required_inputs:
                input_args['expected_returns'] = expected_returns[valid_child_names].values
            if 'covariance_matrix' in required_inputs:
                input_args['covariance_matrix'] = covariance_matrix.loc[valid_child_names, valid_child_names].values

            valid_children = [child for child in node.children if child.name in valid_child_names]
            weight_bounds = self._get_nodes_bounds(valid_children)
            if weight_bounds:
                input_args['weight_bounds'] = weight_bounds

            if optimizer_func == mean_variance_optimizer:
                node_weights = optimizer_func(
                    valid_children,
                    expected_returns=input_args['expected_returns'],
                    covariance_matrix=input_args['covariance_matrix'],
                    weight_bounds=input_args.get('weight_bounds', None),
                )

            elif optimizer_func == goal_based_optimizer:
                node_weights = optimizer_func(
                    valid_children,
                    expected_returns=input_args['expected_returns'],
                    covariance_matrix=input_args['covariance_matrix'],
                    weight_bounds=input_args.get('weight_bounds', None),
                )

            elif optimizer_func == dynamic_risk_optimizer:
                node_weights = optimizer_func(
                    valid_children,
                    covariance_matrix=input_args['covariance_matrix'],
                    weight_bounds=input_args.get('weight_bounds', None),
                )

            elif optimizer_func == risk_parity_optimizer:
                node_weights = optimizer_func(
                    valid_children,
                    covariance_matrix=input_args['covariance_matrix'],
                    weight_bounds=input_args.get('weight_bounds', None),
                )

            else:
                node_weights = optimizer_func(valid_children,
                                              **input_args,
                                              )

            if optimizer_name == 'SAA':
                node_weights = self._handle_saa(node_weights, saa_rebalance_month)

            for child_node, weight in zip(valid_children, node_weights):
                allocations[child_node.name] = weight * parent_weight
                self._optimize_node(
                    child_node,
                    depth + 1,
                    allocations,
                    expected_returns,
                    covariance_matrix,
                    weight * parent_weight
                )

            for child_node in node.children:
                if child_node.name not in valid_child_names:
                    allocations[child_node.name] = 0.0

    def _get_nodes_bounds(self, nodes: List[Node]) -> List[Tuple]:
        return [node.params['weight_bounds'] for node in nodes]

    def _handle_saa(self, node_weights, saa_rebalance_month):
        if self.saa_memory is not None:
            if saa_rebalance_month != 1:
                return self.saa_memory
            elif saa_rebalance_month == 1:
                self.saa_memory = None
        elif self.saa_memory is None:
            self.saa_memory = node_weights
        return node_weights

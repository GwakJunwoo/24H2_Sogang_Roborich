from typing import List, Dict, Any

"""
Tree and Node are foundational classes to represent hierarchical structures of assets 
or other entities. These classes support creating, manipulating, and visualizing 
tree structures for portfolio optimization or other hierarchical data processing.

Classes:
    Node:
        Represents an individual node in the tree with optional parameters, such as weight bounds.

    Tree:
        Represents a hierarchical tree structure with a root node and methods for insertion, 
        traversal, and visualization.

Node Attributes:
    name (str): The name or identifier of the node.
    children (List[Node]): A list of child nodes connected to this node.
    params (dict): A dictionary of parameters associated with the node, 
                   including weight bounds.

Tree Attributes:
    root (Node): The root node of the tree.
    nodes (Dict[str, Node]): A dictionary of all nodes in the tree, indexed by name.

Tree Methods:
    insert(parent_name, child_name, **params):
        Inserts a child node under the specified parent node.

    draw():
        Prints a visual representation of the tree structure.

    get_all_nodes():
        Returns a list of all nodes in the tree.

    get_all_nodes_name():
        Returns a list of names of all nodes in the tree, excluding the root node.

    get_leaf_nodes():
        Returns a list of names of all leaf nodes (nodes without children).
"""


class Node:
    def __init__(self, name: str, **params: Any):
        self.name = name
        self.children: List[Node] = []
        self.params = params
        self.params['weight_bounds'] = params.get('weight_bounds', (0, 1))

    def add_child(self, child_node) -> None:
        self.children.append(child_node)

    def __repr__(self):
        return f"Node({self.name}, weight_bounds={self.params['weight_bounds']})"


class Tree:
    def __init__(self, root_name: str):
        self.root = Node(root_name)
        self.nodes: Dict[str, Node] = {root_name: self.root}

    def insert(self, parent_name: str, child_name: str, **params: Any) -> bool:
        parent_node = self.nodes.get(parent_name)
        if parent_node:
            child_node = Node(child_name, **params)
            parent_node.add_child(child_node)
            self.nodes[child_name] = child_node
            return True
        return False

    def draw(self) -> None:
        lines = self._build_tree_string(self.root, '')
        print('\n'.join(lines))

    def _build_tree_string(self, node: Node, prefix: str, is_tail: bool = True) -> List[str]:
        lines = [f"{prefix}{'`-- ' if is_tail else '|-- '}{node.name}"]
        prefix += '    ' if is_tail else '|   '
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            is_last_child = i == child_count - 1
            lines.extend(self._build_tree_string(child, prefix, is_last_child))
        return lines

    def get_all_nodes(self) -> List[Node]:
        return list(self.nodes.values())

    def get_all_nodes_name(self) -> List:
        return [k.name for k in list(self.nodes.values())][1:]

    def get_leaf_nodes(self) -> List[str]:
        leaf_nodes = [node.name for node in self.nodes.values() if not node.children]
        return leaf_nodes

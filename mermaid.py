import py_trees
from py_trees.composites import Selector, Sequence

def get_node_repr(node):
    """
    Returns a Mermaid-formatted node representation.
    
    - For Selector nodes, returns a square block with a '?'.
    - For Sequence nodes, returns a square block with a '->'.
    - For all other nodes, returns a square block with the node's name.
    """
    if isinstance(node, Selector):
        return "[?]"
    elif isinstance(node, Sequence):
        return "[->]"
    else:
        return f"[{node.name}]"

def tree_to_mermaid(root):
    """
    Recursively converts a py_trees tree into a Mermaid diagram string.
    
    Args:
        root (py_trees.behaviour.Behaviour): The root node of the tree.
    
    Returns:
        str: A string containing the Mermaid diagram.
    """
    lines = ["graph TD"]

    def traverse(node):
        for child in node.children:
            # Use the unique id() to avoid naming collisions.
            parent_repr = get_node_repr(node)
            child_repr = get_node_repr(child)
            lines.append(f"    {id(node)}{parent_repr} --> {id(child)}{child_repr}")
            traverse(child)

    traverse(root)
    return "\n".join(lines)

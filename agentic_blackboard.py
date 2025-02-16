import py_trees
from typing import Any, List, Optional, Dict

from agentic_ai import ConversationThread

def initialize_blackboard(root: py_trees.behaviour.Behaviour, initial_values: dict):
    """
    Initialize the behavior tree's blackboard with a dictionary of values.
    
    Args:
        root: Root node of the behavior tree
        initial_values: Dictionary of key-value pairs to initialize in the blackboard
        
    Example:
        root = py_trees.composites.Sequence("Root")
        # Add your nodes to the root
        ...
        # Initialize blackboard
        initialize_blackboard(root, {
            "context": "initial context",
            "user_input": "sample input",
            "previous_result": None
        })
    """
    # Create a blackboard client for initialization
    blackboard = py_trees.blackboard.Client(name="Initializer")
    
    initial_values["thread"] = ConversationThread()
    initial_values["content"] = ""
    # Register and set each key-value pair
    for key, value in initial_values.items():
        blackboard.register_key(
            key=key,
            access=py_trees.common.Access.WRITE
        )
        setattr(blackboard, key, value)
    
    print(f"Initialized blackboard with values: {initial_values}")

def get_blackboard_values(keys: List[str], default_values: Optional[Dict[str, Any]] = None) -> dict:
    """
    Get multiple values from the blackboard with optional default values.
    
    Args:
        keys: List of keys to retrieve from the blackboard
        default_values: Optional dictionary of default values for keys
            
    Returns:
        Dictionary containing the requested blackboard values or defaults
        
    Raises:
        TypeError: If keys is not a list of strings
    """
    if not isinstance(keys, list):
        raise TypeError(f"keys must be a list of strings, got {type(keys)}")
    if not all(isinstance(k, str) for k in keys):
        raise TypeError("all keys must be strings")
        
    blackboard = py_trees.blackboard.Client(name="ValueReader")
    results = {}
    defaults = default_values or {}
    
    for key in keys:
        try:
            # Register key for reading
            blackboard.register_key(
                key=key,
                access=py_trees.common.Access.READ
            )
            
            # Get value with default if provided
            if hasattr(blackboard, key):
                results[key] = getattr(blackboard, key)
            else:
                results[key] = defaults.get(key)
        except (AttributeError, TypeError) as e:
            print(f"Error accessing blackboard key '{key}': {str(e)}")
            results[key] = defaults.get(key)
            
    return results

def get_blackboard_value(key: str, default: Any = None) -> Any:
    """
    Get a single value from the blackboard with an optional default.
    
    Args:
        key: Key to retrieve from the blackboard
        default: Default value to return if key doesn't exist
            
    Returns:
        Value from the blackboard or the default
        
    Raises:
        TypeError: If key is not a string
    """
    if not isinstance(key, str):
        raise TypeError(f"key must be a string, got {type(key)}")
        
    blackboard = py_trees.blackboard.Client(name="SingleValueReader")
    
    try:
        # Register key for reading
        blackboard.register_key(
            key=key,
            access=py_trees.common.Access.READ
        )
        
        return getattr(blackboard, key, default)
    except (AttributeError, TypeError) as e:
        print(f"Error accessing blackboard key '{key}': {str(e)}")
        return default
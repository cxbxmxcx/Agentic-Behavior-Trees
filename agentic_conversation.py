import py_trees
import time
from typing import Optional

from agentic_blackboard import get_blackboard_value

def run_conversation_loop(tree: py_trees.trees.BehaviourTree, 
                         root: py_trees.behaviour.Behaviour,
                         tick_interval: float = 0.1) -> None:
    """
    Run a conversation loop with the behavior tree, handling user input and tree responses.
    """
    # Initialize the tree
    tree.setup()
    print("Starting conversation (type 'exit' to end)...")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("Ending conversation...")
            break
            
        # Set the question in the blackboard
        blackboard = py_trees.blackboard.Client(name="ConversationClient")
        blackboard.register_key("question", py_trees.common.Access.WRITE)
        blackboard.question = user_input
        
        # Get all nodes in the tree
        nodes = [root]
        for node in root.iterate():
            nodes.append(node)
            
        # Reset all nodes to a clean state
        for node in nodes:
            node.initialise()
        
        # Track if we've completed one full iteration
        initial_tick_count = tree.count
        
        # Run the tree until success or full iteration completed
        while True:
            tree.tick()
            time.sleep(tick_interval)
            
            # Check for success
            if root.status == py_trees.common.Status.SUCCESS:
                content = get_blackboard_value("content", default="No response generated.")
                print(f"\nAssistant: {content}")
                break
                
            # Check if we've completed a full iteration without success
            if tree.count > initial_tick_count + len(nodes):
                content = get_blackboard_value("content", default="Failed to generate a response.")
                print(f"\nAssistant: {content}")
                break
        
        # Cleanup nodes after iteration
        for node in nodes:
            if node.status != py_trees.common.Status.INVALID:
                node.stop()
                
        # Clear or reset blackboard values for next iteration
        blackboard.question = ""
    
    # Cleanup at end
    tree.shutdown()
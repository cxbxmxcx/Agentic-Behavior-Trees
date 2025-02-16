import py_trees
from jinja2 import Template

class AgentWrapper:
    def __init__(self, agent, agent_instructions=None, **kwargs):
        """
        Wrapper class for the agent to handle calling with template instructions.
        
        Args:
            agent: The agent instance
            agent_instructions: A Jinja2 template string for agent instructions
            kwargs: Additional keyword arguments for the agent
        """
        self.agent = agent
        self.template = Template(agent_instructions) if agent_instructions else None
        self.kwargs = kwargs

    def __call__(self, context=None):
        """
        Execute the agent call synchronously with template rendering.
        
        Args:
            context: Dictionary of values to render the template with
        """
        if self.template and context:
            instructions = self.template.render(**context)
        elif self.template:
            instructions = self.template.render()
        else:
            instructions = None
            
        return self.agent.ask_agent(instructions, **self.kwargs)


class ActionWrapper(py_trees.behaviour.Behaviour):
    def __init__(self, 
                 name, 
                 agent_wrapper, 
                 is_condition=False,
                 input_keys=None,
                 output_keys=None,
                 ):
        """
        A synchronous action wrapper for behavior trees.
        
        Args:
            name: Name of the action
            agent_wrapper: Wrapped agent instance
            is_condition: Whether this action is a condition check
            input_keys: List of input keys to read from blackboard
            output_keys: List of output keys to write to blackboard
        """
        super(ActionWrapper, self).__init__(name=name)        
        self.agent_wrapper = agent_wrapper
        self.is_condition = is_condition
        self.success = False
        self.run_context = None

        # Blackboard data management
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        
        # Create blackboard client with node name
        self.blackboard = self.attach_blackboard_client(name=name)
        
        # Create blackboard references for all input and output keys
        for key in self.input_keys:
            self.blackboard.register_key(
                key=key,
                access=py_trees.common.Access.READ
            )
        
        for key in self.output_keys:
            self.blackboard.register_key(
                key=key,
                access=py_trees.common.Access.WRITE
            )

    def setup(self):
        """Set up any necessary resources."""
        print(f"{self.name}.setup()")
        return py_trees.common.Status.SUCCESS

    def initialise(self):
        """Initialize the action state."""
        print(f"{self.name}.initialise()")
        self.success = False

    def update(self):
        """
        Execute the action synchronously and return the status.
        This is called every tick of the behavior tree.
        """
        print(f"{self.name}.update()")
        
        try:
            # Execute the agent action synchronously
            context = {}
            for key in self.input_keys:
                if hasattr(self.blackboard, key):
                    context[key] = getattr(self.blackboard, key)
            
            # Execute the agent action synchronously with context
            result = self.agent_wrapper(context=context)
            print(result)

            # Write any output data to blackboard
            if isinstance(result, dict):
                for key in self.output_keys:
                    if key in result:
                        setattr(self.blackboard, key, result[key])

            # Check for explicit failure
            if "FAILURE" in result.get("content", ""):
                print(f"{self.name}: Action completed with failure.")
                return py_trees.common.Status.FAILURE

            # Handle condition vs action behavior
            if self.is_condition:
                self.success = "SUCCESS" in result.get("content", "")
            else:
                self.success = True

            print(f"{self.name}: Action completed successfully.")
            return (
                py_trees.common.Status.SUCCESS
                if self.success
                else py_trees.common.Status.FAILURE
            )

        except Exception as e:
            print(f"{self.name}: Exception in action: {str(e)}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Clean up when the action terminates."""
        print(f"{self.name}.terminate({new_status})")
        self.success = False


def create_agent_node(name: str, 
                     agent, 
                     agent_instructions: str, 
                     is_condition: bool = False,
                     input_keys: list = None,
                     output_keys: list = None,
                     **kwargs):
    """
    Create a new agent action node for the behavior tree.
    
    Args:
        name: Name of the action node
        agent: Agent instance
        agent_instructions: Instructions template for the agent
        is_condition: Whether this action is a condition check
        input_keys: List of input keys to read from blackboard
        output_keys: List of output keys to write to blackboard
        kwargs: Additional keyword arguments to pass to the agent
        
    Returns:
        ActionWrapper instance configured with the provided parameters
    """
    agent_wrapper = AgentWrapper(
        agent=agent,
        agent_instructions=agent_instructions,
        **kwargs
    )
    return ActionWrapper(
        name=name,      
        agent_wrapper=agent_wrapper,
        is_condition=is_condition,
        input_keys=input_keys,
        output_keys=output_keys
    )


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
    
    # Register and set each key-value pair
    for key, value in initial_values.items():
        blackboard.register_key(
            key=key,
            access=py_trees.common.Access.WRITE
        )
        setattr(blackboard, key, value)
    
    print(f"Initialized blackboard with values: {initial_values}")
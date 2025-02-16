import py_trees

class AgentWrapper:
    def __init__(self, agent, *args, **kwargs):
        """
        Wrapper class for the agent to handle calling with args and kwargs.
        """
        self.agent = agent
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        """Execute the agent call synchronously."""
        return self.agent.ask_agent(*self.args, **self.kwargs)


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
            # Read any required input data from blackboard
            context = {}
            for key in self.input_keys:
                if hasattr(self.blackboard, key):
                    context[key] = getattr(self.blackboard, key)
            
            # Update agent instructions with context if needed
            if context:
                if isinstance(self.agent_wrapper.args[0], str):
                    instructions = self.agent_wrapper.args[0].format(**context)
                    self.agent_wrapper.args = (instructions,) + self.agent_wrapper.args[1:]
            
            # Execute the agent action synchronously
            result = self.agent_wrapper()
            print(result)

            # Write any output data to blackboard
            if isinstance(result, dict):
                for key in self.output_keys:
                    if key in result:
                        setattr(self.blackboard, key, result[key])

            # Check for explicit failure
            if "FAILURE" in result.get("text", ""):
                print(f"{self.name}: Action completed with failure.")
                return py_trees.common.Status.FAILURE

            # Handle condition vs action behavior
            if self.is_condition:
                self.success = "SUCCESS" in result["content"]
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


def create_agent_node(name: str, agent, agent_instructions: str, is_condition: bool = False):
    """
    Create a new agent action node for the behavior tree.
    
    Args:
        name: Name of the action node
        agent: Agent instance
        agent_instructions: Instructions for the agent
        is_condition: Whether this action is a condition check
        
    Returns:
        ActionWrapper instance configured with the provided parameters
    """
    agent_wrapper = AgentWrapper(
        agent, agent_instructions
    )
    return ActionWrapper(
        name=name,      
        agent_wrapper=agent_wrapper,
        is_condition=is_condition
    )
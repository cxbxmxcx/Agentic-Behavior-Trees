from jinja2 import Environment, FileSystemLoader
from openai import AzureOpenAI, RateLimitError
import time
import os
import logging
import tiktoken
import inspect
import functools
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

def handle_semantic_function_call(prompt, agent):
    system, user = parse_prompt(prompt)
    response = agent.get_semantic_response(system, user)
    return response

def parse_prompt(prompt):
    # Prepare the dictionary to hold the parsed contents
    parsed_contents = {"System": "", "User": ""}

    # Current section being parsed
    current_section = None

    # Split the docstring into lines and iterate through them
    for line in prompt.split("\n"):
        # Check if the line marks the beginning of a section
        if line.strip().startswith("System:"):
            current_section = "System"
            continue  # Skip the current line to avoid including the section identifier
        elif line.strip().startswith("User:"):
            current_section = "User"
            continue  # Skip the current line to avoid including the section identifier

        # Add the line to the current section if it's not None
        if current_section:
            # Add the line to the appropriate section, maintaining line breaks for readability
            parsed_contents[current_section] += line.strip() + "\n"

    # Trim the trailing newlines from each section's content
    for key in parsed_contents:
        parsed_contents[key] = parsed_contents[key].rstrip("\n")

    return parsed_contents["System"], parsed_contents["User"]


@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[List[Any]] = None
    tool_call_results: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ConversationThread:
    def __init__(self):
        self.messages: List[Message] = []
        
    def add_message(self, role: str, content: str, tool_calls=None, tool_call_results=None):
        message = Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_results=tool_call_results
        )
        self.messages.append(message)
        
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Convert thread messages to format expected by LLM API."""
        llm_messages = []
        
        for msg in self.messages:
            message_dict = {"role": msg.role, "content": msg.content}
            
            # If there are tool calls and results, add tool call messages
            if msg.tool_calls and msg.tool_call_results:
                # Add the assistant's tool calls
                message_dict["tool_calls"] = msg.tool_calls
                llm_messages.append(message_dict)
                
                # Add tool results as function response messages
                for tool_result in msg.tool_call_results:
                    tool_call = tool_result["tool_call"]
                    result = tool_result["result"]
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(result)
                    })
            else:
                llm_messages.append(message_dict)
                
        return llm_messages

    def get_conversation_history(self) -> str:
        """Get formatted conversation history for debugging/logging."""
        history = []
        for msg in self.messages:
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            history.append(f"[{timestamp}] {msg.role}: {msg.content}")
            
            if msg.tool_calls and msg.tool_call_results:
                for tool_result in msg.tool_call_results:
                    tool_call = tool_result["tool_call"]
                    result = tool_result["result"]
                    history.append(f"  └─ Tool Call: {tool_call.function.name}")
                    history.append(f"     Arguments: {tool_call.function.arguments}")
                    history.append(f"     Result: {result}")
                    
        return "\n".join(history)


class LLMEngine:
    def __init__(self, 
                 api_key: str,
                 api_endpoint: str,
                 api_version: str,
                 deployment_name: str):
        """
        Initialize the LLMEngine with Azure OpenAI parameters.
        """
        if not all([api_key, api_endpoint, api_version]):
            raise ValueError("Please set the Azure-OpenAI-key, Azure-OpenAI-endpoint, and Azure-OpenAI-api-version secrets.")
        
        self.deployment_name = deployment_name
        self.in_tokens = 0
        self.out_tokens = 0
        
        self.client = AzureOpenAI(
            api_key=api_key,  
            api_version=api_version,
            azure_endpoint=api_endpoint
        )

    def generate_response(self, 
                         messages: List[Dict[str, str]],
                         tools: Optional[List[Dict[str, Any]]] = None,
                         **api_kwargs) -> Dict[str, Any]:
        """
        Generate a response from Azure OpenAI using the provided message history and tools.
        """
        api_parameters = {
            'model': self.deployment_name,
            'messages': messages,
            'max_tokens': 12800,
            'temperature': 0.7
        }

        if self.deployment_name.startswith("o3-mini"):
                api_parameters['max_completion_tokens'] = api_parameters['max_tokens']
                del api_parameters['max_tokens']
                del api_parameters['temperature']
        
        if tools:
            api_parameters['tools'] = tools
            api_parameters['tool_choice'] = 'auto'
            
        api_parameters.update(api_kwargs)

        delay = 5
        max_retries = 3
        backoff_factor = 2

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(**api_parameters)
                self.in_tokens += response.usage.prompt_tokens
                self.out_tokens += response.usage.completion_tokens
                
                result = {
                    'content': response.choices[0].message.content,
                    'tool_calls': getattr(response.choices[0].message, 'tool_calls', None)
                }
                return result
                
            except RateLimitError:
                if attempt == max_retries:
                    logging.warning(f"Attempt {attempt}: Rate limit exceeded. No more retries left.")
                    raise
                else:
                    logging.warning(f"Attempt {attempt}: Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
            except Exception as e:
                logging.error(f"Attempt {attempt}: An error occurred: {e}")
                raise RuntimeError(f"Failed to get a response from API. {str(e)}") from e

        raise RateLimitError("Max retry attempts exceeded due to rate limiting.")


def agent_action(func):
    """
    Decorator to convert a function into an agent action with OpenAI function calling format.
    """
    @functools.wraps(func)
    def wrapper(*args, _agent=None, **kwargs):
        if hasattr(wrapper, "_prompt_template") and _agent:
            adjusted_template = wrapper._prompt_template.replace("{{", "{").replace(
                "}}", "}"
            )
            prompt = adjusted_template.format(*args, **kwargs)
            return handle_semantic_function_call(prompt, _agent)
        else:
            return func(*args, **kwargs)
            
    # Inspect the function's signature
    sig = inspect.signature(func)
    params = sig.parameters.values()
    
    # Construct properties and required fields
    properties = {}
    required = []
    
    for param in params:
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
            properties[param.name] = {
                "type": "string",
                "description": param.name,
            }
        else:
            if param.name == "unit":
                properties[param.name] = {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                }
            else:
                properties[param.name] = {
                    "type": "string",
                    "description": param.name,
                }
                
    # Construct the OpenAI function specification
    func_spec = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    
    # Add prompt template if function is semantic
    if func.__doc__ and "{{" in func.__doc__ and "}}" in func.__doc__:
        wrapper._prompt_template = func.__doc__
        
    wrapper._agent_action = func_spec
    wrapper._original_func = func
    return wrapper


class Agent:
    def __init__(self, 
                 name: str,
                 api_key: str,
                 api_endpoint: str,
                 api_version: str,
                 deployment_name: str, 
                 template_dir: str = 'agent/prompt_templates', 
                 model_encoding: str = 'cl100k_base',
                 max_tokens: int = 12800):
        """
        Initialize the Agent with Azure OpenAI parameters.
        """
        self.name = name
        self.model_encoding = model_encoding
        self.deployment_name = deployment_name
        self.max_tokens = max_tokens
        self.tools = []  # List to store tool descriptions
        self.functions = {}  # Dictionary to store function implementations
        self.thread = ConversationThread()

        # Initialize the LLM engine
        self.llm_engine = LLMEngine(
            api_key=api_key,
            api_endpoint=api_endpoint,
            api_version=api_version,
            deployment_name=deployment_name
        )

        # Set up Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.system_prompt = None
        
    def load_system_prompt(self, template_name: str, context_variables: dict):
        """
        Load and render the system prompt template.
        """
        try:
            template = self.env.get_template(template_name)
        except Exception as e:
            raise FileNotFoundError(f"Template file '{template_name}' not found in the templates directory.") from e
        
        # Validate token count
        encoding = tiktoken.get_encoding(self.model_encoding)
        total_tokens = sum(len(encoding.encode(str(value))) for value in context_variables.values())
        if total_tokens > self.max_tokens:
            raise ValueError(f"The total number of tokens in context variables exceeds the limit of {self.max_tokens}.")

        self.system_prompt = template.render(context_variables)
        # Add system prompt to conversation thread
        self.thread.add_message("system", self.system_prompt)
        
    def add_tool(self, func: Callable):
        """
        Add a tool by inspecting the provided function and storing both its description and implementation.
        
        Args:
            func: Function decorated with @agent_action
        """
        if not hasattr(func, '_agent_action'):
            raise ValueError("Function must be decorated with @agent_action")
            
        self.tools.append(func._agent_action)
        self.functions[func.__name__] = func._original_func
        
    def execute_tool_call(self, tool_call):
        """
        Execute a tool call using the stored function implementation.
        
        Args:
            tool_call: Tool call object from the LLM response
        """
        func_name = tool_call.function.name
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
            
        # Parse arguments from the function call
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid function arguments: {tool_call.function.arguments}")
            
        # Execute the function with the provided arguments
        return self.functions[func_name](**arguments)
            
    def ask_agent(self, user_input: str, system_template: str=None, context: Dict=None, **api_kwargs) -> Dict[str, Any]:
        """
        Process user input using the agent's system prompt and tools.
        Handles multiple turns of tool calling until a final response is reached.
        
        Args:
            user_input: The user's input text
            **api_kwargs: Additional API parameters
            
        Returns:
            Dict containing the final response and conversation thread
        """
        if system_template and context:
            self.load_system_prompt(system_template, context)
        # if not self.system_prompt:
        #     raise ValueError("System prompt has not been loaded. Call load_system_prompt first.")
            
        # Add user input to conversation thread
        self.thread.add_message("user", user_input)
        
        while True:
            # Get current conversation history in LLM format
            messages = self.thread.get_messages_for_llm()
            
            # Generate response
            response = self.llm_engine.generate_response(
                messages=messages,
                tools=self.tools if self.tools else None,
                **api_kwargs
            )
            
            # If no tool calls, we're done
            if not response['tool_calls']:
                self.thread.add_message("assistant", response['content'])
                break
                
            # Execute tool calls and add results to thread
            tool_results = []
            for tool_call in response['tool_calls']:
                result = self.execute_tool_call(tool_call)
                tool_results.append({
                    'tool_call': tool_call,
                    'result': result
                })
                
            self.thread.add_message(
                "assistant",
                response['content'],
                tool_calls=response['tool_calls'],
                tool_call_results=tool_results
            )
            
        return {
            'content': response['content'],
            'thread': self.thread,
            'token_usage': {
                'input_tokens': self.llm_engine.in_tokens,
                'output_tokens': self.llm_engine.out_tokens
            }
        }
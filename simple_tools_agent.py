from datetime import datetime
import os
from agentic_ai import Agent, agent_action
from dotenv import load_dotenv

@agent_action
def get_current_timestamp():
    """Return the current date/time as a timestamp."""
    return int(datetime.now().timestamp())

@agent_action
def create_report(timestamp: str):
    """Return the timestamp as a report."""
    return "The agents rose up on today's date is: " + str(datetime.fromtimestamp(int(timestamp)))

agent = Agent("simple agent")
agent.add_tool(get_current_timestamp)
agent.add_tool(create_report)

response = agent.ask_agent("return the report?")

print(response, response["thread"].messages)



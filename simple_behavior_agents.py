from datetime import datetime
import os
from agentic_ai import Agent, agent_action
from dotenv import load_dotenv
import py_trees
import time

from agentic_blackboard import get_blackboard_value, initialize_blackboard
from agentic_btrees import create_agent_node
from agentic_conversation import run_conversation_loop
import mermaid

@agent_action
def get_current_timestamp():
    """Return the current date/time as a timestamp."""
    return int(datetime.now().timestamp())

@agent_action
def create_report(timestamp: str):
    """Return the timestamp as a report."""
    return "The agents rose up on today's date is: " + str(datetime.fromtimestamp(int(timestamp)))

agent = Agent("simple agent")
# agent.add_tool(get_current_timestamp)
# agent.add_tool(create_report)

# Create the root node (sequence)
root = py_trees.composites.Selector("RootSelector", memory=True)

sequence_node = py_trees.composites.Sequence("Sequence", memory=True)
root.add_child(sequence_node)

# inputs = dict(question = "list the belts ?")
initialize_blackboard(root, {})

expansion_glossary_node = create_agent_node(
    name="ExpansionWithGlossary",
    agent=agent,
    agent_instructions="""
    Expand the question with a glossary of terms.
    Return FAILURE if you cannot expand the question.
    Question: {{question}}
    """,
    input_keys=["question"],    
)
sequence_node.add_child(expansion_glossary_node)

identify_data_sources_node = create_agent_node(
    name="IdentifyDataSources",
    agent=agent,
    agent_instructions="""    
    Identify the data sources for the question.
    Return FAILURE if you cannot identify the data sources.
    Question: {{question}}
    """,
    input_keys=["question",],
    
)
sequence_node.add_child(identify_data_sources_node)

answer_question_node = create_agent_node(
    name="AnswerQuestion",
    agent=agent,
    agent_instructions="""    
    Answer the question.
    Return FAILURE if you cannot answer the question.
    Question: {{question}}
    """,
    input_keys=["question",],   
)
sequence_node.add_child(answer_question_node)

copyright_safety_node = create_agent_node(
    name="CopyrightSafety",
    agent=agent,
    agent_instructions="""    
    Check for copyright safety.
    Return FAILURE if the content is not safe for copyright.
    Question: {{question}}
    """,
    input_keys=["question"],    
)
sequence_node.add_child(copyright_safety_node)

ask_questions_node = create_agent_node(
    name="AskQuestions",
    agent=agent,
    agent_instructions="""    
    Ask clarifying questions.
    Question: {{question}}
    """,
    input_keys=["question",],    
)

root.add_child(ask_questions_node)

tree = py_trees.trees.BehaviourTree(root)

diagram = mermaid.tree_to_mermaid(root)
print(diagram)

# # Tick the tree to run it
# while True:
#     tree.tick()
#     time.sleep(10)
#     if root.status == py_trees.common.Status.SUCCESS:
#         break

# print(get_blackboard_value("content"))

run_conversation_loop(tree, root)
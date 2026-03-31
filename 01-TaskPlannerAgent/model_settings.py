import os

from agents import Agent, Runner, ModelSettings
from dotenv import load_dotenv

MY_INSTRUCTIONS = """
You are a research planning assistant.

- Provide a research plan
- Output exactly 5 concise tasks
- Each task must be 5 words or less
- Return plain text only
"""

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing")

# Agent =
#     system prompt (instructions)
#     + metadata (name)
#     + default model
agent = Agent(

    name="Research Planner",
    instructions=MY_INSTRUCTIONS,
    model="gpt-4.1",  # Specify the model to use
    model_settings=ModelSettings(
        temperature=0.0,  # temperature <> repeatability
        max_tokens=150,  #  maximum number of tokens in the response
        top_p=1.0,  # Set the top-p sampling parameter
        frequency_penalty=0.5,  # frequency penalty
        presence_penalty=0.5,  #  presence penalty
    )
)
# Mini orchestration engine
result = Runner.run_sync(
    agent,
    input="learn about AI agents",
)

print(result.final_output)
#  with parameters

# 1. Review foundational AI agent literature
# 2. Analyze agent architectures and types
# 3. Study real - world agent applications
# 4. Compare agent frameworks and tools
# 5. Summarize key findings and trends
#
# 1. Review foundational AI agent literature
# 2. Analyze types of AI agents
# 3. Study agent - environment interaction models
# 4. Examine real - world AI agent applications
# 5. Summarize key findings and trends

# without parameters
#
# 1. Review foundational AI agent literature
# 2. Identify types of AI agents
# 3. Analyze agent design architectures
# 4. Explore real-world agent applications
# 5. Summarize key findings and trends
# (agents-env) ubuntu@ip-172-31-35-241:~$ python basic_agent.py
# 1. Review foundational AI agent literature
# 2. Identify key types of agents
# 3. Analyze agent architectures and models
# 4. Examine real-world agent applications
# 5. Summarize current research challenges

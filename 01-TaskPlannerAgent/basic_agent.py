import os

from agents import Agent, Runner
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
)
# Mini orchestration engine
result = Runner.run_sync(
    agent,
    input="learn about AI agents",
)

print(result.final_output)
# 54t = 54 tokens (input)
# 43t = 43 tokens (output)
# Properties
# Created
# Mar 29, 2026, 8:33 AM
# ID
# resp_XXX
# Model
# gpt-4.1-2025-04-14
# Tokens
# 97 total
# Configuration
# Response
# text
# Verbosity
# medium

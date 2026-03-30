from typing import List

from agents import Agent, Runner, trace, function_tool
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# load environment
load_dotenv()

INSTRUCTIONS = """
You are a research planning assistant.

You will receive a research topic.

Steps:
- First, use the tool get_research_sources() to retrieve the available research sources.
- Then create a research plan using only those available sources.

Rules:
- Return exactly 5 tasks.
- Each task must be concise.
- Each task description must be 5 words or fewer.
- Each task must specify which available research source will be used.
- Make the tasks practical and high-level.
- Return structured output only.
"""


# define strict output schema
class Task(BaseModel):
    step: int = Field(..., description="Task number starting from 1")
    research_source: str = Field(..., description="One available research source to use")
    description: str = Field(..., description="Concise task, 5 words or fewer")

    model_config = ConfigDict(extra="forbid")


class ResearchPlanModel(BaseModel):
    tasks: List[Task] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly 5 numbered research tasks",
    )

    model_config = ConfigDict(extra="forbid")


# register a tool that returns allowed sources
# the runtime:
#
# finds the Python function
# executes it
# returns the result back to the LLM
@function_tool
def get_research_sources() -> List[str]:
    """Provides a list of available research sources."""
    return [
        "Wikipedia",
        "Google",
        "YouTube",
    ]


# create the agent
agent = Agent(
    name="Research Planner",
    instructions=INSTRUCTIONS,
    output_type=ResearchPlanModel,
    tools=[get_research_sources],
)

topic = "learn about AI agents"
# trace the workflow
# runner sends:
# instructions
# topic
# output schema
# tool definition
# model calls get_research_sources()
# runtime executes the Python function
# tool result goes back to the model
# model returns structured JSON with:
# step
# research_source
# description
with trace("Research Planning Workflow"):
    result = Runner.run_sync(
        agent,
        input=topic,
    )

print(result.final_output.model_dump_json(indent=2))

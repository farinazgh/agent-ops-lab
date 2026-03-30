from typing import List

from agents import Agent, Runner, function_tool, trace
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

INSTRUCTIONS = """
You are a research planning assistant.

You will receive a research topic.

Steps:
- First, use the tool get_research_sources() to retrieve the available research sources.
- Then, use the tool get_resource_url() to look up the URL for each research source you use.
- Create a research plan using only the available research sources.

Rules:
- Return exactly 5 tasks.
- Each task must be concise.
- Each task description must be 5 words or fewer.
- Each task must specify the research source used.
- Each research source must include both a name and a URL.
- Make the tasks practical and high-level.
- Return structured output only.
"""


class ResearchSource(BaseModel):
    name: str = Field(..., description="Name of the research source")
    url: str = Field(..., description="URL of the research source")

    model_config = ConfigDict(extra="forbid")


class Task(BaseModel):
    step: int = Field(..., description="Task number starting from 1")
    research_source: ResearchSource = Field(
        ..., description="Research source to use for this task"
    )
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


@function_tool
def get_research_sources() -> List[str]:
    """Provides a list of available research sources."""
    return [
        "Wikipedia",
        "Google",
        "YouTube",
    ]


@function_tool
def get_resource_url(research_source: str) -> str:
    """Provides the URL for a given research source."""
    search_sources = {
        "Wikipedia": "https://www.wikipedia.org",
        "Google": "https://www.google.com",
        "YouTube": "https://www.youtube.com",
    }
    return search_sources[research_source]


agent = Agent(
    name="Research Planner",
    instructions=INSTRUCTIONS,
    output_type=ResearchPlanModel,
    tools=[get_research_sources, get_resource_url],
)

topic = "learn about AI agents"

with trace("Tool-Augmented Research Planner"):
    result = Runner.run_sync(
        agent,
        input=topic,
    )

print(result.final_output.model_dump_json(indent=2))
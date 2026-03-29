from typing import List

from agents import Agent, Runner
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
# Format the output as if it will be used as input for another agent; it is unified and standardized as JSON.
INSTRUCTIONS = """
You are a research planning assistant.

You will receive a research topic.
Return a research plan with exactly 5 tasks.

Rules:
- Each task must be concise.
- Each task description must be 5 words or fewer.
- Make the tasks practical and high-level.
- Return structured output only.
"""


class Task(BaseModel):
    id: int = Field(..., description="Task number starting from 1")
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


agent = Agent(
    name="Research Planner",
    instructions=INSTRUCTIONS,
    output_type=ResearchPlanModel,
)

topic = "learn about AI agents"

result = Runner.run_sync(
    agent,
    input=topic,
)

print(result.final_output.model_dump_json(indent=2))
# {
#   "tasks": [
#     {
#       "id": 1,
#       "description": "Define AI agents conceptually"
#     },
#     {
#       "id": 2,
#       "description": "Explore AI agent architectures"
#     },
#     {
#       "id": 3,
#       "description": "Review common applications"
#     },
#     {
#       "id": 4,
#       "description": "Compare learning methods"
#     },
#     {
#       "id": 5,
#       "description": "Analyze current research trends"
#     }
#   ]
# }

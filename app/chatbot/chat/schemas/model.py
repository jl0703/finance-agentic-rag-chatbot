from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """
    The schema for plan execution by a supervisor agent.
    """

    steps: list[str] = Field(
        description="A list of steps to execute the plan sequentially to answer the user's query.",
        default_factory=list,
    )


class Supervisor(BaseModel):
    """
    The schema for the supervisor agent to decide the next node to execute.
    """

    next_node: Literal["retrieval", "generator", "__end__"] = Field(
        description="The next node to execute in the workflow."
    )

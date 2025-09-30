from typing import TypedDict, Optional, List


class InputState(TypedDict):
    """Input state for chat"""

    user_id: str
    message: str


class OutputState(TypedDict):
    """Output state for chat"""

    response: str


class OverallState(InputState, OutputState):
    """Overall state for chat"""

    plan: Optional[str]
    tools: Optional[str]
    retrieved_docs: Optional[str]
    tools_used: Optional[List[str]]
    is_cached: bool = False

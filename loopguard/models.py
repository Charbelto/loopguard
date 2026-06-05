from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class AgentStep:
    """Represents a single iteration/turn of an agent's execution loop."""

    action: str  # The high-level action category (e.g. tool name, thought)
    input_text: str  # Inputs provided to the action (e.g. tool arguments, query)
    observation: Optional[str] = None  # Observation or result returned from the action
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class State:
    """Represents a semantically unique state clusters formed from step histories."""

    id: int
    representative_step: AgentStep
    steps_count: int = 1

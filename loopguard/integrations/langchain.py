from typing import Any, Dict
from ..core import LoopGuard

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    # Fallback mock class if LangChain is not installed
    class BaseCallbackHandler:  # type: ignore
        pass


class LoopGuardCallbackHandler(BaseCallbackHandler):
    """
    A LangChain-compatible callback handler that hooks into tool start/end
    lifecycles to prevent repeating execution patterns.
    """

    def __init__(self, guard: LoopGuard):
        self.guard = guard

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Runs right before a tool is invoked."""
        tool_name = serialized.get("name", "unknown_tool")
        self.guard.add_step(action=tool_name, input_text=input_str)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Runs after a tool completes execution."""
        if self.guard.steps_history:
            self.guard.steps_history[-1].observation = str(output)
            # Re-fit the vocabulary to include the new observation
            text_to_fit = f"{self.guard.steps_history[-1].input_text} {output}"
            self.guard.tracker.tfidf.fit_step(text_to_fit)

from typing import Any, Dict, Optional
from ..core import LoopGuard

try:
    from llama_index.core.callbacks import BaseCallbackHandler
except ImportError:
    # Fallback mock class if LlamaIndex is not installed
    class BaseCallbackHandler:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


class LlamaIndexLoopGuardHandler(BaseCallbackHandler):
    """
    A LlamaIndex-compatible CallbackHandler that tracks tool events
    and intercepts execution loops.
    """

    def __init__(self, guard: LoopGuard):
        # Initialize parent class with default ignorable events if it matches expected signature
        try:
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        except TypeError:
            super().__init__()
        self.guard = guard

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when a LlamaIndex event starts (e.g. tool execution)."""
        event_str = str(event_type).lower()
        if "tool" in event_str or "function" in event_str:
            if payload:
                tool_name = payload.get("tool_name") or payload.get("name") or "tool"
                tool_input = payload.get("tool_input") or payload.get("input") or ""
                self.guard.add_step(action=str(tool_name), input_text=str(tool_input))
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when a LlamaIndex event ends (e.g. tool returns observation)."""
        event_str = str(event_type).lower()
        if "tool" in event_str or "function" in event_str:
            if payload and self.guard.steps_history:
                output = payload.get("response") or payload.get("output") or ""
                self.guard.steps_history[-1].observation = str(output)
                # Re-fit the vocabulary to include the new observation
                text_to_fit = f"{self.guard.steps_history[-1].input_text} {output}"
                self.guard.tracker.tfidf.fit_step(text_to_fit)

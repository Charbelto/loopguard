from typing import Dict, Any, Callable, Tuple
from ..core import LoopGuard, LoopDetectedError


def extract_action_inputs(message: Any) -> Tuple[str, str]:
    """
    Extracts action name and input string from various message formats.
    Supports raw dictionaries, LangChain message objects, and tool calls.
    """
    # 1. Dictionary format
    if isinstance(message, dict):
        action = message.get("action") or message.get("role") or "agent"
        inputs = message.get("inputs") or message.get("content") or ""
        return str(action), str(inputs)

    # 2. LangChain AIMessage with tool_calls attribute
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_call = message.tool_calls[0]
        name = tool_call.get("name", "tool")
        args = tool_call.get("args", "")
        return str(name), str(args)

    # 3. LangChain message with additional_kwargs tool_calls
    if (
        hasattr(message, "additional_kwargs")
        and "tool_calls" in message.additional_kwargs
    ):
        tc = message.additional_kwargs["tool_calls"]
        if tc and isinstance(tc, list):
            first = tc[0]
            if isinstance(first, dict):
                func = first.get("function", {})
                name = func.get("name", "tool")
                args = func.get("arguments", "")
                return str(name), str(args)

    # 4. Fallback for objects (e.g. BaseMessage subclasses)
    action = getattr(message, "type", "agent")
    content = getattr(message, "content", "")
    return str(action), str(content)


def wrap_node_with_guard(
    node_fn: Callable[[Dict[str, Any]], Dict[str, Any]], guard: LoopGuard
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Wraps a LangGraph node to verify inputs and automatically inject healing prompts
    into state messages if a loop is detected.

    Args:
        node_fn: The LangGraph node function to wrap.
        guard: The LoopGuard coordinator instance.

    Returns:
        The wrapped node function.
    """

    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return node_fn(state)

        last_message = messages[-1]
        action, inputs = extract_action_inputs(last_message)

        try:
            guard.add_step(action=action, input_text=inputs)
            return node_fn(state)
        except LoopDetectedError as e:
            healing_prompt = guard.get_healing_prompt(e.violated_rule)

            # Inject healing notice into state messages to guide LLM redirection
            if "messages" in state:
                # If messages are dictionaries, append as dictionary
                if isinstance(messages[0], dict) if len(messages) > 0 else True:
                    state["messages"].append(
                        {"role": "system", "content": healing_prompt}
                    )
                else:
                    # Try to import LangChain SystemMessage if possible, otherwise use dict
                    try:
                        from langchain_core.messages import SystemMessage

                        state["messages"].append(SystemMessage(content=healing_prompt))
                    except ImportError:
                        state["messages"].append(
                            {"role": "system", "content": healing_prompt}
                        )

            return state

    return wrapped

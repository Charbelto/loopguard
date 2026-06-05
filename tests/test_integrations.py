from typing import Dict, Any
import pytest
from loopguard.core import LoopGuard, LoopDetectedError
from loopguard.integrations.langchain import LoopGuardCallbackHandler
from loopguard.integrations.langgraph import wrap_node_with_guard, extract_action_inputs
from loopguard.integrations.llamaindex import LlamaIndexLoopGuardHandler


# --- Mock Classes to mimic LangChain message objects ---
class MockMessage:

    def __init__(
        self,
        msg_type: str,
        content: str,
        tool_calls: list = None,
        additional_kwargs: dict = None,
    ):
        self.type = msg_type
        self.content = content
        self.tool_calls = tool_calls or []
        self.additional_kwargs = additional_kwargs or {}


# --- Tests for extract_action_inputs helper ---
def test_extract_action_inputs():
    # 1. Dictionary format
    action, inputs = extract_action_inputs({"action": "search", "inputs": "query"})
    assert action == "search"
    assert inputs == "query"

    # 2. Mock LangChain AIMessage with tool_calls attribute
    msg_tc = MockMessage(
        "ai", "", tool_calls=[{"name": "calculator", "args": {"expr": "2+2"}}]
    )
    action, inputs = extract_action_inputs(msg_tc)
    assert action == "calculator"
    assert "2+2" in inputs

    # 3. Mock message with additional_kwargs tool_calls
    msg_kw = MockMessage(
        "ai",
        "",
        additional_kwargs={
            "tool_calls": [
                {"function": {"name": "web_search", "arguments": "chicago weather"}}
            ]
        },
    )
    action, inputs = extract_action_inputs(msg_kw)
    assert action == "web_search"
    assert inputs == "chicago weather"

    # 4. Fallback object
    msg_plain = MockMessage("ai", "Hello world")
    action, inputs = extract_action_inputs(msg_plain)
    assert action == "ai"
    assert inputs == "Hello world"


# --- Tests for LangChain Integration ---
def test_langchain_callback_handler():
    guard = LoopGuard(min_repeats=2)
    handler = LoopGuardCallbackHandler(guard)

    # Start tool call 1
    handler.on_tool_start({"name": "web_search"}, "weather chicago")
    handler.on_tool_end("Sunny, 72F")

    assert len(guard.steps_history) == 1
    assert guard.steps_history[0].action == "web_search"
    assert guard.steps_history[0].observation == "Sunny, 72F"

    # Start tool call 2 -> similar query, min_repeats=2 -> raises LoopDetectedError
    with pytest.raises(LoopDetectedError):
        handler.on_tool_start({"name": "web_search"}, "weather in chicago")


# --- Tests for LangGraph Integration ---
def test_langgraph_node_wrapper_normal():
    guard = LoopGuard(min_repeats=3)

    def my_node(state: Dict[str, Any]) -> Dict[str, Any]:
        state["node_executed"] = True
        return state

    wrapped = wrap_node_with_guard(my_node, guard)

    # State with simple message list
    state = {"messages": [{"role": "assistant", "action": "tool_a", "inputs": "abc"}]}
    res = wrapped(state)

    assert res["node_executed"] is True
    assert len(guard.steps_history) == 1


def test_langgraph_node_wrapper_loop_interception():
    guard = LoopGuard(min_repeats=2)

    def my_node(state: Dict[str, Any]) -> Dict[str, Any]:
        state["node_executed"] = True
        return state

    wrapped = wrap_node_with_guard(my_node, guard)

    # Step 1
    state1 = {"messages": [{"role": "assistant", "action": "tool_a", "inputs": "abc"}]}
    wrapped(state1)

    # Step 2 -> Similar state, should trigger loop and append recovery system prompt
    state2 = {"messages": [{"role": "assistant", "action": "tool_a", "inputs": "abc"}]}
    res = wrapped(state2)

    # Node function should NOT execute on loop detection, instead system message gets appended
    assert "node_executed" not in res
    assert len(res["messages"]) == 2
    assert res["messages"][1]["role"] == "system"
    assert (
        "violating" in res["messages"][1]["content"]
        or "repeated" in res["messages"][1]["content"]
    )


# --- Tests for LlamaIndex Integration ---
def test_llamaindex_callback_handler():
    guard = LoopGuard(min_repeats=2)
    handler = LlamaIndexLoopGuardHandler(guard)

    # Event 1 starts
    handler.on_event_start("tool", payload={"name": "my_tool", "input": "input_val"})
    # Event 1 ends
    handler.on_event_end("tool", payload={"response": "output_val"})

    assert len(guard.steps_history) == 1
    assert guard.steps_history[0].action == "my_tool"
    assert guard.steps_history[0].observation == "output_val"

    # Event 2 starts -> similar input, should trigger LoopDetectedError
    with pytest.raises(LoopDetectedError):
        handler.on_event_start(
            "tool", payload={"name": "my_tool", "input": "input_val"}
        )

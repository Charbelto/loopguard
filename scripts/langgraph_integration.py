from typing import Dict, Any
from loopguard import LoopGuard, LoopDetectedError


# Mock representation of LangGraph's StateGraph and MessagesState
class GraphState:
    def __init__(self):
        self.values: Dict[str, Any] = {
            "messages": [],
            "loop_guard": LoopGuard(min_repeats=3, max_cycle_length=2),
        }


def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mock Agent Node generating plan."""
    print("-> Running Agent Node: LLM planning next step...")
    # Simulate agent getting stuck returning the same tool calls
    messages = state["messages"]

    # If a system healing notice exists in messages, agent pivots
    has_notice = any("[SYSTEM NOTICE]" in msg.get("content", "") for msg in messages)

    if has_notice:
        print("   LLM saw the loop healing notice! Pivoting action.")
        return {
            "messages": [
                {
                    "role": "assistant",
                    "action": "request_permissions",
                    "inputs": "chmod 755 config",
                }
            ]
        }

    # Otherwise, it oscillates
    if len(messages) == 0:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "action": "write_config",
                    "inputs": "path=config.json",
                }
            ]
        }

    last_msg = messages[-1]
    if last_msg.get("action") == "check_directory":
        return {
            "messages": [
                {
                    "role": "assistant",
                    "action": "write_config",
                    "inputs": "path=./config.json",
                }
            ]
        }
    else:
        return {
            "messages": [
                {"role": "assistant", "action": "check_directory", "inputs": "dir=."}
            ]
        }


def loop_guard_router(state: Dict[str, Any]) -> str:
    """
    Graph Router: Checks loopguard before sending execution control
    to the tool node. Routes to 'heal_node' if a cycle is found.
    """
    messages = state["messages"]
    guard: LoopGuard = state["loop_guard"]

    if not messages:
        return "continue"

    last_msg = messages[-1]
    action = last_msg.get("action")
    inputs = last_msg.get("inputs")

    if not action:
        return "continue"

    try:
        # Check for loop cycle before running tools
        print(f"   [Router Check] Analyzing action '{action}' with inputs '{inputs}'")
        guard.add_step(action=action, input_text=inputs)
        return "call_tools"
    except LoopDetectedError:
        print("   [Router Intercept] Cycle detected! Routing to heal_node.")
        return "heal_node"


def tools_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mock Tool execution node."""
    messages = state["messages"]
    last_msg = messages[-1]
    action = last_msg["action"]
    last_msg["inputs"]

    print(f"-> Running Tools Node: Executing {action}...")

    # Simulate execution failures causing loop
    if action == "write_config":
        observation = "Error: Permission denied"
    elif action == "check_directory":
        observation = "Info: Directory is owned by root"
    else:
        observation = "Success: Done"

    # Symmetrically update LoopGuard telemetry
    guard: LoopGuard = state["loop_guard"]
    if guard.steps_history:
        guard.steps_history[-1].observation = observation

    return {"messages": [{"role": "tool", "content": observation}]}


def heal_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mock Healing Node that injects LoopGuard prompt back into messages."""
    print("-> Running Healer Node: Generating and injecting recovery prompt...")
    guard: LoopGuard = state["loop_guard"]
    notice = guard.get_healing_prompt()

    # Inject system instruction containing context about the loop
    return {"messages": [{"role": "system", "content": notice}]}


# Simulating Graph execution
if __name__ == "__main__":
    print("==================================================")
    # Instantiate state
    state = {"messages": [], "loop_guard": LoopGuard(min_repeats=3, max_cycle_length=2)}

    # Execution simulation loop
    step = 0
    max_steps = 10
    while step < max_steps:
        # 1. Agent runs
        agent_out = agent_node(state)
        state["messages"].extend(agent_out["messages"])

        # 2. Check routing decision
        route = loop_guard_router(state)

        if route == "call_tools":
            # 3a. Execute tools
            tool_out = tools_node(state)
            state["messages"].extend(tool_out["messages"])
        elif route == "heal_node":
            # 3b. Execute heal node to break the loop
            heal_out = heal_node(state)
            state["messages"].extend(heal_out["messages"])
            # Clear loop history to avoid double triggering immediately
            state["loop_guard"].tracker.state_sequence.clear()
        else:
            # Done
            break

        step += 1
        print("-" * 50)

        # Break condition if task succeeded
        if any(msg.get("action") == "request_permissions" for msg in state["messages"]):
            print("\nTASK COMPLETED: Mock LangGraph recovered and broke the loop!")
            break

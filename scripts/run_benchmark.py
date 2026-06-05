import time
from loopguard import LoopGuard, LoopDetectedError


# Mock Database Agent
class DatabaseAgent:
    def __init__(self, use_loopguard=False):
        self.use_loopguard = use_loopguard
        self.state = "START"
        self.db_locked = True
        self.history = []

    def think(self, last_observation=None, system_notice=None):
        """Mock LLM planning step."""
        if system_notice:
            # Pivots reasoning when notified of loop
            self.state = "HEALED"
            return "reset_connection", "connection_string=db://prod_main"

        if self.state == "START":
            self.state = "QUERYING"
            return "query_user", "select * from users where id=42"

        if self.state == "QUERYING":
            if "lock error" in (last_observation or ""):
                self.state = "WAITING"
                return "wait_database", "seconds=5"

        if self.state == "WAITING":
            self.state = "QUERYING"
            return "query_user", "select * from users where id=42"

        if self.state == "HEALED":
            self.state = "FINISHED"
            return "query_user", "select * from users where id=42"

        return "stop", ""


def execute_tool(action, inputs, db_locked):
    """Mock execution of agent tools."""
    if action == "query_user":
        if db_locked:
            return "Error: Database query failed. Transaction lock error (code 1205)."
        return "Success: Found User {id: 42, name: 'Alice', role: 'admin'}"
    elif action == "wait_database":
        return "Info: Wait finished. DB lock status unchecked."
    elif action == "reset_connection":
        return "Success: Connection pool reset. Stale locks cleared."
    return "Info: Done."


def run_simulation(use_guard):
    agent = DatabaseAgent(use_loopguard=use_guard)
    guard = (
        LoopGuard(similarity_threshold=0.8, min_repeats=3, max_cycle_length=2)
        if use_guard
        else None
    )

    last_obs = None
    system_notice = None
    step_count = 0
    max_steps = 15

    total_input_tokens = 0
    total_output_tokens = 0

    trace = []
    start_time = time.time()

    while step_count < max_steps:
        # Simulate prompt composition size
        prompt_size = 1000 if not system_notice else 1200
        total_input_tokens += prompt_size

        action, inputs = agent.think(
            last_observation=last_obs, system_notice=system_notice
        )
        if action == "stop":
            break

        system_notice = None
        total_output_tokens += 150  # LLM output size

        trace.append(f"Step {step_count + 1}: Agent called '{action}' with '{inputs}'")

        if use_guard:
            try:
                # Intercept before executing the tool
                guard.add_step(action=action, input_text=inputs)

                # Execute the tool
                last_obs = execute_tool(action, inputs, agent.db_locked)
                guard.steps_history[-1].observation = last_obs

            except LoopDetectedError:
                trace.append(">>> [LoopGuard Intercepted execution loop!]")
                system_notice = guard.get_healing_prompt()

                # Resolve the mock block in the environment
                agent.db_locked = False
                continue
        else:
            # Without loopguard
            last_obs = execute_tool(action, inputs, agent.db_locked)

        trace.append(f"          Observation: {last_obs}")
        step_count += 1

    duration = time.time() - start_time
    success = agent.state == "FINISHED"

    # Calculate costs ($0.015 per 1k input tokens, $0.060 per 1k output tokens)
    input_cost = (total_input_tokens / 1000.0) * 0.015
    output_cost = (total_output_tokens / 1000.0) * 0.060
    total_cost = input_cost + output_cost

    return {
        "success": success,
        "steps": step_count,
        "duration_ms": duration * 1000.0,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_cost": total_cost,
        "trace": trace,
    }


def main():
    print("==================================================")
    print("RUNNING BENCHMARK: AGENT EXECUTION CYCLE COMPARISON")
    print("==================================================")

    print("\n--- RUN 1: Standard Agent (Without LoopGuard) ---")
    result_standard = run_simulation(use_guard=False)
    for line in result_standard["trace"]:
        print(line)
    print(
        f"Outcome: {'SUCCESS' if result_standard['success'] else 'FAILURE (Max steps reached)'}"
    )

    print("\n--- RUN 2: Protected Agent (With LoopGuard) ---")
    result_protected = run_simulation(use_guard=True)
    for line in result_protected["trace"]:
        print(line)
    print(f"Outcome: {'SUCCESS' if result_protected['success'] else 'FAILURE'}")

    print("\n==================================================")
    print("COMPARATIVE METRICS SUMMARY")
    print("==================================================")
    print(f"{'Metric':<25} | {'Without LoopGuard':<20} | {'With LoopGuard':<20}")
    print("-" * 75)
    print(f"{'Task Outcome':<25} | {'TIMEOUT / FAIL':<20} | {'SUCCESS':<20}")
    print(
        f"{'Total Execution Steps':<25} | {result_standard['steps']:<20} | {result_protected['steps']:<20}"
    )
    print(
        f"{'Input Tokens Used':<25} | {result_standard['input_tokens']:<20} | {result_protected['input_tokens']:<20}"
    )
    print(
        f"{'Output Tokens Used':<25} | {result_standard['output_tokens']:<20} | {result_protected['output_tokens']:<20}"
    )
    print(
        f"{'Total API Cost ($)':<25} | ${result_standard['total_cost']:<19.4f} | ${result_protected['total_cost']:<19.4f}"
    )

    cost_saved = result_standard["total_cost"] - result_protected["total_cost"]
    percent_saved = (
        (cost_saved / result_standard["total_cost"]) * 100.0
        if result_standard["total_cost"]
        else 0
    )
    print("-" * 75)
    print(
        f"Efficiency Improvement: {percent_saved:.1f}% Cost Reduction (Saved ${cost_saved:.4f})"
    )
    print("==================================================")


if __name__ == "__main__":
    main()

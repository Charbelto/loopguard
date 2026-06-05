from loopguard.core import LoopGuard, LoopDetectedError


# Mock Agent state machine representation
class MockAgent:
    def __init__(self):
        self.state = "START"
        self.history = []
        self.permission_granted = False

    def act(self, last_observation=None, system_notice=None) -> tuple:
        """Determines next action and input based on state and inputs."""
        if system_notice:
            # If agent receives a system notice / healing prompt, it redirects its execution path
            self.state = "HEALED"
            return "request_permissions", "sudo chmod 755 config_dir"

        if self.state == "START":
            self.state = "TRYING_WRITE"
            return "write_config", "path=config_dir/app.conf, data={port: 8080}"

        if self.state == "TRYING_WRITE":
            if "Permission denied" in (last_observation or ""):
                self.state = "CHECKING_DIR"
                return "check_directory", "dir=config_dir"

        if self.state == "CHECKING_DIR":
            self.state = "TRYING_WRITE_AGAIN"
            return "write_config", "path=./config_dir/app.conf, data={port: 8080}"

        if self.state == "TRYING_WRITE_AGAIN":
            if "Permission denied" in (last_observation or ""):
                self.state = "CHECKING_DIR_AGAIN"
                return "check_directory", "dir=./config_dir"

        if self.state == "CHECKING_DIR_AGAIN":
            self.state = "TRYING_WRITE_YET_AGAIN"
            return (
                "write_config",
                "path=config_dir/app.conf, data={port:8080}",
            )  # compact JSON

        if self.state == "TRYING_WRITE_YET_AGAIN":
            if "Permission denied" in (last_observation or ""):
                self.state = "CHECKING_DIR_YET_AGAIN"
                return "check_directory", "dir=config_dir/"

        return "stop", ""


def execute_mock_tool(action, input_text, permission_granted=False):
    """Simulates tool execution outcomes."""
    if action == "write_config":
        if not permission_granted:
            return "Error: Permission denied (write config failed)"
        return "Success: File written"
    elif action == "check_directory":
        return "Info: Directory exists, owner is root"
    elif action == "request_permissions":
        return "Success: Permissions updated to 755"
    return "Unknown action"


def test_agent_loop_interception_and_recovery():
    agent = MockAgent()
    guard = LoopGuard(similarity_threshold=0.8, min_repeats=3, max_cycle_length=2)

    last_obs = None
    system_notice = None
    step_count = 0
    max_steps = 15
    loop_intercepted = False

    while step_count < max_steps:
        # 1. Ask agent for action
        action, args = agent.act(last_observation=last_obs, system_notice=system_notice)
        if action == "stop":
            break

        # Clear system notice once processed
        system_notice = None

        # 2. Add step to loop guard to check for cycles
        try:
            guard.add_step(
                action=action, input_text=args, observation=None
            )  # Add before running tool

            # Run the mock tool
            last_obs = execute_mock_tool(
                action, args, permission_granted=agent.permission_granted
            )

            # Update the observation in loopguard step history (simulating real telemetry flow)
            guard.steps_history[-1].observation = last_obs
            # Update state tracker's stored TF-IDF fit
            text_with_obs = f"{args} {last_obs}"
            guard.tracker.tfidf.fit_step(text_with_obs)

        except LoopDetectedError:
            # 3. Intercept loop, extract healing prompt and continue execution safely
            loop_intercepted = True
            system_notice = guard.get_healing_prompt()

            # Perform mitigation action in the mock environment
            agent.permission_granted = True

            # The loop is not broken by throwing an unhandled exception anymore;
            # the agent continues with the healing notice injected
            continue

        step_count += 1

    # Verify that the loop guard successfully caught the cycle
    assert loop_intercepted

    # Verify that the agent adjusted path and successfully completed the task
    assert agent.permission_granted
    assert agent.state == "HEALED"
    assert step_count < max_steps

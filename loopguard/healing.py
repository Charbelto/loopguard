from typing import List
from .rules import (
    BaseRule,
    CycleDetectionRule,
    MaxConsecutiveSimilarityRule,
    ActionOscillationRule,
    StateFrequencyRule,
    ResourceLimitRule,
    RegexGuardRule,
    JSONGuardRule,
)
from .models import AgentStep


class HealingStrategy:
    """Generates structured instructions to guide LLMs out of execution loops."""

    def format_prompt(self, rule: BaseRule, steps: List[AgentStep]) -> str:
        """
        Format a mitigation prompt based on the specific rule that was violated.

        Args:
            rule: The violated BaseRule.
            steps: The list of agent steps leading up to the violation.

        Returns:
            A string containing feedback/instructions for the agent.
        """
        if not steps:
            return ""

        last_step = steps[-1]
        base = (
            f"\n[SYSTEM NOTICE]: Telemetry guardrail rule '{rule.name}' was violated."
        )

        if isinstance(rule, CycleDetectionRule):
            # Formatted to support existing assertions looking for action name & input text
            return (
                f"\n[SYSTEM NOTICE]: You have repeated the action '{last_step.action}' with inputs/results "
                f"similar to '{last_step.input_text[:120]}' multiple times. This indicates you are stuck in an execution cycle. "
                "Do not retry this action again with similar parameters. Try a completely different reasoning path "
                "or check for alternative tools to achieve your goal."
            )
        elif isinstance(rule, MaxConsecutiveSimilarityRule):
            return (
                f"{base} You have consecutively run the action '{last_step.action}' with highly similar inputs: "
                f"'{last_step.input_text[:80]}...'. "
                "Continuing this will not change the outcome. Refactor your inputs or try another command."
            )
        elif isinstance(rule, ActionOscillationRule):
            return (
                f"{base} An oscillation between actions was detected (e.g. ping-ponging between tools). "
                "Break this loop. Inspect why the actions are repeating and alter your workflow direction."
            )
        elif isinstance(rule, StateFrequencyRule):
            return (
                f"{base} The execution has repeatedly returned to the same system state configuration. "
                "You are not making forward progress. Pivot to a new strategy or check for failures."
            )
        elif isinstance(rule, ResourceLimitRule):
            return (
                f"{base} Execution resources (steps or duration limits) are exhausted. "
                "Please finalize execution immediately, compile your findings, and present the final status."
            )
        elif isinstance(rule, RegexGuardRule):
            return (
                f"{base} The {rule.target} content violates text pattern rules. "
                "Adjust your input/output format to ensure compliance with the rules."
            )
        elif isinstance(rule, JSONGuardRule):
            return (
                f"{base} The {rule.target} field must be valid JSON but syntax errors were found. "
                "Verify your json format, escape quotes properly, and re-output valid json syntax."
            )

        return (
            f"{base} Please review your execution history, alter your reasoning path, "
            "and try a different strategy."
        )

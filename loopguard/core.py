from typing import List, Callable, Optional, Dict, Any
from .models import AgentStep
from .graph import StateTracker
from .rules import BaseRule, CycleDetectionRule
from .healing import HealingStrategy


class LoopDetectedError(Exception):
    """Exception raised when an agentic execution loop or guardrail rule is violated."""

    def __init__(
        self,
        message: str,
        violated_rule: Optional[BaseRule] = None,
        cycle: Optional[List[int]] = None,
        repeat_count: int = 0,
    ):
        super().__init__(message)
        self.message = message
        self.violated_rule = violated_rule
        # For backward compatibility
        self.cycle = cycle or []
        self.repeat_count = repeat_count


class LoopGuard:
    """The developer interface class coordinating LoopGuard's telemetry."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        min_repeats: int = 3,
        max_cycle_length: int = 5,
        max_history: int = 100,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_fn: Optional[Callable[[AgentStep, AgentStep], float]] = None,
        on_loop_detected: Optional[Callable[..., None]] = None,
        rules: Optional[List[BaseRule]] = None,
    ):
        self.tracker = StateTracker(
            similarity_threshold=similarity_threshold,
            max_history=max_history,
            embedding_fn=embedding_fn,
            similarity_fn=similarity_fn,
        )
        self.min_repeats = min_repeats
        self.max_cycle_length = max_cycle_length
        self.max_history = max_history
        self.on_loop_detected = on_loop_detected
        self.steps_history: List[AgentStep] = []
        self.healing_strategy = HealingStrategy()

        if rules is None:
            # Default backwards compatibility cycle checking rule
            self.rules: List[BaseRule] = [
                CycleDetectionRule(
                    min_length=1,
                    max_length=self.max_cycle_length,
                    min_repeats=self.min_repeats,
                )
            ]
        else:
            self.rules = list(rules)

    def add_rule(self, rule: BaseRule) -> None:
        """Register a new guardrail constraint rule."""
        self.rules.append(rule)

    def add_step(
        self,
        action: str,
        input_text: str,
        observation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record an execution step.
        Returns True if safe, raises LoopDetectedError if a loop or rule violation is detected.
        """
        step = AgentStep(
            action=action,
            input_text=input_text,
            observation=observation,
            metadata=metadata or {},
        )
        self.steps_history.append(step)
        if len(self.steps_history) > self.max_history:
            self.steps_history = self.steps_history[-self.max_history :]
        self.tracker.add_step(step)

        # Evaluate all registered rules
        for rule in self.rules:
            if rule.check(
                self.steps_history, self.tracker.state_sequence, self.tracker
            ):
                message = (
                    f"Loop detected: Rule '{rule.name}' violated. {rule.description}."
                )

                cycle = getattr(rule, "violated_cycle", None)
                repeats = getattr(rule, "min_repeats", 0) if cycle else 0

                if self.on_loop_detected:
                    if isinstance(rule, CycleDetectionRule):
                        try:
                            # Try old signature first: on_loop_detected(cycle, repeats)
                            self.on_loop_detected(cycle, repeats)
                        except TypeError:
                            # Fallback if they expect (rule, message)
                            try:
                                self.on_loop_detected(rule, message)
                            except TypeError:
                                pass
                    else:
                        try:
                            # Try passing rule and message
                            self.on_loop_detected(rule, message)
                        except TypeError:
                            # Fallback to (cycle, repeats) signature
                            try:
                                self.on_loop_detected([], 0)
                            except TypeError:
                                pass

                raise LoopDetectedError(
                    message=message,
                    violated_rule=rule,
                    cycle=cycle,
                    repeat_count=repeats,
                )

        return True

    def get_healing_prompt(self, violated_rule: Optional[BaseRule] = None) -> str:
        """Get a system instructions prompt to guide the LLM out of its execution loop."""
        if not self.steps_history:
            return ""

        # If no specific rule passed, look for the first rule that currently violates history
        rule_to_heal = violated_rule
        if rule_to_heal is None:
            for rule in self.rules:
                if rule.check(
                    self.steps_history, self.tracker.state_sequence, self.tracker
                ):
                    rule_to_heal = rule
                    break

        if rule_to_heal is None:
            # Fallback to general formatting using the first rule or default
            rule_to_heal = self.rules[0] if self.rules else CycleDetectionRule()

        return self.healing_strategy.format_prompt(rule_to_heal, self.steps_history)

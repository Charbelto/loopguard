import time
from loopguard.models import AgentStep
from loopguard.graph import StateTracker
from loopguard.rules import (
    CycleDetectionRule,
    MaxConsecutiveSimilarityRule,
    ActionOscillationRule,
    StateFrequencyRule,
    ResourceLimitRule,
    RegexGuardRule,
    JSONGuardRule,
)


def test_cycle_detection_rule():
    tracker = StateTracker()
    rule = CycleDetectionRule(min_length=1, max_length=2, min_repeats=2)

    # Sequence of states: [1, 2, 1, 2] -> repeating cycle [1, 2]
    step1 = AgentStep(action="action_a", input_text="x")
    step2 = AgentStep(action="action_b", input_text="y")

    tracker.add_step(step1)
    tracker.add_step(step2)
    assert not rule.check([step1, step2], tracker.state_sequence, tracker)

    tracker.add_step(step1)
    assert not rule.check([step1, step2, step1], tracker.state_sequence, tracker)

    tracker.add_step(step2)
    assert rule.check([step1, step2, step1, step2], tracker.state_sequence, tracker)
    assert rule.violated_cycle == [1, 2]


def test_max_consecutive_similarity_rule():
    tracker = StateTracker()
    rule = MaxConsecutiveSimilarityRule(max_consecutive=3)

    step = AgentStep(action="query", input_text="weather")

    # Add 1st
    tracker.add_step(step)
    assert not rule.check([step], tracker.state_sequence, tracker)

    # Add 2nd
    tracker.add_step(step)
    assert not rule.check([step, step], tracker.state_sequence, tracker)

    # Add 3rd -> violation
    tracker.add_step(step)
    assert rule.check([step, step, step], tracker.state_sequence, tracker)


def test_max_consecutive_similarity_rule_with_action():
    tracker = StateTracker()
    rule = MaxConsecutiveSimilarityRule(action="web_search", max_consecutive=2)

    step_search = AgentStep(action="web_search", input_text="query")
    step_other = AgentStep(action="database", input_text="query")

    # Same semantic state (query matches query, but action is different)
    tracker.add_step(step_search)
    tracker.add_step(step_other)

    # It shouldn't violate because step_other action is different
    assert not rule.check([step_search, step_other], tracker.state_sequence, tracker)

    # Clean tracker
    tracker2 = StateTracker()
    tracker2.add_step(step_search)
    tracker2.add_step(step_search)
    assert rule.check([step_search, step_search], tracker2.state_sequence, tracker2)


def test_action_oscillation_rule():
    tracker = StateTracker()
    rule = ActionOscillationRule(max_oscillations=3, window_size=6)

    # Action sequence: A, B, A, B, A, B -> oscillation
    steps = [
        AgentStep(action="A", input_text="1"),
        AgentStep(action="B", input_text="2"),
        AgentStep(action="A", input_text="3"),
        AgentStep(action="B", input_text="4"),
        AgentStep(action="A", input_text="5"),
        AgentStep(action="B", input_text="6"),
    ]

    for step in steps:
        tracker.add_step(step)

    assert rule.check(steps, tracker.state_sequence, tracker)


def test_state_frequency_rule():
    tracker = StateTracker()
    rule = StateFrequencyRule(max_frequency=2, window_size=5)

    step_a = AgentStep(action="action", input_text="alpha")  # state 1
    step_b = AgentStep(action="action", input_text="beta")  # state 2

    # Sequence: alpha, beta, alpha, beta -> alpha appears 2 times, beta 2 times
    steps = [step_a, step_b, step_a, step_b]
    for s in steps:
        tracker.add_step(s)
    assert not rule.check(steps, tracker.state_sequence, tracker)

    # Let's add one more alpha -> sequence is [1, 2, 1, 2, 1], alpha appears 3 times in window of 5 -> violation
    steps.append(step_a)
    tracker.add_step(step_a)
    assert rule.check(steps, tracker.state_sequence, tracker)


def test_resource_limit_rule():
    rule_steps = ResourceLimitRule(max_steps=3)
    rule_time = ResourceLimitRule(max_duration_seconds=0.1)

    steps = [
        AgentStep(action="A", input_text="1"),
        AgentStep(action="A", input_text="2"),
    ]
    # Step limit not hit
    assert not rule_steps.check(steps, [1, 2], None)

    # Step limit hit
    steps.append(AgentStep(action="A", input_text="3"))
    assert rule_steps.check(steps, [1, 2, 3], None)

    # Time limit not hit yet
    assert not rule_time.check(steps[:1], [1], None)

    # Wait for time limit to expire
    time.sleep(0.15)
    assert rule_time.check(steps[:1], [1], None)


def test_regex_guard_rule():
    rule_block = RegexGuardRule(pattern="ERROR", target="observation", block=True)
    rule_allow = RegexGuardRule(pattern="SUCCESS", target="observation", block=False)

    step_err = AgentStep(
        action="tool", input_text="x", observation="Something failed: ERROR code 5"
    )
    step_ok = AgentStep(
        action="tool", input_text="x", observation="Operation SUCCESS details"
    )

    # Blocklist check
    assert rule_block.check(
        [step_err], [1], None
    )  # Violates block list (contains ERROR)
    assert not rule_block.check([step_ok], [1], None)  # Safe (no ERROR)

    # Allowlist check
    assert not rule_allow.check([step_ok], [1], None)  # Safe (contains SUCCESS)
    assert rule_allow.check([step_err], [1], None)  # Violates allow list (no SUCCESS)


def test_json_guard_rule():
    rule = JSONGuardRule(target="observation")

    step_valid = AgentStep(
        action="tool", input_text="x", observation='{"status": "ok", "code": 200}'
    )
    step_invalid = AgentStep(
        action="tool", input_text="x", observation="status: ok, code: 200 (Not JSON)"
    )
    step_none = AgentStep(action="tool", input_text="x", observation=None)

    assert not rule.check([step_valid], [1], None)  # Valid JSON, no violation
    assert rule.check([step_invalid], [1], None)  # Invalid JSON, violation!
    assert not rule.check([step_none], [1], None)  # None field is skipped, no violation

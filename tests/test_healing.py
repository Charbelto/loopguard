from loopguard.models import AgentStep
from loopguard.healing import HealingStrategy
from loopguard.rules import (
    CycleDetectionRule,
    MaxConsecutiveSimilarityRule,
    ActionOscillationRule,
    StateFrequencyRule,
    ResourceLimitRule,
    RegexGuardRule,
    JSONGuardRule,
)


def test_healing_strategy_formatting():
    healer = HealingStrategy()
    step = AgentStep(action="write_file", input_text="app.py", observation="Error")
    steps = [step]

    # 1. CycleDetection
    rule_cycle = CycleDetectionRule()
    rule_cycle.violated_cycle = [1, 2]
    prompt_cycle = healer.format_prompt(rule_cycle, steps)
    assert "[SYSTEM NOTICE]" in prompt_cycle
    assert "write_file" in prompt_cycle
    assert "app.py" in prompt_cycle

    # 2. MaxConsecutiveSimilarity
    rule_sim = MaxConsecutiveSimilarityRule(action="write_file")
    prompt_sim = healer.format_prompt(rule_sim, steps)
    assert "MaxConsecutiveSimilarity" in prompt_sim
    assert "consecutively run the action 'write_file'" in prompt_sim

    # 3. ActionOscillation
    rule_osc = ActionOscillationRule()
    prompt_osc = healer.format_prompt(rule_osc, steps)
    assert "ActionOscillation" in prompt_osc
    assert "oscillation between actions" in prompt_osc

    # 4. StateFrequency
    rule_freq = StateFrequencyRule()
    prompt_freq = healer.format_prompt(rule_freq, steps)
    assert "StateFrequency" in prompt_freq
    assert "repeatedly returned to the same system state" in prompt_freq

    # 5. ResourceLimit
    rule_res = ResourceLimitRule(max_steps=5)
    prompt_res = healer.format_prompt(rule_res, steps)
    assert "ResourceLimit" in prompt_res
    assert "Execution resources" in prompt_res

    # 6. RegexGuard
    rule_regex = RegexGuardRule(pattern="Error")
    prompt_regex = healer.format_prompt(rule_regex, steps)
    assert "RegexGuard" in prompt_regex
    assert "violates text pattern rules" in prompt_regex

    # 7. JSONGuard
    rule_json = JSONGuardRule()
    prompt_json = healer.format_prompt(rule_json, steps)
    assert "JSONGuard" in prompt_json
    assert "must be valid JSON" in prompt_json


def test_healing_strategy_empty_steps():
    healer = HealingStrategy()
    rule = CycleDetectionRule()
    assert healer.format_prompt(rule, []) == ""

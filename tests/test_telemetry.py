from loopguard.models import AgentStep
from loopguard.graph import StateTracker
from loopguard.telemetry import TelemetryVisualizer


def test_telemetry_visualizer_mermaid_empty():
    tracker = StateTracker()
    mermaid_out = TelemetryVisualizer.to_mermaid_flowchart(tracker)
    assert "Empty[No states tracked]" in mermaid_out


def test_telemetry_visualizer_mermaid_flowchart():
    tracker = StateTracker()
    step1 = AgentStep(action="web_search", input_text="chicago weather")
    step2 = AgentStep(action="write_file", input_text="log.txt")

    tracker.add_step(step1)
    tracker.add_step(step2)
    tracker.add_step(step1)  # transitions: S1 -> S2 -> S1

    mermaid_out = TelemetryVisualizer.to_mermaid_flowchart(tracker)
    assert "flowchart TD" in mermaid_out
    assert 'S1["State 1: web_search (chicago weather...)"' in mermaid_out
    assert 'S2["State 2: write_file (log.txt...)"' in mermaid_out
    assert "S1 --> S2" in mermaid_out
    assert "S2 --> S1" in mermaid_out


def test_telemetry_visualizer_to_dict():
    step1 = AgentStep(
        action="web_search", input_text="chicago weather", observation="sunny"
    )
    step2 = AgentStep(action="write_file", input_text="log.txt")
    steps = [step1, step2]
    seq = [1, 2]

    telemetry_dict = TelemetryVisualizer.to_dict(steps, seq)
    assert "steps" in telemetry_dict
    assert len(telemetry_dict["steps"]) == 2
    assert telemetry_dict["steps"][0]["step_index"] == 0
    assert telemetry_dict["steps"][0]["action"] == "web_search"
    assert telemetry_dict["steps"][0]["input_text"] == "chicago weather"
    assert telemetry_dict["steps"][0]["observation"] == "sunny"
    assert telemetry_dict["steps"][0]["state_id"] == 1
    assert telemetry_dict["steps"][1]["state_id"] == 2

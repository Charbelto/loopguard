from typing import List, Dict, Any
from .graph import StateTracker
from .models import AgentStep


class TelemetryVisualizer:
    """Helper class to visualize loopguard graphs and export execution histories."""

    @staticmethod
    def to_mermaid_flowchart(tracker: StateTracker) -> str:
        """
        Generates a Mermaid.js flowchart of the states and transitions.

        Args:
            tracker: The StateTracker instance tracking states.

        Returns:
            A string containing Mermaid.js markup.
        """
        if not tracker.states:
            return "graph TD\n  Empty[No states tracked]"

        lines = ["flowchart TD"]
        # Nodes
        for state in tracker.states:
            rep = state.representative_step
            # Clean label to avoid quotes/formatting issues
            text_preview = rep.input_text[:40].replace('"', '\\"').replace("\n", " ")
            label = f"State {state.id}: {rep.action} ({text_preview}...)"
            lines.append(f'  S{state.id}["{label}"]')

        # Transitions
        seq = tracker.state_sequence
        transitions: Dict[tuple, int] = {}
        for i in range(len(seq) - 1):
            t = (seq[i], seq[i + 1])
            transitions[t] = transitions.get(t, 0) + 1

        for (src, dest), count in transitions.items():
            if count > 1:
                lines.append(f"  S{src} -->|x{count}| S{dest}")
            else:
                lines.append(f"  S{src} --> S{dest}")

        return "\n".join(lines)

    @staticmethod
    def to_dict(steps: List[AgentStep], state_sequence: List[int]) -> Dict[str, Any]:
        """
        Exports history in a structured representation.

        Args:
            steps: The raw AgentStep history.
            state_sequence: The computed sequence of State IDs.

        Returns:
            A dictionary of the execution telemetry log.
        """
        return {
            "steps": [
                {
                    "step_index": idx,
                    "action": step.action,
                    "input_text": step.input_text,
                    "observation": step.observation,
                    "metadata": step.metadata,
                    "state_id": (
                        state_sequence[idx] if idx < len(state_sequence) else None
                    ),
                }
                for idx, step in enumerate(steps)
            ]
        }

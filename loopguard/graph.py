import math
from typing import List, Tuple, Optional, Callable
from .models import AgentStep, State
from .similarity import TFIDFSimilarity


class StateTracker:
    """Tracks agent steps, maps them to semantic states, and runs cycle detection."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_fn = embedding_fn
        self.tfidf = TFIDFSimilarity()
        self.states: List[State] = []
        self.state_sequence: List[int] = []

    def _get_similarity(self, step1: AgentStep, step2: AgentStep) -> float:
        """Calculate similarity between two agent steps."""
        # The actions (e.g. tool name) must match exactly
        if step1.action != step2.action:
            return 0.0

        # Combine input and observation for evaluation, but only include observation if both have it
        if step1.observation and step2.observation:
            text1 = f"{step1.input_text} {step1.observation}"
            text2 = f"{step2.input_text} {step2.observation}"
        else:
            text1 = step1.input_text
            text2 = step2.input_text

        if self.embedding_fn:
            # Calculate Cosine similarity of embedded vector lists
            v1 = self.embedding_fn(text1)
            v2 = self.embedding_fn(text2)
            dot = sum(a * b for a, b in zip(v1, v2))
            n1 = math.sqrt(sum(a**2 for a in v1))
            n2 = math.sqrt(sum(b**2 for b in v2))
            return dot / (n1 * n2) if n1 and n2 else 0.0

        # Fallback to pure-Python TF-IDF similarity
        return self.tfidf.cosine_similarity(text1, text2)

    def add_step(self, step: AgentStep) -> int:
        """Add step to tracker history, cluster it to a State, and return State ID."""
        text_content = f"{step.input_text} {step.observation or ''}"
        self.tfidf.fit_step(text_content)

        matched_state: Optional[State] = None
        for state in self.states:
            sim = self._get_similarity(step, state.representative_step)
            if sim >= self.similarity_threshold:
                matched_state = state
                break

        if matched_state:
            matched_state.steps_count += 1
            state_id = matched_state.id
        else:
            state_id = len(self.states) + 1
            new_state = State(id=state_id, representative_step=step)
            self.states.append(new_state)

        self.state_sequence.append(state_id)
        return state_id

    def check_cycles(
        self, min_length: int = 1, max_length: int = 5, min_repeats: int = 3
    ) -> Tuple[Optional[List[int]], int]:
        """
        Check if the state transition history ends with a repeating sequence.
        Returns:
            Tuple[cycle_sequence, cycle_length] or (None, 0)
        """
        seq = self.state_sequence
        n = len(seq)
        for length in range(min_length, max_length + 1):
            if length * min_repeats > n:
                continue

            sub = seq[-length:]
            is_loop = True
            for r in range(1, min_repeats):
                start = -length * (r + 1)
                end = -length * r
                if end == 0:
                    chunk = seq[start:]
                else:
                    chunk = seq[start:end]
                if chunk != sub:
                    is_loop = False
                    break

            if is_loop:
                return sub, length
        return None, 0

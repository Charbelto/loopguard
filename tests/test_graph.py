from loopguard.models import AgentStep
from loopguard.graph import StateTracker


def test_state_clustering_identical_action():
    tracker = StateTracker(similarity_threshold=0.6)

    step1 = AgentStep(
        action="web_search", input_text="weather in chicago", observation="sunny"
    )
    step2 = AgentStep(
        action="web_search",
        input_text="chicago weather tomorrow",
        observation="sunny and warm",
    )
    step3 = AgentStep(
        action="database_query", input_text="weather in chicago", observation="sunny"
    )

    id1 = tracker.add_step(step1)
    id2 = tracker.add_step(step2)
    id3 = tracker.add_step(step3)

    # id1 and id2 have the same action and similar text, should cluster together
    assert id1 == id2

    # id3 has a different action, should get a new state ID
    assert id3 != id1
    assert len(tracker.states) == 2


def test_state_clustering_threshold_sensitivity():
    # High threshold: shouldn't cluster slightly different queries
    tracker_high = StateTracker(similarity_threshold=0.95)
    step1 = AgentStep(
        action="web_search", input_text="weather in chicago", observation="sunny"
    )
    step2 = AgentStep(
        action="web_search",
        input_text="chicago weather tomorrow",
        observation="sunny and warm",
    )

    id1 = tracker_high.add_step(step1)
    id2 = tracker_high.add_step(step2)
    assert id1 != id2

    # Low threshold: should cluster them
    tracker_low = StateTracker(similarity_threshold=0.4)
    id1_l = tracker_low.add_step(step1)
    id2_l = tracker_low.add_step(step2)
    assert id1_l == id2_l


def test_cycle_detection_length_1():
    tracker = StateTracker()
    tracker.state_sequence = [1, 2, 3, 3, 3]  # 3 repeats of state '3'
    cycle, length = tracker.check_cycles(min_repeats=3, min_length=1, max_length=5)
    assert cycle == [3]
    assert length == 1

    tracker.state_sequence = [1, 2, 3, 3]  # Only 2 repeats of state '3'
    cycle, length = tracker.check_cycles(min_repeats=3, min_length=1, max_length=5)
    assert cycle is None
    assert length == 0


def test_cycle_detection_length_2():
    tracker = StateTracker()
    # repeats: [1, 2], [1, 2], [1, 2]
    tracker.state_sequence = [5, 1, 2, 1, 2, 1, 2]
    cycle, length = tracker.check_cycles(min_repeats=3, min_length=1, max_length=5)
    assert cycle == [1, 2]
    assert length == 2

    # Incomplete repeats
    tracker.state_sequence = [5, 1, 2, 1, 2, 1]
    cycle, length = tracker.check_cycles(min_repeats=3, min_length=1, max_length=5)
    assert cycle is None


def test_cycle_detection_length_3():
    tracker = StateTracker()
    # repeats: [10, 20, 30], [10, 20, 30], [10, 20, 30]
    tracker.state_sequence = [10, 20, 30, 10, 20, 30, 10, 20, 30]
    cycle, length = tracker.check_cycles(min_repeats=3, min_length=1, max_length=5)
    assert cycle == [10, 20, 30]
    assert length == 3


def test_cycle_detection_custom_min_repeats():
    tracker = StateTracker()
    tracker.state_sequence = [1, 2, 1, 2]  # 2 repeats of [1, 2]

    # Should find cycle if min_repeats is 2
    cycle, length = tracker.check_cycles(min_repeats=2, min_length=1, max_length=5)
    assert cycle == [1, 2]
    assert length == 2

    # Should NOT find cycle if min_repeats is 3
    cycle, length = tracker.check_cycles(min_repeats=3, min_length=1, max_length=5)
    assert cycle is None


def test_state_tracker_max_history():
    tracker = StateTracker(max_history=3)
    step1 = AgentStep(action="a", input_text="x")
    step2 = AgentStep(action="b", input_text="y")
    step3 = AgentStep(action="c", input_text="z")
    step4 = AgentStep(action="d", input_text="w")

    tracker.add_step(step1)
    tracker.add_step(step2)
    tracker.add_step(step3)
    tracker.add_step(step4)

    # Sequence should be truncated to length 3
    assert len(tracker.state_sequence) == 3
    assert tracker.state_sequence == [2, 3, 4]


def test_state_tracker_custom_similarity_fn():
    # Define a custom similarity function checking character difference length
    def length_diff_similarity(s1, s2):
        if abs(len(s1.input_text) - len(s2.input_text)) <= 2:
            return 1.0
        return 0.0

    tracker = StateTracker(
        similarity_threshold=0.8, similarity_fn=length_diff_similarity
    )
    step1 = AgentStep(action="a", input_text="abc")
    step2 = AgentStep(action="b", input_text="abcd")  # diff of 1, matches!
    step3 = AgentStep(action="a", input_text="abcdefgh")  # diff of 5, doesn't match!

    id1 = tracker.add_step(step1)
    id2 = tracker.add_step(step2)
    id3 = tracker.add_step(step3)

    assert id1 == id2
    assert id3 != id1

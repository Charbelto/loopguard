import pytest
from loopguard.core import LoopGuard, LoopDetectedError


def test_loopguard_safe_execution():
    guard = LoopGuard(min_repeats=3)

    # Run multiple steps that are different
    assert guard.add_step(action="web_search", input_text="chicago weather")
    assert guard.add_step(action="database_query", input_text="select users")
    assert guard.add_step(action="web_search", input_text="london weather")
    assert guard.add_step(action="file_write", input_text="done.txt")


def test_loopguard_detects_self_loop():
    guard = LoopGuard(similarity_threshold=0.7, min_repeats=3)

    guard.add_step(action="web_search", input_text="weather chicago")
    guard.add_step(action="web_search", input_text="weather in chicago today")

    with pytest.raises(LoopDetectedError) as exc_info:
        guard.add_step(action="web_search", input_text="weather now chicago")

    assert exc_info.value.cycle == [1]
    assert exc_info.value.repeat_count == 3


def test_loopguard_callback():
    callback_called = False
    callback_cycle = None
    callback_repeats = None

    def on_loop(cycle, repeats):
        nonlocal callback_called, callback_cycle, callback_repeats
        callback_called = True
        callback_cycle = cycle
        callback_repeats = repeats

    guard = LoopGuard(min_repeats=3, on_loop_detected=on_loop)

    guard.add_step(action="query", input_text="x")
    guard.add_step(action="query", input_text="x")

    try:
        guard.add_step(action="query", input_text="x")
    except LoopDetectedError:
        pass

    assert callback_called
    assert callback_cycle == [1]
    assert callback_repeats == 3


def test_loopguard_embedding_injection():
    # Inject a simple mock embedding function
    # In this mock, we returns vectors based on whether text contains "search"
    def dummy_embedding(text: str):
        if "search" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    guard = LoopGuard(min_repeats=2, embedding_fn=dummy_embedding)

    # Add step 1
    guard.add_step(action="web_search", input_text="search query A")

    # Step 2 has same action and similar mock embedding, should loop with min_repeats=2
    with pytest.raises(LoopDetectedError):
        guard.add_step(action="web_search", input_text="search query B")


def test_healing_prompt():
    guard = LoopGuard()
    assert guard.get_healing_prompt() == ""

    guard.add_step(
        action="file_write", input_text="data.json", observation="Permission denied"
    )
    prompt = guard.get_healing_prompt()

    assert "[SYSTEM NOTICE]" in prompt
    assert "file_write" in prompt
    assert "data.json" in prompt

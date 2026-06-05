try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    # Fallback mock class for demonstration if LangChain is not installed
    class BaseCallbackHandler:
        pass

from loopguard import LoopGuard, LoopDetectedError

class LoopGuardCallbackHandler(BaseCallbackHandler):
    """
    A LangChain-compatible callback handler that hooks into tool start/end
    lifecycles to prevent repeating execution patterns.
    """
    def __init__(self, similarity_threshold=0.8, min_repeats=3, max_cycle_length=3):
        self.guard = LoopGuard(
            similarity_threshold=similarity_threshold,
            min_repeats=min_repeats,
            max_cycle_length=max_cycle_length
        )

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Runs right before a tool is invoked."""
        tool_name = serialized.get("name", "unknown_tool")
        
        # Log step in LoopGuard and check for cycles
        try:
            self.guard.add_step(action=tool_name, input_text=input_str)
        except LoopDetectedError as e:
            # We intercept and raise a clean error which breaks the chain
            # and exposes the recovery instructions.
            print(f"\n[LoopGuard Callback Intercepted Loop!]")
            print(f"Detected repeating cycle: {e.cycle} after {e.repeat_count} iterations.")
            raise e

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Runs after a tool completes execution."""
        # Append the observation/result to the current step representation
        if self.guard.steps_history:
            self.guard.steps_history[-1].observation = str(output)
            # Re-fit the vocabulary to include the new observation
            text_to_fit = f"{self.guard.steps_history[-1].input_text} {output}"
            self.guard.tracker.tfidf.fit_step(text_to_fit)


# Demo run mapping
if __name__ == "__main__":
    print("LoopGuard LangChain Callback Handler loaded successfully.")
    
    # Mocking LangChain lifecycle calls
    handler = LoopGuardCallbackHandler(similarity_threshold=0.7, min_repeats=2)
    
    # Simulate step 1
    print("\nSimulating Step 1 (Search weather):")
    handler.on_tool_start({"name": "web_search"}, "weather chicago")
    handler.on_tool_end("Sunny, 72F")
    
    # Simulate step 2
    print("Simulating Step 2 (Search weather - rephrased):")
    try:
        handler.on_tool_start({"name": "web_search"}, "weather in chicago")
        handler.on_tool_end("Sunny, 72F")
    except LoopDetectedError:
        print("Loop detected successfully on step 2!")

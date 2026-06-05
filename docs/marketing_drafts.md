# LoopGuard Launch Marketing Materials

Use these pre-formatted drafts to share LoopGuard with developer communities.

---

## 1. Hacker News ("Show HN")

**Title**: Show HN: LoopGuard – An agentic guardrail to prevent infinite LLM agent loops

**Body**:
```text
Hey HN,

I built LoopGuard (https://github.com/Charbelto/loopguard) because I was tired of watching autonomous AI agents waste API tokens by getting stuck in infinite tool-calling loops. 

Most developers try to prevent loops using two basic approaches:
1. Hard turn/iteration limits: Stopping after N steps. This is a brute-force limit that crashes the execution and doesn't prevent the agent from wasting money before hitting the cap.
2. Exact argument hash checking: Fails the moment the LLM slightly rephrases inputs, reorders arguments, or adds whitespace.

LoopGuard is a lightweight, zero-dependency Python framework that acts as a middleware telemetry interceptor. It works by:
1. Normalizing inputs and observations (stripping punctuation and stopwords).
2. Fitting step telemetry into a streaming/incremental TF-IDF vectorizer (no Scikit-learn or NumPy required).
3. Mapping steps to semantic state clusters using cosine similarity.
4. Analyzing the transition sequence of state IDs using a sliding-window cycle checker to catch cycles of any length (up to 5) repeating K times.
5. Raising a LoopDetectedError and providing self-healing prompts to feed directly back to the LLM (telling it what action failed, what inputs were repeated, and urging it to pivot).

It integrates out-of-the-box with:
- LangChain (CallbackHandler)
- LangGraph (Guarded Node wrapper)
- LlamaIndex (CallbackHandler)

I'd love to get your thoughts on the approach, the similarity threshold controls, or how you handle loop prevention in your own agent architectures!

Code: https://github.com/Charbelto/loopguard
```

---

## 2. Reddit Posts

### Subreddits: `r/LocalLLM`, `r/LangChain`, `r/Python`

**Title**: I built LoopGuard: A zero-dependency guardrail to prevent LLM agents from infinite loop cycles

**Body**:
```text
Hey everyone,

One of the biggest issues with autonomous agents (using LangChain, LangGraph, LlamaIndex, or custom loops) is **infinite loops**:
* Alternating between two tools (e.g. `web_search` -> `write_file` -> `web_search`).
* Rephrasing the same query (e.g. `"weather chicago"` -> `"weather in chicago today"` -> `"weather now chicago"`).
* Repeatedly hitting permission/format errors and retrying.

Exact string hash checking fails because LLMs change wording slightly. Hard-stop limits are expensive and crash the session.

To solve this, I built **LoopGuard**: a lightweight Python library that tracks agent execution telemetry, maps steps to semantic states using incremental TF-IDF and cosine similarity, and catches sequence cycles before they waste your API tokens.

**Key Features:**
* 🛡️ **Declarative Rules Engine**: Add custom rules (oscillations, similarity thresholds, state frequency limits, JSON format check, Regex checks, resource/time caps).
* 🩹 **Self-Healing prompts**: Dynamically formatted instructions based on the violated rule to feed back to the LLM observation, allowing the agent to pivot and recover instead of crashing.
* 🔌 **Plug-and-play Integrations**: Pre-coded callback handlers and node wrappers for LangChain, LangGraph, and LlamaIndex.
* 📊 **Observability**: Export execution graphs directly to Mermaid.js flowcharts or structured dictionaries.
* 📦 **Zero-dependency Core**: Relies only on standard Python libraries.

Check out the repo, docs, and benchmarks:
https://github.com/Charbelto/loopguard

I'd love to hear how you deal with loops in your agent setups and any features you'd like to see added!
```

---

## 3. Twitter / X Thread

### Tweet 1:
> Autonomous LLM agents are notorious for getting stuck in infinite loop cycles—costing you money and crashing workflows. 
> 
> That's why I built **LoopGuard** 🛡️, a lightweight Python framework to auto-detect and heal agent loops before they drain your API budget.
> 
> https://github.com/Charbelto/loopguard

### Tweet 2:
> Why standard fixes fail:
> ❌ **Max Iteration Limits**: Hard-stops after N turns, but fails to prevent token waste on identical retries and crashes instead of recovering.
> ❌ **Exact Hash Matching**: The moment the LLM adds whitespace or rephrases, hash comparisons fail.

### Tweet 3:
> How LoopGuard works:
> 1️⃣ Tokenizes & filters stopwords to clean grammatical noise.
> 2️⃣ Uses incremental TF-IDF & cosine similarity to map steps to semantic states.
> 3️⃣ Runs a sliding-window cycle checker to catch tool-calling oscillations and repetitions.

### Tweet 4:
> If a loop is detected, it raises a `LoopDetectedError` and generates **self-healing prompts** tailored to the violated rule (JSON format error, oscillation, frequency limit) to guide the LLM back on track.

### Tweet 5:
> Seamlessly drops into:
> 🦜🔗 LangChain (Callbacks)
> 🕸️ LangGraph (Guarded nodes)
> 🗂️ LlamaIndex (Event listeners)
> 
> Best of all, the core has **zero external dependencies**! Check it out, give it a star, and let me know your thoughts! 👇
> https://github.com/Charbelto/loopguard

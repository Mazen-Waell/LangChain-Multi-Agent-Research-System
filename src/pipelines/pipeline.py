from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)


# ============================================================
# HELPERS
# ============================================================

def extract_text(content) -> str:
    """
    Convert LangChain message content into plain text.

    LangChain message content can be a plain string, or a list of
    content blocks (e.g. {"type": "text", "text": "..."}). This
    normalizes both cases into a single string.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)

        return "\n".join(parts)

    return str(content)


def get_final_agent_text(agent_result: dict) -> str:
    """
    Robustly extract an agent's final written answer.

    Naively taking messages[-1] can fail because the last message
    might be an AIMessage that only contains tool_calls (no text),
    or has genuinely empty content. This walks backwards through the
    conversation and returns the first non-empty AIMessage text,
    skipping any message that is a bare tool call.
    """

    messages = agent_result.get("messages", [])

    for message in reversed(messages):

        # Skip messages that are pure tool calls with no final text
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            continue

        text = extract_text(getattr(message, "content", ""))

        if text and text.strip():
            return text

    return ""


def run_reader_agent_with_retry(
    reader_agent,
    reader_prompt: str,
    max_retries: int = 1,
) -> str:
    """
    Invoke the Reader Agent and retry once if it returns no usable
    final text (e.g. the model ended its turn on a bare tool call,
    or produced an empty response).
    """

    for attempt in range(max_retries + 1):

        reader_result = reader_agent.invoke(
            {"messages": [("user", reader_prompt)]},
            config={"recursion_limit": 25},
        )

        final_text = get_final_agent_text(reader_result)

        if final_text.strip():
            return final_text

        print(f"\n[Reader Agent] Empty final answer on attempt {attempt + 1}, retrying...")

    return "NO_USABLE_CONTENT"


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_research_pipeline(topic: str) -> dict:

    state = {}

    # --------------------------------------------------------
    # STEP 1 - SEARCH AGENT
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 - SEARCH AGENT")
    print("=" * 60)

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Find the best recent and reliable sources about:

{topic}

IMPORTANT:
Your job is ONLY to discover sources.

Return ONLY the following information for each source:

Title: ...
URL: ...

Do NOT provide:
- summaries
- snippets
- explanations
- analysis
- reliability descriptions
- conclusions

Find up to 5 relevant sources.
"""
            )
        ]
    })

    search_content = search_result["messages"][-1].content
    state["search_results"] = extract_text(search_content)

    print("\nSearch Results:\n")
    print(state["search_results"])

    # --------------------------------------------------------
    # STEP 2 - READER AGENT
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 - READER AGENT")
    print("=" * 60)

    reader_agent = build_reader_agent()

    reader_prompt = f"""
The Search Agent found the following sources:

{state["search_results"]}

Use at most 3 of the most relevant sources. Call scrape_url for each,
then write your final structured summary as instructed.

Focus on facts, statistics, and findings relevant to:

{topic}
"""

    state["scraped_content"] = run_reader_agent_with_retry(
        reader_agent,
        reader_prompt,
        max_retries=1,
    )

    print("\nReader Results:\n")
    print(state["scraped_content"])

    # --------------------------------------------------------
    # STEP 3 - WRITER
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 - WRITER")
    print("=" * 60)

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": state["scraped_content"],
    })

    print("\nFinal Report:\n")
    print(state["report"])

    # --------------------------------------------------------
    # STEP 4 - CRITIC
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 - CRITIC")
    print("=" * 60)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"],
    })

    print("\nCritic Report:\n")
    print(state["feedback"])

    return state
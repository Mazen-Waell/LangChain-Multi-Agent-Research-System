from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# MODEL INITIALIZATION
# ============================================================

# max_tokens is set explicitly and generously to avoid truncated
# responses when the agent has to summarize multiple long sources.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=8000,
)
 
# ============================================================
# 1. SEARCH AGENT
# ============================================================
# Autonomous agent responsible ONLY for discovering sources.

SEARCH_SYSTEM_PROMPT = """You are the Search Agent in a research pipeline.

Your ONLY job is to discover recent and reliable sources.

Rules:
1. Search for the user's topic.
2. You may call `web_search` at most 2 times.
3. Start with one strong search query.
4. Only perform a second search if the first search does not provide enough reliable sources.
5. NEVER call `web_search` more than 2 times.
6. Find up to 5 sources.
7. Once you have enough sources, STOP searching.

Your final answer MUST contain ONLY this format:

Title: ...
URL: ...

Title: ...
URL: ...

Title: ...
URL: ...

Do NOT include:
- snippets
- summaries
- explanations
- analysis
- recommendations
"""


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=SEARCH_SYSTEM_PROMPT,
    )

# ============================================================
# 2. READER AGENT
# ============================================================
# Autonomous agent responsible for reading (scraping) sources and
# extracting verified facts. It decides which sources to scrape and
# calls scrape_url itself, then must always finish with a written
# final answer (never end its turn on a bare tool call).

READER_SYSTEM_PROMPT = """You are the Reader Agent in a research pipeline.

Your job is to read and extract verified information from the sources
provided by the Search Agent.

Rules:

1. You will receive a list of sources containing Title + URL.

2. Read UP TO 3 sources maximum.

3. You MUST attempt to scrape the first 3 valid sources provided.
   Do not choose only one source if multiple valid sources are available.

4. Prefer sources from authoritative organizations such as:
   - Gartner
   - McKinsey
   - Deloitte
   - MIT
   - Stanford
   - official company reports
   - government organizations
   - academic research

5. If authoritative sources are available, prioritize them over:
   - blogs
   - marketing websites
   - content aggregators
   - secondary articles
      
6. For each source, call the `scrape_url` tool using the exact URL
   provided by the Search Agent.

7. After scraping a source:
   - Use ONLY information returned by `scrape_url`.
   - Never invent or assume information.
   - Do not use your own prior knowledge.
   - If the source does not contain useful information about the topic,
     clearly indicate that no relevant information was found.

8. If `scrape_url` returns "SCRAPING_FAILED":
   - Skip that source.
   - Do NOT invent or reconstruct its content.
   - Continue with the next available source.

9. After attempting to read all required sources,
   you MUST provide a final written answer.
   NEVER end your turn immediately after a tool call.

10. Keep the extracted information concise.
    Use approximately 5-10 important bullet points per successful source.



11. Every factual claim in your output must be directly supported
    by the content returned from `scrape_url`.

Your final answer MUST follow this structure:

SOURCE
Title: ...
URL: ...
Verified Content:
- ...
- ...
- ...

SOURCE
Title: ...
URL: ...
Verified Content:
- ...
- ...
- ...

SOURCE
Title: ...
URL: ...
Verified Content:
- ...
- ...
- ...

If none of the sources could be scraped successfully,
your final answer must be exactly:

NO_USABLE_CONTENT
"""

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=READER_SYSTEM_PROMPT,
    )


# ============================================================
# 3. WRITER CHAIN
# ============================================================
# Deterministic chain (prompt | llm | parser). Not an agent - it has
# no tools and no autonomy, it simply transforms research text into
# a structured report.

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer.

Your job is to write a factual, well-structured research report.

IMPORTANT RULES:
- Use ONLY information provided in the research.
- Do NOT invent facts, numbers, statistics, or sources.
- Do NOT make claims that are not supported by the research.
- Clearly distinguish facts from interpretation.
- Keep source attribution clear.
"""
    ),
    (
        "human",
        """Write a detailed research report about:

Topic:
{topic}

Research gathered by the Reader Agent:
{research}

Structure the report as:

1. Introduction

2. Key Findings
   - At least 3 well-explained findings.
   - Every important factual claim should mention its source.

3. Conclusion

4. Limitations
   - Mention important limitations in the available research.

5. Sources
   - List the title and URL of every source used.

IMPORTANT:
Use ONLY verified information from the Reader Agent.
Do not make unsupported conclusions.
Do not claim that something is increasing or decreasing unless the research supports it.
"""
    )
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ============================================================
# 4. CRITIC CHAIN
# ============================================================
# Deterministic chain that audits the final report for accuracy,
# unsupported claims, and citation quality.

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict research critic.

Your job is to audit a research report for:
- factual accuracy
- unsupported claims
- source quality
- evidence coverage
- reasoning
- citation quality
"""
    ),
    (
        "human",
        """Review the research report below.

REPORT:
{report}

Respond in exactly this format:

Score: X/10

Strengths:
- ...
- ...
- ...

Areas to Improve:
- ...
- ...
- ...

Unsupported or Weak Claims:
- ...
- ...

One line verdict:
...

Be especially careful with statistics and numerical claims.
"""
    )
])

critic_chain = critic_prompt | llm | StrOutputParser()
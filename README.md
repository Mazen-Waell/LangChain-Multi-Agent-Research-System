# LangChain Multi-Agent Research System

A multi-agent research system built with LangChain that autonomously researches topics, gathers information from the web (including PDFs), writes comprehensive reports, and evaluates their quality using AI-powered agents.

<p align="center">
  <strong>🔬 Research Automation • 🤖 Multi-Agent Orchestration • 📝 Intelligent Report Generation</strong>
</p>

---

## 🌟 Features

- **Multi-Agent Architecture**: Autonomous agents for searching and reading, plus deterministic chains for writing and critiquing
- **Automated Web Research**: Intelligent web search with the Tavily API
- **Smart Content Extraction**: Multi-strategy web scraping (trafilatura → readability → raw HTML fallback)
- **PDF Support**: Automatically detects and extracts text from PDF sources (via `pdfplumber`), not just HTML pages
- **AI-Powered Report Generation**: Structured, source-attributed research reports
- **Quality Evaluation**: Built-in critic agent that scores reports and flags unsupported claims
- **Interactive UI**: Streamlit-based interface with a live pipeline status view
- **Reliability Safeguards**: Retry logic and robust output parsing to handle empty/incomplete agent responses (see [Challenges We Faced](#-challenges-we-faced--how-we-solved-them))

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│           Streamlit UI (app.py)                     │
│      Multi-Agent Research Assistant Interface       │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│      Research Pipeline (pipeline.py)                │
│   Orchestrates the Search → Reader → Writer → Critic │
│   workflow and handles retries / safe text parsing   │
└──────────────────┬──────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┐
    │              │              │              │
┌───▼────┐   ┌────▼─────┐   ┌────▼────┐   ┌─────▼─────┐
│ Search │   │  Reader   │   │ Writer  │   │  Critic   │
│ Agent  │   │  Agent    │   │ Chain   │   │  Chain    │
│(auto)  │   │  (auto)   │   │(deter.) │   │ (deter.)  │
└───┬────┘   └────┬─────┘   └────┬────┘   └─────┬─────┘
    │             │              │              │
    │  ┌──────────▼─────────┐    │              │
    └─▶│     Tools Layer     │◀──┘              │
       │                     │                  │
       │ • web_search        │                  │
       │ • scrape_url         │                  │
       │   (HTML + PDF)       │                  │
       └─────────────────────┘                  │
                                                  │
                    Report ─────────────────────▶┘
```

### Agent / Chain Responsibilities

| Component | Type | Responsibility |
|---|---|---|
| **Search Agent** | Autonomous agent | Discovers relevant sources across the web using Tavily; decides on its own how many search queries to run |
| **Reader Agent** | Autonomous agent | Selects up to 3 sources, scrapes each one, and extracts only verified facts (never invents content) |
| **Writer Chain** | Deterministic chain | Synthesizes the Reader Agent's verified content into a structured report |
| **Critic Chain** | Deterministic chain | Audits the report for factual accuracy, unsupported claims, and citation quality |

> **Agent vs. Chain**: The Search and Reader components are true LangChain *agents* — they decide for themselves when and how many times to call their tools. The Writer and Critic are *chains* (`prompt → llm → parser`) — fixed, deterministic transformations with no tool access and no autonomy.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **LangChain** (`langchain`, `langchain-core`) | Agent framework and chain composition |
| **langchain-groq** | LLM integration — this project runs on **Groq**, not OpenAI |
| **Groq `openai/gpt-oss-20b`** | The language model powering every agent and chain |
| **Streamlit** | Interactive web UI |
| **Tavily API** | Web search and source discovery |
| **BeautifulSoup4** | HTML parsing and fallback content extraction |
| **Trafilatura** | Primary article/blog content extraction |
| **Readability-lxml** | Secondary content extraction strategy |
| **pdfplumber** | Text extraction from PDF sources |
| **Python-dotenv** | Environment variable management |
| **uv** | Package and virtual environment management |

---

## 📋 Prerequisites

- Python 3.12 or higher
- A [Groq API key](https://console.groq.com/keys) (the project uses Groq's **free tier**, `openai/gpt-oss-20b`)
- A [Tavily API key](https://tavily.com)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LangChain-Multi-Agent-Research-System.git
cd LangChain-Multi-Agent-Research-System
```

### 2. Install Dependencies

**Using `uv` (recommended):**

```bash
uv sync
```

**Using `pip`:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys from:
- [Groq Console](https://console.groq.com/keys)
- [Tavily](https://tavily.com)

---

## 💡 Usage

### Run with Streamlit UI (Recommended)

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Run as a CLI Script

```bash
python main.py
```

Edit the `topic` variable in `main.py` to research a different subject.

---

## ☁️ Deployment (Render)

This project  deployed as a web service on [Render](https://render.com). A few things differ from running it locally:
- Render assigns a dynamic port via the `$PORT` environment variable — Streamlit must be told to bind to it explicitly.

---

## 📁 Project Structure

```
.
├── app.py                      # Streamlit web interface
├── main.py                     # CLI entry point / testing script
├── pyproject.toml              # Project metadata & dependencies (uv)
├── requirements.txt            # Python dependencies (pip alternative)
├── README.md                   # This file
├── LICENSE                     # License file
│
└── src/
    ├── agents/
    │   └── agents.py           # Search Agent, Reader Agent, Writer Chain, Critic Chain
    ├── tools/
    │   └── tools.py            # web_search, scrape_url (HTML + PDF support)
    └── pipelines/
        └── pipeline.py         # Pipeline orchestration + reliability helpers
```

---

## 🔄 Workflow

1. **User Input** — Enter a research topic via the UI or script
2. **Search Phase** — The Search Agent queries Tavily and returns up to 5 sources
3. **Reading Phase** — The Reader Agent scrapes up to 3 of those sources (HTML or PDF) and extracts only verified facts
4. **Writing Phase** — The Writer Chain synthesizes the verified facts into a structured report
5. **Review Phase** — The Critic Chain scores the report and flags unsupported claims
6. **Output** — The final report and critic feedback are displayed, with a Markdown download option

---

## ⚠️ Challenges We Faced & How We Solved Them

Building this on a **free-tier LLM** (`gpt-oss-20b` via Groq) surfaced several real reliability issues that a more powerful/expensive model might mask. Here's what came up during development and how each was addressed:

### 1. Reader Agent returning empty responses

**Problem:** The Reader Agent would sometimes finish its turn after calling `scrape_url` without ever writing a final summary — the last message in the conversation was a bare tool call with no text, so naively reading `messages[-1].content` returned an empty string.

**Fix:**
- Rewrote the Reader Agent's system prompt to explicitly require a final written answer in a fixed structure after all tool calls are made.
- Added `get_final_agent_text()`, which walks the conversation backwards and returns the first message that actually contains text, skipping bare tool-call messages.
- Added a one-time retry (`run_reader_agent_with_retry()`) for the rare case where the model still returns nothing.

### 2. Truncated output due to default token limits

**Problem:** `ChatGroq` wasn't given an explicit `max_tokens`, so it fell back to a low default. When an agent needed to summarize multiple long sources in one response, the output would get cut off mid-sentence.

**Fix:** Set `max_tokens=8000` explicitly on the shared LLM instance. This is a direct consequence of using a free-tier model — defaults tend to be conservative, so they need to be set deliberately rather than assumed.

### 3. PDF sources silently failing to scrape

**Problem:** Some high-quality sources (e.g., World Economic Forum reports) are served as PDFs, not HTML. The original scraper only knew how to parse HTML, so PDF sources returned garbled or empty text.

**Fix:** Added PDF detection to `scrape_url` using three signals (`Content-Type` header, URL suffix, and the file's `%PDF` byte signature), and added a dedicated `pdfplumber`-based extraction path for PDF content.

### 4. `uv add` failing with a build error

**Problem:** `uv` treated this project as an installable package (because of a `[project.scripts]` entry point that didn't actually exist), so every `uv add <package>` attempt failed while trying to build the project first.

**Fix:** Added `[tool.uv] package = false` to `pyproject.toml` so `uv` treats this as an application, not a distributable library.

---

## 🔍 Known Limitations

- **Search Agent call count is unbounded.** Its system prompt allows it to re-run `web_search` as many times as it feels necessary. This usually produces better sources, but occasionally results in a long tool-calling chain that can affect response consistency. Constraining this further is a planned improvement.
- **Free-tier rate limits.** Since the project runs on Groq's free tier, heavy or repeated testing can hit rate limits — if a step fails unexpectedly, check for a `429`/rate-limit error before assuming a code bug.
- **Bot-blocked sources.** Some sites (e.g., paywalled news, heavily JS-rendered pages) will still fail to scrape even with the PDF/HTML fallback strategies, since they require a full browser (e.g., Playwright) to render.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- LLM inference powered by [Groq](https://groq.com/)
- Search powered by [Tavily](https://tavily.com)
- UI built with [Streamlit](https://streamlit.io/)

---

## 📧 Support

For support, open an issue on GitHub or contact the maintainers.

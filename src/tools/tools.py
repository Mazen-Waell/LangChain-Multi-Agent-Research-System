import io
import re
import os

import requests
import pdfplumber
import trafilatura
from bs4 import BeautifulSoup
from readability import Document
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ============================================================
# SEARCH TOOL
# ============================================================

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic.
    Returns Titles, URLs and snippets."""

    results = tavily.search(query=query, max_results=5)
    out = []

    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )

    return "\n------\n".join(out)


# ============================================================
# SCRAPING HELPERS
# ============================================================

def _extract_pdf_text(content: bytes, max_chars: int = 5000) -> str:
    """
    Extract readable text from raw PDF bytes using pdfplumber.
    Limited to the first 15 pages for performance.
    """

    try:
        text_parts = []

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages[:15]:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        full_text = "\n".join(text_parts)
        cleaned = re.sub(r"\s+", " ", full_text).strip()

        if cleaned:
            return cleaned[:max_chars]

        return "Could not extract meaningful text from the PDF."

    except Exception as e:
        return f"SCRAPING_FAILED: PDF extraction error - {str(e)}"


def _is_pdf_response(url: str, response: requests.Response) -> bool:
    """
    Detect whether an HTTP response is actually a PDF, using three
    signals (some servers don't set Content-Type correctly, so we
    also check the URL and the file's magic number).
    """

    content_type = response.headers.get("Content-Type", "").lower()

    return (
        "application/pdf" in content_type
        or url.lower().endswith(".pdf")
        or response.content[:4] == b"%PDF"
    )


def _extract_html_text(html: str, max_chars: int = 5000) -> str:
    """
    Extract clean readable text from an HTML page using a cascade of
    strategies, from best (article-focused) to most permissive
    (full-page fallback).
    """

    # Strategy 1: trafilatura (best for articles/blogs)
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
    )

    if extracted and len(extracted.strip()) > 200:
        return re.sub(r"\s+", " ", extracted)[:max_chars]

    # Strategy 2: readability
    doc = Document(html)
    clean_html = doc.summary()

    soup = BeautifulSoup(clean_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    if text and len(text.strip()) > 200:
        return re.sub(r"\s+", " ", text)[:max_chars]

    # Strategy 3: full-page fallback
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    cleaned = re.sub(r"\s+", " ", text)

    if cleaned:
        return cleaned[:max_chars]

    return "Could not extract meaningful content from the page."


# ============================================================
# SCRAPE TOOL
# ============================================================

@tool
def scrape_url(url: str) -> str:
    """
    Scrape and extract clean readable content from a URL.
    Automatically detects PDFs and extracts their text.
    Uses multiple extraction strategies for HTML pages.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        if _is_pdf_response(url, response):
            return _extract_pdf_text(response.content)

        return _extract_html_text(response.text)

    except requests.exceptions.Timeout:
        return f"SCRAPING_FAILED: {url} - Request timed out"

    except requests.exceptions.HTTPError as e:
        return f"SCRAPING_FAILED: {url} - {str(e)}"

    except Exception as e:
        return f"SCRAPING_FAILED: {url} - {str(e)}"
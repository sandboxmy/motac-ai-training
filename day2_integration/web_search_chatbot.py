from __future__ import annotations

import datetime
import os
from typing import List, Dict

from flask import Flask, jsonify, request
from duckduckgo_search import DDGS
import requests

app = Flask(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma3:1b")


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using DuckDuckGo and return a list of results.

    Each result contains: title, href, body (snippet).
    """
    if not query:
        return []

    results: List[Dict[str, str]] = []
    try:
        with DDGS(timeout=20) as ddgs:
            for r in ddgs.text(query, max_results=max_results, safesearch="moderate"):  # type: ignore[arg-type]
                results.append(
                    {
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", ""),
                    }
                )
    except Exception as exc:  # pragma: no cover - simple demo logging
        results = [
            {
                "title": "Search error",
                "href": "",
                "body": f"Failed to fetch results: {exc}",
            }
        ]

    return results


def news_search(query: str, max_results: int = 5, timelimit: str | None = None) -> List[Dict[str, str]]:
    """Search recent news via DuckDuckGo News API.

    timelimit examples per duckduckgo-search: 'd' (day), 'w' (week), 'm' (month), or 'y' (year).
    """
    if not query:
        return []

    results: List[Dict[str, str]] = []
    try:
        with DDGS(timeout=20) as ddgs:
            for r in ddgs.news(query, max_results=max_results, timelimit=timelimit):  # type: ignore[arg-type]
                results.append(
                    {
                        "title": r.get("title", ""),
                        "href": r.get("url", r.get("href", "")),
                        "body": r.get("body", r.get("excerpt", "")),
                        "date": r.get("date"),
                        "source": r.get("source"),
                    }
                )
    except Exception as exc:  # pragma: no cover - simple demo logging
        results = [
            {
                "title": "News search error",
                "href": "",
                "body": f"Failed to fetch news: {exc}",
            }
        ]

    return results


def is_recent_query(q: str) -> bool:
    ql = (q or "").lower()
    triggers = ["latest", "recent", "today", "this week", "breaking", "news", "update"]
    return any(t in ql for t in triggers)


def compose_prompt(query: str, hits: List[Dict[str, str]]) -> str:
    """Create a prompt for Ollama using the query and gathered results, asking for a concise summary with citations."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = []
    lines.append("You are a helpful research assistant.")
    lines.append("Use the provided snippets to answer the user's query with a brief up-to-date summary.")
    lines.append("Requirements:")
    lines.append("- Write 3-6 concise bullet points summarizing key findings.")
    lines.append("- Include inline citations like [1], [2] referencing the sources list below.")
    lines.append("- If results conflict, note it and provide the most reliable view.")
    lines.append("- End with a 'Sources' section that lists each [n] and its URL.")
    lines.append("")
    lines.append(f"Current date/time: {now}")
    lines.append(f"User query: {query}")
    lines.append("")
    lines.append("Sources:")
    for i, h in enumerate(hits, start=1):
        title = (h.get("title") or "Untitled").strip()
        url = (h.get("href") or "").strip()
        snippet = (h.get("body") or "").strip()
        source = (h.get("source") or "").strip()
        date = (h.get("date") or "").strip() if isinstance(h.get("date"), str) else str(h.get("date") or "")
        meta = f" ({source}, {date})" if source or date else ""
        lines.append(f"[{i}] {title}{meta}\n{snippet}\n{url}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def ask_ollama(prompt: str) -> str:
    """Send the composed prompt to Ollama and return the model's response text."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "")).strip() or "(No response from model)"
    except requests.RequestException as exc:
        return f"Ollama request failed. Ensure 'ollama serve' is running and model '{MODEL_NAME}' is pulled. Details: {exc}"


@app.get("/search")
def search_get():
    """GET /search with parameters:

    query: required. The user query.
    max_results: optional int, default 5.
    source: optional str in {"web","news","auto"}, default "auto".
    timelimit: optional str for news recency (e.g., d, w, m, y), default None.

    Returns JSON: { query, timestamp, results, answer, model, source_used }
    """
    query = str(request.args.get("query", "")).strip()
    try:
        max_results = int(request.args.get("max_results", 5))
    except Exception:
        max_results = 5
    source = str(request.args.get("source", "auto")).strip().lower()
    timelimit = request.args.get("timelimit")

    if not query:
        return jsonify({"error": "Please provide 'query' as a query parameter."}), 400

    if source not in {"web", "news", "auto"}:
        source = "auto"

    source_used = source
    if source == "auto":
        source_used = "news" if is_recent_query(query) else "web"

    if source_used == "news":
        hits = news_search(query, max_results=max_results, timelimit=timelimit)
    else:
        hits = web_search(query, max_results=max_results)

    prompt = compose_prompt(query, hits)
    answer = ask_ollama(prompt)
    return jsonify({
        "query": query,
        "timestamp": datetime.datetime.now().isoformat(timespec="minutes"),
        "results": hits,
        "answer": answer,
        "model": MODEL_NAME,
        "source_used": source_used,
    })


@app.post("/search")
def search_post():
    """POST /search with JSON body

    {"query": str, "max_results": int, "source": "web|news|auto", "timelimit": "d|w|m|y"}
    """
    body = request.get_json(silent=True) or {}
    query = str(body.get("query", "")).strip()
    try:
        max_results = int(body.get("max_results", 5))
    except Exception:
        max_results = 5
    source = str(body.get("source", "auto")).strip().lower()
    timelimit = body.get("timelimit")

    if not query:
        return jsonify({"error": "Please provide 'query' in JSON body."}), 400

    if source not in {"web", "news", "auto"}:
        source = "auto"

    source_used = source
    if source == "auto":
        source_used = "news" if is_recent_query(query) else "web"

    if source_used == "news":
        hits = news_search(query, max_results=max_results, timelimit=timelimit)
    else:
        hits = web_search(query, max_results=max_results)

    prompt = compose_prompt(query, hits)
    answer = ask_ollama(prompt)
    return jsonify({
        "query": query,
        "timestamp": datetime.datetime.now().isoformat(timespec="minutes"),
        "results": hits,
        "answer": answer,
        "model": MODEL_NAME,
        "source_used": source_used,
    })


if __name__ == "__main__":
    print("Starting Flask dev server on http://127.0.0.1:5000")
    print("Use: GET /search?query=latest+AI+news or POST /search { query } ")
    print(f"Ollama endpoint: {OLLAMA_URL} | Model: {MODEL_NAME}")
    app.run(debug=True)

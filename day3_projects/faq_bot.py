"""FAQ chatbot with embedding-based retrieval augmented responses using Ollama."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

FAQ_PATH = Path(__file__).resolve().parents[1] / "data" / "faq_data.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def load_faq_data() -> list[dict[str, Any]]:
    """Read questions and answers from the JSON file."""
    with FAQ_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_embedding(text: str) -> list[float] | None:
    """Request an embedding vector from the Ollama embeddings endpoint."""
    try:
        response = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=45,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()
    return data.get("embedding")


def build_repository_with_embeddings(
    faq_items: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """Return FAQ items paired with cached embeddings."""
    items = list(faq_items)
    vectors: list[list[float]] = []

    for item in items:
        text_for_embedding = f"Question: {item['question']}\nAnswer: {item['answer']}"
        vector = get_embedding(text_for_embedding)
        if vector is None:
            # Embedding failed; append empty vector so we can fall back later.
            vectors.append([])
        else:
            vectors.append(vector)

    return items, vectors


FAQ_ITEMS, FAQ_VECTORS = build_repository_with_embeddings(load_faq_data())


def rank_faq_items(question: str) -> list[tuple[float, dict[str, Any]]]:
    """Return FAQ items sorted by cosine similarity to the question."""
    question_vector = get_embedding(question)
    if question_vector is None:
        return []

    scored_items: list[tuple[float, dict[str, Any]]] = []
    for vector, item in zip(FAQ_VECTORS, FAQ_ITEMS):
        score = cosine_similarity(question_vector, vector)
        scored_items.append((score, item))

    scored_items.sort(key=lambda pair: pair[0], reverse=True)
    return scored_items


def call_ollama_with_context(question: str, context_answer: str) -> str:
    """Ask Ollama to craft a response using the retrieved FAQ answer."""
    prompt = (
        "You are a helpful FAQ assistant. Use the provided answer as trusted"
        " context to respond to the user's question. If the context does not"
        " cover the question, say you are unsure and ask the user to rephrase.\n\n"
        f"Context answer: {context_answer}\n"
        f"User question: {question}\n"
        "Respond in 2-3 friendly sentences."
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=45,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - simple demo
        return (
            "I found a similar answer but could not reach the AI writer. "
            "Please try again later. Technical details: "
            f"{exc}"
        )

    data: dict[str, Any] = response.json()
    return data.get("response", context_answer)


@app.post("/faq")
def faq_endpoint():
    """Accept JSON {"question": "..."} and return the best answer."""
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please send a question."}), 400

    ranked = rank_faq_items(question)

    top_score, top_item = ranked[0] if ranked else (0.0, None)
    if top_item is None or top_score < 0.5:
        return jsonify(
            {
                "answer": (
                    "I could not find a close match. Please rephrase or ask a team member."
                ),
                "match_score": round(top_score, 3),
            }
        )

    context_answer = top_item["answer"]
    generated = call_ollama_with_context(question, context_answer)

    return jsonify(
        {
            "answer": generated,
            "match_question": top_item["question"],
            "match_score": round(top_score, 3),
        }
    )


if __name__ == "__main__":
    print("Starting FAQ bot on http://127.0.0.1:5000")
    print("Send POST requests to /faq with JSON: { 'question': '...' }")
    app.run(debug=True)

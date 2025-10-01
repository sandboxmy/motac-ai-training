"""Flask API that lets trainers set a prompt and chat with Ollama."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from flask import Flask, jsonify, request


app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"

# Default system prompt used if the trainer does not set one yet.
DEFAULT_SYSTEM_PROMPT = (
    "You are a patient AI coach helping beginners understand AI concepts. "
    "Answer in 2 points only"
)


@dataclass
class ChatEntry:
    """Represents each turn in the conversation history."""

    role: str  # Either "user" or "assistant"
    content: str


# In-memory session state for the demo. Reset when the server restarts.
session_state: dict[str, Any] = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "history": [],  # list[ChatEntry]
}


def build_prompt(system_prompt: str, history: list[ChatEntry], user_message: str) -> str:
    """Return a single text prompt that includes history and the new message."""
    lines: list[str] = [f"System: {system_prompt}", ""]

    for entry in history:
        role = entry.role.capitalize()
        lines.append(f"{role}: {entry.content}")

    lines.extend(
        [
            f"User: {user_message}",
            "Assistant:",
        ]
    )

    return "\n".join(lines)


def call_model(system_prompt: str, history: list[ChatEntry], user_message: str) -> str:
    """Send the constructed prompt to Ollama and return the AI reply."""
    prompt_text = build_prompt(system_prompt, history, user_message)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt_text, "stream": False},
            timeout=45,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - simple demo
        return f"Could not reach Ollama. Please start it first. Details: {exc}"

    data: dict[str, Any] = response.json()
    return data.get("response", "Model did not return text.")


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the server is live."""
    return jsonify({"status": "ok"})


@app.post("/setup")
def setup_prompt():
    """Set the system prompt and reset the conversation history."""
    body = request.get_json(silent=True) or {}
    system_prompt = str(body.get("system_prompt", "")).strip()

    if not system_prompt:
        return jsonify({"error": "Please provide system_prompt in the JSON body."}), 400

    session_state["system_prompt"] = system_prompt
    session_state["history"] = []

    return jsonify({
        "message": "System prompt updated and history cleared.",
        "system_prompt": system_prompt,
    })


@app.post("/chat")
def chat():
    """Send a user message and receive an AI reply using the stored prompt."""
    body = request.get_json(silent=True) or {}
    user_message = str(body.get("message", "")).strip()

    if not user_message:
        return jsonify({"error": "Please provide 'message' in the JSON body."}), 400

    system_prompt = session_state["system_prompt"]
    history: list[ChatEntry] = session_state["history"]

    reply = call_model(system_prompt, history, user_message)

    # Update history so the conversation has context for the next turn.
    history.append(ChatEntry(role="user", content=user_message))
    history.append(ChatEntry(role="assistant", content=reply))

    return jsonify({
        "system_prompt": system_prompt,
        "user_message": user_message,
        "assistant_reply": reply,
        "history": [entry.__dict__ for entry in history],
    })


@app.get("/history")
def get_history():
    """Return the current system prompt and the full chat history."""
    history: list[ChatEntry] = session_state["history"]
    return jsonify({
        "system_prompt": session_state["system_prompt"],
        "history": [entry.__dict__ for entry in history],
    })


def main() -> None:
    """Run the Flask development server for Postman-based chatting."""
    print("Starting prompt chat API on http://127.0.0.1:5001")
    print("Use POST /setup to define the system prompt, then POST /chat to talk.")
    app.run(host="127.0.0.1", port=5001, debug=True)


if __name__ == "__main__":
    main()

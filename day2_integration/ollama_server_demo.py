"""Demo script to interact with a locally running Ollama server."""

import json
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # Adjust based on available models on your machine.


def call_ollama(prompt: str) -> dict[str, Any]:
    """Send a prompt to the Ollama server and return the JSON response."""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Run a simple request and print the AI output."""
    print("Connecting to your local Ollama server...")
    try:
        result = call_ollama("Explain AI to a beginner in two sentences.")
    except requests.RequestException as exc:
        print("Could not reach the Ollama server. Make sure it is running.")
        print(f"Technical details: {exc}")
        return

    # The response from Ollama contains a 'response' field with the generated text.
    print("Server responded successfully! Here is the AI message:\n")
    print(result.get("response", "No response field returned."))

    # Show the full JSON for teaching purposes.
    print("\nFull JSON payload (useful for debugging):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

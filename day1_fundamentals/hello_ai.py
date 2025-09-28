"""Simple introduction script for explaining AI vs traditional programming."""

# In a traditional program, we give the computer exact step-by-step instructions.
def traditional_program(name: str) -> str:
    """Return a fixed greeting following a strict set of rules."""
    greeting = f"Hello {name}!"
    # The steps are predictable and repeatable.
    return greeting + " This message never changes because the rules are fixed."


# In a very simple AI-style program, we can make decisions based on patterns or data.
def simple_ai_response(prompt: str) -> str:
    """Return a response based on keywords in the prompt."""
    prompt_lower = prompt.lower()

    if "weather" in prompt_lower:
        return "It looks like a sunny day! (AI guessed based on the word 'weather')"
    if "joke" in prompt_lower:
        return "Why did the computer visit the doctor? Because it had a virus!"
    if "advice" in prompt_lower:
        return "Remember to take breaks while you code and drink water."

    # If we do not recognize the prompt, we still try to respond politely.
    return "I'm still learning, but I want to help!"


def main() -> None:
    """Show the difference between the two approaches."""
    print("--- Traditional Program Output ---")
    print(traditional_program("Participant"))

    print("\n--- Simple AI-Style Output ---")
    # Imagine the participant asking different questions.
    sample_prompts = [
        "What's the weather today?",
        "Tell me a joke!",
        "Any advice for learning Python?",
        "What is AI?",
    ]

    for prompt in sample_prompts:
        print(f"Prompt: {prompt}")
        print(f"AI response: {simple_ai_response(prompt)}\n")


if __name__ == "__main__":
    main()

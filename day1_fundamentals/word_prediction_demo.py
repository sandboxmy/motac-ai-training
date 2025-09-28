"""Tiny next-word prediction demo using simple probabilities."""

from collections import defaultdict, Counter

import nltk


# Download the tokenizer data once. quiet=True keeps the console tidy for students.
nltk.download("punkt", quiet=True)


TRAINING_TEXT = (
    "artificial intelligence makes smart tools possible. "
    "artificial intelligence helps people solve problems. "
    "machine learning is a helpful part of artificial intelligence. "
    "smart tools can answer simple questions."
)


def build_bigram_model(text: str) -> dict[str, Counter]:
    """Create a mapping of current word -> counter of possible next words."""
    model: dict[str, Counter] = defaultdict(Counter)
    # `word_tokenize` handles punctuation better than simple string split.
    words = [word.lower() for word in nltk.word_tokenize(text)]

    for index in range(len(words) - 1):
        current_word = words[index]
        next_word = words[index + 1]
        model[current_word][next_word] += 1

    return model


def predict_next_word(model: dict[str, Counter], current_word: str) -> str:
    """Return the most common next word. Fallback to a friendly message."""
    choices = model.get(current_word.lower())
    if not choices:
        return "(I am unsure what comes next.)"
    next_word, _ = choices.most_common(1)[0]
    return next_word


def main() -> None:
    """Train the model and let the user test it."""
    print("Training simple bigram model...")
    model = build_bigram_model(TRAINING_TEXT)

    print("Model ready! Try a word to see the predicted next word.")
    print("Type 'quit' to exit.\n")

    while True:
        user_word = input("Enter a word: ").strip()
        if not user_word:
            print("Please type something so I can predict!")
            continue
        if user_word.lower() == "quit":
            print("Goodbye! Keep exploring language models.")
            break

        prediction = predict_next_word(model, user_word)
        print(f"Most likely next word: {prediction}\n")


if __name__ == "__main__":
    main()

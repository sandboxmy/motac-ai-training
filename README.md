# 3-Day AI & Ollama Training Examples

This repository contains runnable Python examples used during the 3-day training. Each day focuses on a different theme, from fundamentals to integrations and mini-projects.

## 1. Prerequisites

1. Install Python 3.10+.
2. (Recommended) Create a virtual environment:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install [Ollama](https://ollama.com/download) and download a model (example uses `llama3`).
   ```bash
   ollama pull llama3
   ```

## 2. Running Ollama

Start the Ollama server in a separate terminal before running scripts that call it:
```bash
ollama serve
```

The examples assume the server is reachable at `http://localhost:11434`.

## 3. Dataset

Sample data files live in the `data/` directory:
- `sales.csv`: Small dataset for reporting and analysis exercises.
- `faq_data.json`: Knowledge base for the FAQ bot.
- `sample_document.txt`: Text used for the document analysis demo.

## 4. Day 1 – Fundamentals (`day1_fundamentals/`)

- `hello_ai.py`
  ```bash
  python day1_fundamentals/hello_ai.py
  ```
  Shows a comparison between a traditional program and a simple AI-style response function.

- `word_prediction_demo.py`
  ```bash
  python day1_fundamentals/word_prediction_demo.py
  ```
  Builds a tiny next-word predictor using NLTK bigrams and lets participants test words interactively.

## 5. Day 2 – Integration & Automation (`day2_integration/`)

- `ollama_server_demo.py`
  ```bash
  python day2_integration/ollama_server_demo.py
  ```
  Sends a simple prompt to Ollama and prints the JSON response for teaching purposes.

- `simple_chatbot.py`
  ```bash
  python day2_integration/simple_chatbot.py
  ```
  Starts a Flask API at `http://127.0.0.1:5000/chat`. Send POST JSON `{ "message": "Hello" }` to receive an AI reply.

- `automation_email_report.py`
  ```bash
  python day2_integration/automation_email_report.py
  ```
  Reads `data/sales.csv`, produces a daily summary, and prints an email-style report.

- `data_analysis_demo.py`
  ```bash
  python day2_integration/data_analysis_demo.py
  ```
  Loads the sales data, prints summary statistics, and saves `data/sales_chart.png` with a simple bar chart.

- `prompt_engineering_demo.py`
  ```bash
  python day2_integration/prompt_engineering_demo.py
  ```
  Compares results from a vague prompt vs a detailed prompt when talking to Ollama.

## 6. Day 3 – Project Examples (`day3_projects/`)

- `faq_bot.py`
  ```bash
  python day3_projects/faq_bot.py
  ```
  Starts a Flask service at `http://127.0.0.1:5000/faq`. Send POST JSON `{ "question": "What are your business hours?" }`.

- `doc_analysis.py`
  ```bash
  python day3_projects/doc_analysis.py
  ```
  Reads `data/sample_document.txt` and asks Ollama for a summary and keywords.

- `auto_reply_system.py`
  ```bash
  python day3_projects/auto_reply_system.py
  ```
  Generates polite email drafts for demo messages using the Ollama model.

- `db_integration.py`
  ```bash
  python day3_projects/db_integration.py
  ```
  Creates a SQLite database of lessons, shows the rows, and requests AI explanations for each lesson.

## 7. Final Project (`final_project/`)

- `summarized_chatbot.py`
  ```bash
  python final_project/summarized_chatbot.py
  ```
  A comprehensive FastAPI application that combines document analysis, data visualization, and AI-powered insights. Accepts PDF, CSV/Excel files, or URLs for analysis. Runs at `http://localhost:8000`. Features:
  - Document summarization with key points extraction
  - Data analysis with automatic chart generation
  - AI-powered insights using Ollama
  - RESTful API with interactive documentation at `/docs`

- `test_api.py`
  ```bash
  python final_project/test_api.py
  ```
  Test script to verify the API functionality with sample data.

## 8. Troubleshooting Tips

- If a script reports `Could not reach the Ollama server`, confirm the server is running and that the model (e.g., `llama3`) is downloaded.
- Adjust the `MODEL_NAME` constant in scripts if you prefer a different model.
- When running Flask apps, stop them with `Ctrl+C` before starting another script on the same port.
- For the FastAPI application in final_project, ensure all dependencies are installed from requirements.txt.

Enjoy the training! Feel free to extend these examples or adapt them to your audience.

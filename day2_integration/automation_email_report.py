"""Simple automation script that prepares and emails a daily sales report."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sales.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")


def load_sales_data() -> pd.DataFrame:
    """Read the CSV containing sales information."""
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def build_report(df: pd.DataFrame) -> str:
    """Create a friendly summary that could go into an email."""
    total_sales = df["sales"].sum()
    total_revenue = df["revenue"].sum()

    latest_date = df["date"].max().date()
    sales_by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)

    lines = [
        "Daily Sales Report",  # Subject line idea.
        f"Date: {latest_date:%Y-%m-%d}",
        "",
        f"Total items sold: {total_sales}",
        f"Total revenue: ${total_revenue:,.2f}",
        "",
        "Revenue by region:",
    ]

    for region, revenue in sales_by_region.items():
        lines.append(f"- {region}: ${revenue:,.2f}")

    lines.extend(
        [
            "",
            "Next steps:",
            "1. Share this summary with the sales team.",
            "2. Review any regions that dipped in performance.",
        ]
    )

    return "\n".join(lines)


def fetch_ai_insight(df: pd.DataFrame) -> str:
    """Ask Ollama for a short insight about the latest sales numbers."""
    latest_date = df["date"].max().date()
    total_revenue = df["revenue"].sum()
    top_region = (
        df.groupby("region")["revenue"].sum().sort_values(ascending=False).index[0]
    )

    prompt = (
        "You are an upbeat sales coach. Review the metrics below and write two"
        " friendly sentences highlighting successes and one suggestion for the"
        " team. Keep it under 80 words.\n\n"
        f"Latest date: {latest_date}\n"
        f"Total revenue: ${total_revenue:,.2f}\n"
        f"Top region by revenue: {top_region}\n"
        "Include an emoji to keep the tone light."
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
            "(AI insight unavailable. Start Ollama with `ollama serve` to add a"
            f" smart summary. Details: {exc})"
        )

    data: dict[str, Any] = response.json()
    return data.get(
        "response",
        "(AI insight unavailable. The model did not return any text.)",
    )


def send_email_stub(subject: str, body: str) -> None:
    """Fallback helper: print the email when real sending is not configured."""
    print("Sending email (console fallback)...")
    print(f"Subject: {subject}")
    print("Body:\n")
    print(body)


def send_email(subject: str, body: str, recipient: str) -> None:
    """Send the report via SMTP using environment variable settings."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))  # Default to SSL port 465.
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender_email = os.getenv("SENDER_EMAIL", smtp_username)

    if not all([smtp_server, smtp_username, smtp_password, sender_email]):
        print("SMTP settings not fully configured. Printing the email instead.")
        send_email_stub(subject, body)
        print(
            "Set SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, and SENDER_EMAIL "
            "environment variables to enable real email delivery."
        )
        return

    message = EmailMessage()
    message["From"] = sender_email
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(smtp_username, smtp_password)
            server.send_message(message)
    except Exception as exc:  # pragma: no cover - simplified demo
        print("Could not send the email. Showing the content in the terminal instead.")
        print(f"Technical details: {exc}")
        send_email_stub(subject, body)
        return

    print(f"Email sent to {recipient}! Please check the inbox to confirm delivery.")


def parse_args() -> argparse.Namespace:
    """Collect the recipient email from CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a sales report and email it to the chosen recipient."
    )
    parser.add_argument(
        "--recipient",
        help="Email address that should receive the report. If omitted, you will be asked.",
    )
    return parser.parse_args()


def main() -> None:
    """Load data, create the report, and send it to the selected email address."""
    args = parse_args()

    # Prompt for the recipient if it was not provided via the CLI.
    recipient = args.recipient or input("Enter the recipient email address: ").strip()
    if not recipient:
        print("No recipient provided. Exiting without sending an email.")

    print(f"Loading sales data from {DATA_PATH}")
    sales_df = load_sales_data()

    # Build the report body
    report_body = build_report(sales_df)

    # Fetch the AI insight
    ai_insight = fetch_ai_insight(sales_df)

    # Combine the report body and the AI insight
    full_body = report_body + "\n\nAI insight:\n" + ai_insight

    today_str = dt.date.today().strftime("%Y-%m-%d")
    subject = f"Daily Sales Report - {today_str}"

    send_email(subject, full_body, recipient)


if __name__ == "__main__":
    main()

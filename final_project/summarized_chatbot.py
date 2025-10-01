"""
FastAPI application that summarizes documents, data, and generates charts.
Accepts PDF, Excel/CSV, or URL links and provides AI-powered analysis.
"""

import os
import io
import base64
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
from urllib.parse import urlparse

# Configure Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma3:1b")

# Initialize FastAPI app
app = FastAPI(title="Document & Data Summarizer API", version="1.0.0")

# Request/Response models for API
class SummaryRequest(BaseModel):
    url: Optional[str] = None
    
class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str]
    ai_insights: str
    chart_base64: Optional[str] = None
    data_stats: Optional[Dict[str, Any]] = None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")


def extract_text_from_url(url: str) -> str:
    """Download and extract text content from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Simple text extraction (you could enhance this with BeautifulSoup for HTML)
        content_type = response.headers.get('content-type', '').lower()
        if 'text' in content_type or 'html' in content_type:
            return response.text
        else:
            return f"Content from URL: {response.text[:5000]}"  # Limit to 5000 chars
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")


def read_csv_or_excel(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or Excel file and return DataFrame."""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format")
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading data file: {str(e)}")


def call_ollama(prompt: str) -> str:
    """Send prompt to Ollama and get response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response from model")
    except requests.RequestException as e:
        return f"AI service unavailable. Please ensure Ollama is running: {str(e)}"


def get_embeddings(text: str) -> list[float]:
    """Get embeddings for text using Ollama."""
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": MODEL_NAME, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])
    except:
        # Return empty list if embedding fails (optional feature)
        return []


def summarize_text(text: str) -> Dict[str, Any]:
    """Create summary and extract key points from text."""
    # Limit text length for processing
    text = text[:10000]  # Limit to 10000 characters
    
    # Get summary
    summary_prompt = (
        "You are a helpful assistant. Provide a clear and concise summary "
        "of the following content in 3-4 sentences. Focus on the main ideas:\n\n"
        f"{text}\n\n"
        "Summary:"
    )
    summary = call_ollama(summary_prompt)
    
    # Extract key points
    key_points_prompt = (
        "Based on this content, list the 5 most important points or findings "
        "provide in numbered list. Be specific and actionable:\n\n"
        f"{text}\n\n"
        "Key points:"
    )
    key_points_text = call_ollama(key_points_prompt)
    
    # Parse key points into list
    key_points = []
    for line in key_points_text.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
            # Clean up the line
            if line[0].isdigit():
                line = line.split('.', 1)[-1].strip() if '.' in line else line
            else:
                line = line[1:].strip()
            if line:
                key_points.append(line)
    
    # Get AI insights
    insights_prompt = (
        "You are an expert analyst. Based on this content, provide 2-3 actionable "
        "insights or recommendations. What should the reader pay attention to? "
        "Keep your response under 100 words:\n\n"
        f"{text}\n\n"
        "Insights and recommendations:"
    )
    insights = call_ollama(insights_prompt)
    
    # Get embeddings (optional - for future use)
    embeddings = get_embeddings(text[:1000])  # Use first 1000 chars for embedding
    
    return {
        "summary": summary,
        "key_points": key_points[:5],  # Limit to 5 points
        "ai_insights": insights,
        "has_embeddings": len(embeddings) > 0
    }


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze DataFrame and generate statistics and chart."""
    stats = {}
    chart_base64 = None
    
    # Basic statistics
    stats["shape"] = {"rows": len(df), "columns": len(df.columns)}
    stats["columns"] = list(df.columns)
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        stats["numeric_summary"] = {}
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            stats["numeric_summary"][col] = {
                "mean": float(df[col].mean()),
                "sum": float(df[col].sum()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
    
    # Generate chart if possible
    try:
        if len(numeric_cols) >= 1:
            # Create a simple chart
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # If there's a categorical column, group by it
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols and numeric_cols:
                # Use first categorical and first numeric column
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                # Group and plot
                grouped = df.groupby(cat_col)[num_col].sum().head(10)  # Top 10
                grouped.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'{num_col} by {cat_col}')
                ax.set_xlabel(cat_col)
                ax.set_ylabel(num_col)
            else:
                # Just plot first numeric column
                df[numeric_cols[0]].head(20).plot(kind='line', ax=ax, color='blue')
                ax.set_title(f'Trend of {numeric_cols[0]}')
                ax.set_xlabel('Index')
                ax.set_ylabel(numeric_cols[0])
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
    except Exception as e:
        print(f"Chart generation failed: {e}")
    
    return stats, chart_base64


def create_data_summary(df: pd.DataFrame) -> str:
    """Create text summary of DataFrame for AI analysis."""
    summary_parts = [
        f"Dataset with {len(df)} rows and {len(df.columns)} columns.",
        f"Columns: {', '.join(df.columns.tolist()[:10])}",  # First 10 columns
    ]
    
    # Add numeric summaries
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        for col in numeric_cols[:3]:  # First 3 numeric columns
            summary_parts.append(
                f"{col}: mean={df[col].mean():.2f}, sum={df[col].sum():.2f}"
            )
    
    # Add categorical value counts
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        for col in categorical_cols[:2]:  # First 2 categorical columns
            unique_vals = df[col].value_counts().head(3)
            summary_parts.append(
                f"{col} top values: {', '.join([f'{k}({v})' for k, v in unique_vals.items()])}"
            )
    
    return "\n".join(summary_parts)


@app.post("/analyze", response_model=SummaryResponse)
async def analyze_document(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    """
    Analyze a document or data file and provide summary, insights, and visualizations.
    
    Accepts one of:
    - PDF file
    - CSV/Excel file  
    - URL to content
    """
    
    # Validate input - must have exactly one source
    sources_provided = sum([file is not None, url is not None])
    if sources_provided == 0:
        raise HTTPException(
            status_code=400, 
            detail="Please provide either a file (PDF/CSV/Excel) or a URL"
        )
    if sources_provided > 1:
        raise HTTPException(
            status_code=400,
            detail="Please provide only one source: either a file or a URL"
        )
    
    content = ""
    data_stats = None
    chart_base64 = None
    
    # Process file upload
    if file:
        file_bytes = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith('.pdf'):
            # Extract text from PDF
            content = extract_text_from_pdf(file_bytes)
            
        elif filename.endswith(('.csv', '.xlsx', '.xls')):
            # Read data file
            df = read_csv_or_excel(file_bytes, filename)
            
            # Generate statistics and chart
            data_stats, chart_base64 = analyze_dataframe(df)
            
            # Create text summary for AI analysis
            content = create_data_summary(df)
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload PDF, CSV, or Excel file"
            )
    
    # Process URL
    elif url:
        content = extract_text_from_url(url)
    
    # Get AI-powered summary and insights
    if not content:
        raise HTTPException(status_code=400, detail="No content to analyze")
    
    analysis = summarize_text(content)
    
    # Prepare response
    response = SummaryResponse(
        summary=analysis["summary"],
        key_points=analysis["key_points"],
        ai_insights=analysis["ai_insights"],
        chart_base64=chart_base64,
        data_stats=data_stats
    )
    
    return response


@app.get("/")
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "Document & Data Summarizer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze document/data (PDF, CSV, Excel, or URL)",
            "/docs": "GET - Interactive API documentation"
        },
        "instructions": "Send a file or URL to /analyze endpoint for AI-powered analysis"
    }


@app.get("/health")
async def health_check():
    """Check if Ollama service is available."""
    try:
        response = requests.get(f"http://localhost:11434/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "error"
    except:
        ollama_status = "disconnected"
    
    return {
        "status": "healthy",
        "ollama": ollama_status,
        "model": MODEL_NAME
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Document & Data Summarizer API...")
    print("Make sure Ollama is running: ollama serve")
    print(f"Using model: {MODEL_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Simple test script for the Document & Data Summarizer API.
This helps beginners understand how to interact with the API.
"""

import requests
import json
import base64
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test if the API and Ollama are running."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ API Status: {data['status']}")
        print(f"✓ Ollama Status: {data['ollama']}")
        print(f"✓ Model: {data['model']}")
        return True
    else:
        print("✗ API is not responding")
        return False


def test_with_csv():
    """Test with a CSV file if it exists."""
    csv_path = Path(__file__).parent.parent / "data" / "sales.csv"
    
    if not csv_path.exists():
        print(f"CSV file not found at {csv_path}")
        return
    
    print("\nTesting with CSV file...")
    
    with open(csv_path, 'rb') as f:
        files = {'file': ('sales.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/analyze", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Analysis successful!")
        print(f"\nSummary: {data['summary'][:200]}...")
        print(f"\nKey Points: {len(data['key_points'])} points found")
        for i, point in enumerate(data['key_points'][:3], 1):
            print(f"  {i}. {point}")
        
        if data.get('chart_base64'):
            print("✓ Chart generated successfully")
            # Optionally save the chart
            save_chart(data['chart_base64'], "output_chart.png")
        
        if data.get('data_stats'):
            print(f"✓ Data statistics: {data['data_stats']['shape']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def test_with_url():
    """Test with a sample URL."""
    print("\nTesting with URL...")
    
    # Using a simple example URL
    test_url = "https://www.example.com"
    
    data = {'url': test_url}
    response = requests.post(f"{BASE_URL}/analyze", data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ URL analysis successful!")
        print(f"\nSummary: {result['summary'][:200]}...")
        print(f"\nAI Insights: {result['ai_insights'][:200]}...")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def test_with_pdf():
    """Test with a PDF file if available."""
    # Check if there's a sample PDF in the data folder
    pdf_path = Path(__file__).parent.parent / "data" / "sample.pdf"
    
    if not pdf_path.exists():
        print(f"\nNo PDF test file found at {pdf_path}")
        return
    
    print("\nTesting with PDF file...")
    
    with open(pdf_path, 'rb') as f:
        files = {'file': ('sample.pdf', f, 'application/pdf')}
        response = requests.post(f"{BASE_URL}/analyze", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ PDF analysis successful!")
        print(f"\nSummary: {data['summary'][:200]}...")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def save_chart(base64_string, filename):
    """Save base64 chart to file."""
    try:
        image_data = base64.b64decode(base64_string)
        with open(filename, 'wb') as f:
            f.write(image_data)
        print(f"  Chart saved to {filename}")
    except Exception as e:
        print(f"  Could not save chart: {e}")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Document & Data Summarizer API Test Suite")
    print("=" * 50)
    
    # Check if API is running
    if not test_health():
        print("\n⚠ Please start the API first:")
        print("  python summarized_chatbot.py")
        return
    
    # Run tests
    test_with_csv()
    test_with_url()
    test_with_pdf()
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("\nTo explore more, visit:")
    print(f"  Interactive docs: {BASE_URL}/docs")
    print(f"  API root: {BASE_URL}/")


if __name__ == "__main__":
    main()

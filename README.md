# 📰 Financial Sentiment API with FinBERT

Analyze financial news sentiment using a domain-specific BERT model.

## Usage
- `POST /analyze` with JSON `{ "text": "Tesla shares rise on earnings beat." }`
- Returns `{ "positive": ..., "negative": ..., "neutral": ... }`

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload


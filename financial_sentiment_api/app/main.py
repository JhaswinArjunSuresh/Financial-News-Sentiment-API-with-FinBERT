from fastapi import FastAPI
from pydantic import BaseModel
from .sentiment import FinSentiment

app = FastAPI()
sentiment_analyzer = FinSentiment()

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze(input: TextInput):
    result = sentiment_analyzer.predict(input.text)
    return {"sentiment": result}

@app.get("/health")
def health():
    return {"status": "ok"}


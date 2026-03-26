from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("models/bert")

app = FastAPI(title="Hate Speech Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: int
    score: float
    scores: Dict[str, float]


_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
def _load_model() -> None:
    global _tokenizer, _model
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.to(_device)
    _model.eval()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "device": _device, "model_dir": str(MODEL_DIR)}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Hate Speech Detection API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


@app.get("/favicon.ico")
def favicon() -> Dict[str, str]:
    return {"detail": "Not Found"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    assert _tokenizer is not None
    assert _model is not None

    inputs = _tokenizer(req.text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    p0 = float(probs[0].cpu())
    p1 = float(probs[1].cpu())
    label = int(p1 >= p0)
    score = float(max(p0, p1))

    return PredictResponse(
        label=label,
        score=score,
        scores={"not_hate": p0, "hate": p1},
    )

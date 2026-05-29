from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.model.model import __version__ as model_version
from app.model.model import classes, prediction_pipeline

ALLOWED_TYPES = [
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/x-m4a",
    "audio/mp4",
]

app = FastAPI(
    title="Speech Command Classifier",
    description="Command prediction from a single audio file using a pretrained Keras model.",
)


class PredicitionOut(BaseModel):
    prediction: str
    confidence: float
    scores: list[float]


@app.get("/")
async def home() -> FileResponse:
    """Serve the frontend HTML."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    return {
        "health_check": "OK",
        "model_version": model_version,
    }


@app.get("/api/classes")
async def get_classes() -> dict[str, list[str]]:
    """Return the list of command classes."""
    return {"classes": classes}


@app.post("/api/predict", response_model=PredicitionOut)
async def predict_audio(
    file: UploadFile = File(...), feature_type: Literal["mfcc", "mel"] = "mel"
) -> dict[str, object]:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Upload a WAV or other audio file. content_type must be audio/wav, audio/mpeg, or audio/mp3. preferebly 1 second long",
        )

    body = await file.read()
    prediction = prediction_pipeline(body, feature_type=feature_type)
    return {
        "prediction": prediction["label"],
        "confidence": prediction["confidence"],
        "scores": prediction["scores"],
    }


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

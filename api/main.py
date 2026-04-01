"""FastAPI app entrypoint for handwriting recognition APIs."""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from api import dependencies
from api.schemas import (
    AnalyzeRequest,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    SheetScoreResponse,
    SimilarityRequest,
    SimilarityResponse,
)
from api.services import (
    classify_base64,
    compute_similarity_from_base64,
    score_handwriting_sheet,
)
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

APP_VERSION = "0.1.0"
MAX_UPLOAD_MB = 10


def _parse_cors_origins() -> list[str]:
    raw_origins = os.getenv(
        "API_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    )
    parsed = [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]
    return parsed if parsed else ["*"]


app = FastAPI(
    title="Handwriting Recognition API",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Healthcheck endpoint used by tests and local monitoring."""
    model_loaded = True
    try:
        _ = dependencies.get_model()
    except Exception:
        model_loaded = False

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "version": APP_VERSION,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest) -> ClassifyResponse:
    """Classify one base64 image into a character prediction."""
    try:
        model = dependencies.get_model()
        character, confidence = classify_base64(model, payload.image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Classification failed"
        ) from exc

    return ClassifyResponse(
        character=character, confidence=confidence
    )


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(payload: SimilarityRequest) -> SimilarityResponse:
    """Compute similarity for two base64 images and include image1 classification."""
    try:
        model = dependencies.get_model()
        encoder = dependencies.get_encoder()

        character, confidence = classify_base64(model, payload.image1)
        similarity_score = compute_similarity_from_base64(
            encoder,
            payload.image1,
            payload.image2,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Similarity scoring failed"
        ) from exc

    return SimilarityResponse(
        character=character,
        confidence=confidence,
        similarity=similarity_score,
        reference_id="uploaded_image2",
    )


@app.post("/analyze", response_model=SimilarityResponse)
def analyze(payload: AnalyzeRequest) -> SimilarityResponse:
    """Combined classify + similarity response for one input image."""
    try:
        model = dependencies.get_model()
        encoder = dependencies.get_encoder()

        character, confidence = classify_base64(model, payload.image)

        # For basic flow, use self-similarity unless an external reference store is added.
        similarity_score = compute_similarity_from_base64(
            encoder,
            payload.image,
            payload.image,
        )

        reference_id = payload.reference_id or "self"
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Analysis failed"
        ) from exc

    return SimilarityResponse(
        character=character,
        confidence=confidence,
        similarity=similarity_score,
        reference_id=reference_id,
    )


@app.post("/sheet/score", response_model=SheetScoreResponse)
async def sheet_score(
    file: UploadFile = File(...),
) -> SheetScoreResponse:
    """Upload handwriting sheet image and return row-level similarity scores."""
    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=422, detail="Uploaded file must be an image"
        )

    image_bytes = await file.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds maximum size of {MAX_UPLOAD_MB}MB",
        )

    try:
        detector = dependencies.get_ocr_detector()
        result = score_handwriting_sheet(
            image_bytes=image_bytes,
            detector=detector,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Sheet scoring failed"
        ) from exc

    return SheetScoreResponse(**result)


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True
    )

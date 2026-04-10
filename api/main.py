"""FastAPI app entrypoint for handwriting recognition APIs."""

from __future__ import annotations

import logging
import os
from time import perf_counter, process_time
from typing import Any

import uvicorn
from api import dependencies
from api.schemas import (
    AnalyzeRequest,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    SheetCreateResponse,
    SheetOptionsResponse,
    SheetScoreResponse,
    SimilarityRequest,
    SimilarityResponse,
)
from api.services import (
    build_sheet_options,
    classify_base64,
    compute_similarity_from_base64,
    create_sheet_artifacts,
    resolve_sheet_output_file,
    score_handwriting_sheet,
)
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

APP_VERSION = "0.1.0"
MAX_UPLOAD_MB = 10
logger = logging.getLogger(__name__)


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


def _is_env_flag_enabled(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _request_slow_threshold_ms() -> float:
    raw_value = os.getenv("API_SLOW_REQUEST_MS", "500")
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        parsed = 500.0
    return max(0.0, parsed)


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


@app.middleware("http")
async def request_metrics_middleware(
    request: Request,
    call_next: Any,
) -> Any:
    if not _is_env_flag_enabled(
        "API_ENABLE_REQUEST_METRICS", default=False
    ):
        return await call_next(request)

    wall_start = perf_counter()
    cpu_start = process_time()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        wall_ms = (perf_counter() - wall_start) * 1000.0
        cpu_ms = (process_time() - cpu_start) * 1000.0
        cpu_pct = (
            (cpu_ms / wall_ms) * 100.0 if wall_ms > 1e-9 else 0.0
        )
        status_code = getattr(response, "status_code", 500)
        slow_threshold_ms = _request_slow_threshold_ms()
        level = (
            logging.WARNING
            if wall_ms >= slow_threshold_ms
            else logging.INFO
        )
        logger.log(
            level,
            "request_metrics method=%s path=%s status=%s wall_ms=%.1f "
            "cpu_ms=%.1f cpu_pct=%.1f",
            request.method,
            request.url.path,
            status_code,
            wall_ms,
            cpu_ms,
            cpu_pct,
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
    include_details: bool = Form(False),
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
        app_config = dependencies.get_config()
        detector = dependencies.get_ocr_detector()
        result = score_handwriting_sheet(
            image_bytes=image_bytes,
            detector=detector,
            config=app_config,
            include_details=include_details,
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


@app.get("/sheet/options", response_model=SheetOptionsResponse)
def sheet_options() -> SheetOptionsResponse:
    """Return config-driven defaults and available server fonts for sheet generation."""
    try:
        payload = build_sheet_options(dependencies.get_config())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to load sheet options",
        ) from exc

    return SheetOptionsResponse(**payload)


@app.post("/sheet/create", response_model=SheetCreateResponse)
async def sheet_create(
    request: Request,
    language: str = Form("auto"),
    server_font: str | None = Form(None),
    custom_text: str | None = Form(None),
    output_basename: str | None = Form(None),
    font_size: int | None = Form(None),
    line_spacing: int | None = Form(None),
    word_spacing: int | None = Form(None),
    margin_left: int | None = Form(None),
    margin_right: int | None = Form(None),
    margin_top: int | None = Form(None),
    margin_bottom: int | None = Form(None),
    divide_horizontal: int | None = Form(None),
    divide_vertical: int | None = Form(None),
    show_vertical_line: bool | None = Form(None),
    font_file: UploadFile | None = File(None),
) -> SheetCreateResponse:
    """Generate handwriting sheet artifacts (PDF + preview PNG)."""
    uploaded_font_bytes: bytes | None = None
    uploaded_font_filename: str | None = None
    if font_file is not None:
        uploaded_font_filename = font_file.filename or "uploaded_font"
        uploaded_font_bytes = await font_file.read()

    overrides = {
        "font_size": font_size,
        "line_spacing": line_spacing,
        "word_spacing": word_spacing,
        "margin_left": margin_left,
        "margin_right": margin_right,
        "margin_top": margin_top,
        "margin_bottom": margin_bottom,
        "divide_horizontal": divide_horizontal,
        "divide_vertical": divide_vertical,
        "show_vertical_line": show_vertical_line,
    }

    try:
        artifacts = create_sheet_artifacts(
            dependencies.get_config(),
            language=language,
            server_font_id=server_font,
            uploaded_font_bytes=uploaded_font_bytes,
            uploaded_font_filename=uploaded_font_filename,
            custom_text=custom_text,
            overrides=overrides,
            output_basename=output_basename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Sheet generation failed",
        ) from exc

    pdf_url = str(
        request.url_for(
            "sheet_file",
            file_name=artifacts["pdf_filename"],
        )
    )
    preview_url = str(
        request.url_for(
            "sheet_file",
            file_name=artifacts["preview_image_filename"],
        )
    )

    return SheetCreateResponse(
        language=artifacts["language"],
        font_source=artifacts["font_source"],
        font_name=artifacts["font_name"],
        pdf_url=pdf_url,
        preview_image_url=preview_url,
        pdf_filename=artifacts["pdf_filename"],
        preview_image_filename=artifacts["preview_image_filename"],
    )


@app.get("/sheet/files/{file_name}", name="sheet_file")
def sheet_file(file_name: str) -> FileResponse:
    """Return generated PDF/PNG artifact by filename."""
    try:
        file_path = resolve_sheet_output_file(
            dependencies.get_config(),
            file_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media_type = (
        "application/pdf"
        if file_path.suffix.lower() == ".pdf"
        else "image/png"
    )
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name,
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True
    )

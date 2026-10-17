"""FastAPI app: POST /predict, GET /health, optional static hosting of the React build.

uv run weldsight-serve --run runs/<swrd_run>             # trained model
uv run weldsight-serve --stub                            # no checkpoint: heuristic stub
uv run weldsight-serve --run runs/<run> --static frontend/dist
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from weldsight.serve.inference import LockedPredictor, Predictor
from weldsight.serve.postprocess import build_response

MAX_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 100_000_000


def create_app(
    predictor: Predictor,
    cors_origins: list[str] | None = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    pred = predictor if isinstance(predictor, LockedPredictor) else LockedPredictor(predictor)
    app = FastAPI(title="WeldSight", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": pred.name,
            "stub": pred.stub,
            "classes": pred.classes,
            **pred.info,
        }

    @app.post("/api/predict")
    def predict(
        image: Annotated[UploadFile, File()],
        mc_samples: Annotated[int, Form(ge=2, le=30)] = 10,
        max_dim: Annotated[int, Form(ge=64, le=4096)] = 1024,
    ) -> dict:
        data = image.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"image larger than {MAX_BYTES // 2**20} MB")
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise HTTPException(415, "cannot decode image (use png / jpg / tif)")
        if gray.size > MAX_PIXELS:
            raise HTTPException(
                413, f"image has {gray.size / 1e6:.0f} MP, limit is {MAX_PIXELS // 10**6} MP"
            )
        out = build_response(pred.predict(gray, mc_samples), pred.classes, max_dim)
        out.update(model=pred.name, stub=pred.stub, mc_samples=mc_samples)
        return out

    if static_dir and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--run", help="run directory with config.yaml and best.pt (trained with masks)"
    )
    src.add_argument(
        "--stub", action="store_true", help="heuristic stand-in, no PyTorch/checkpoint needed"
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--static", help="directory with the built React app (frontend/dist)")
    ap.add_argument("--cors", nargs="*", help="allowed origins (default: the Vite dev server)")
    a = ap.parse_args()

    if a.stub:
        from weldsight.serve.inference import StubPredictor

        predictor: Predictor = StubPredictor()
    else:
        from weldsight.serve.inference import TorchPredictor

        predictor = TorchPredictor(a.run, device=a.device)
    uvicorn.run(create_app(predictor, a.cors, a.static), host=a.host, port=a.port)


if __name__ == "__main__":
    main()

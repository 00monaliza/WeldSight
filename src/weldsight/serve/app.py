"""FastAPI app: POST /predict, GET /health, optional static hosting of the React build.

uv run weldsight-serve --run runs/<swrd_run>             # trained model
uv run weldsight-serve --stub                            # no checkpoint: heuristic stub
uv run weldsight-serve --run runs/<run> --static frontend/dist
uv run weldsight-serve --run runs/<run> --data data/processed/riawelc_v1   # + dataset browser

With --data (default: the run's processed_dir, if it exists) the server also exposes the
prepared dataset read-only: GET /api/samples (patch list with split, specimen and labels)
and GET /api/samples/<patch_id>.png.
"""

from __future__ import annotations

import argparse
import io
import sys
import threading
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from weldsight.data.sources import to_uint8
from weldsight.serve.inference import LockedPredictor, Predictor
from weldsight.serve.postprocess import build_response

MAX_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 50_000_000
MAX_INFLIGHT = 2  # decoded images held in memory at once (decode happens before the model lock)


def create_app(
    predictor: Predictor,
    cors_origins: list[str] | None = None,
    static_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> FastAPI:
    pred = predictor if isinstance(predictor, LockedPredictor) else LockedPredictor(predictor)
    app = FastAPI(title="WeldSight", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    slots = threading.BoundedSemaphore(MAX_INFLIGHT)

    @app.middleware("http")
    async def limit_body(request, call_next):
        # reject before the multipart parser spools the body to disk
        if int(request.headers.get("content-length") or 0) > MAX_BYTES + 2**20:
            return JSONResponse({"detail": f"body larger than {MAX_BYTES // 2**20} MB"}, 413)
        return await call_next(request)

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
        try:  # dimensions from the header, before any raster is allocated
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
        except Image.DecompressionBombError:
            raise HTTPException(413, "image dimensions too large") from None
        except (UnidentifiedImageError, OSError):
            raise HTTPException(415, "cannot decode image (use png / jpg / tif)") from None
        if w * h > MAX_PIXELS:
            raise HTTPException(
                413, f"image has {w * h / 1e6:.0f} MP, limit is {MAX_PIXELS // 10**6} MP"
            )
        if not slots.acquire(timeout=30):
            raise HTTPException(503, "server busy, try again")
        try:
            raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise HTTPException(415, "cannot decode image (use png / jpg / tif)")
            gray = to_uint8(raw)  # same percentile windowing for 16-bit as in training data
            del raw, data
            result = pred.predict(gray, mc_samples)
        finally:
            slots.release()
        out = build_response(result, pred.classes, max_dim)
        out.update(model=pred.name, stub=pred.stub, mc_samples=mc_samples)
        return out

    samples = _load_samples(data_dir) if data_dir else None

    @app.get("/api/samples")
    def list_samples(
        split: str | None = None,
        cls: str | None = None,
        per_group: Annotated[int, Query(ge=1, le=200)] = 12,
        seed: int = 0,
    ) -> dict:
        """A stratified sample: up to `per_group` patches per (split, label) cell."""
        if samples is None:
            raise HTTPException(404, "server started without a dataset (--data)")
        df, card = samples
        demo = card.get("demo_subset") or {}
        d = df
        if split:
            d = d[d["split"] == split]
        if cls:
            d = d[d["label"] == cls]
        d = (
            d.sample(frac=1.0, random_state=seed)
            .groupby(["split", "label"])
            .head(per_group)
            .sort_values(["split", "label", "patch_id"])
        )
        return {
            "dataset": card.get("name"),
            "data_version": card.get("data_version"),
            "classes": card["class_names"],
            "supervision": card.get("supervision"),
            # a demo subset (weldsight-demo-subset) carries the full dataset's numbers
            "counts": demo.get("label_counts")
            or {
                sp: {lab: int(n) for lab, n in g["label"].value_counts().items()}
                for sp, g in df.groupby("split")
            },
            "specimens": demo.get("specimens")
            or {sp: sorted(g["group_id"].unique().tolist()) for sp, g in df.groupby("split")},
            "demo_subset": {"n_patches": demo["n_patches"], "of": demo["of"]} if demo else None,
            "items": [
                {
                    "id": r.patch_id,
                    "split": r.split,
                    "specimen": r.group_id,
                    "radiograph": r.radiograph_id,
                    "labels": [c for c in card["class_names"] if getattr(r, f"cls_{c}")],
                }
                for r in d.itertuples()
            ],
        }

    @app.get("/api/samples/{patch_id}.png")
    def sample_image(patch_id: str):
        if samples is None:
            raise HTTPException(404, "server started without a dataset (--data)")
        df, _ = samples
        row = df[df["patch_id"] == patch_id]  # look-up by id: no user-built paths
        if row.empty:
            raise HTTPException(404, "unknown patch id")
        return FileResponse(Path(data_dir) / row.iloc[0]["image"], media_type="image/png")

    if static_dir and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    return app


def _load_samples(data_dir: str | Path):
    from weldsight.data.dataset import load_manifest
    from weldsight.data.demo_subset import label_column

    df, card = load_manifest(data_dir)
    # one display label per patch: its class, "multi" for several, "none" for no defect
    df["label"] = label_column(df, card["class_names"])
    return df, card


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
    ap.add_argument(
        "--data", help="prepared dataset to browse (default: the run's processed_dir, if present)"
    )
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"WARNING: listening on {a.host} with no authentication. Anyone who can reach this port "
            "can run inference. Put it behind an authenticated proxy.",
            file=sys.stderr,
        )

    if a.stub:
        from weldsight.serve.inference import StubPredictor

        predictor: Predictor = StubPredictor()
    else:
        from weldsight.serve.inference import TorchPredictor

        predictor = TorchPredictor(a.run, device=a.device)
    data = a.data
    if data is None and a.run:
        from weldsight.config import load_config

        guess = Path(load_config(Path(a.run) / "config.yaml")["data"]["processed_dir"])
        # full prepared data first; else the small sample shipped with a released model
        shipped = Path(a.run) / "demo_data" / guess.name
        data = next((d for d in (guess, shipped) if (d / "manifest.csv").exists()), None)
        if data is not None:
            print(f"[serve] dataset browser: {data}")
    uvicorn.run(create_app(predictor, a.cors, a.static, data), host=a.host, port=a.port)


if __name__ == "__main__":
    main()

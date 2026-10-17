"""Server tests: postprocess + HTTP contract with the stub; TorchPredictor on a tiny random model."""

import base64

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from weldsight.serve.app import create_app  # noqa: E402
from weldsight.serve.inference import StubPredictor  # noqa: E402
from weldsight.serve.postprocess import Prediction, build_response, output_size  # noqa: E402

CLASSES = ["porosity", "crack"]


def _film(h=200, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(110, 6, (h, w)).clip(0, 255)
    cv2.circle(img, (160, 100), 6, 30, -1)  # one dark blob
    return img.astype(np.uint8)


def _png(a):
    return cv2.imencode(".png", a)[1].tobytes()


def _decode(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)


def test_output_size_matches_client_rule():
    assert output_size(400, 2000, 1000) == (200, 1000)
    assert output_size(100, 200, 1000) == (100, 200)  # never upscales


def test_postprocess_components_and_resize():
    K, h, w = 2, 100, 200
    probs = np.zeros((K + 1, h, w), np.float32)
    probs[0] = 1.0
    probs[:, 40:50, 60:80] = np.array([0.05, 0.05, 0.9], np.float32)[
        :, None, None
    ]  # confident crack
    probs[:, 70:76, 150:156] = np.array([0.6, 0.3, 0.1], np.float32)[:, None, None]  # weak porosity
    pe = np.zeros((h, w), np.float32)
    pe[70:76, 150:156] = 0.9
    res = build_response(Prediction(probs, pe, pe * 0.5), CLASSES, max_dim=100)
    assert (res["height"], res["width"]) == (50, 100)
    d = res["defects"]
    assert [x["cls"] for x in d] == ["porosity", "crack"]  # sorted by uncertainty
    assert d[1]["prob"] > 0.9 and d[0]["prob"] < 0.5
    assert d[1]["bbox"] == [30, 20, 10, 5]  # scaled by 0.5
    mask = _decode(res["mask_png"])
    assert mask.shape == (50, 100) and set(np.unique(mask)) == {0, 2}  # weak region is not painted
    assert _decode(res["maps"]["pe"]).max() == round(0.9 * 255)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(StubPredictor()))


def test_health(client):
    j = client.get("/api/health").json()
    assert j["status"] == "ok" and j["stub"] is True and "crack" in j["classes"]


def test_predict_contract(client):
    r = client.post(
        "/api/predict",
        files={"image": ("f.png", _png(_film()), "image/png")},
        data={"mc_samples": "4", "max_dim": "160"},
    )
    assert r.status_code == 200
    j = r.json()
    assert (j["width"], j["height"]) == (160, 100)
    for key in ("mask_png", "classes", "defects", "maps"):
        assert key in j
    assert _decode(j["mask_png"]).shape == (100, 160)
    assert _decode(j["maps"]["mi"]).dtype == np.uint8
    assert j["defects"], "the dark blob should be found"
    d = j["defects"][0]
    assert set(d) == {"cls", "prob", "bbox", "uncertainty"} and set(d["uncertainty"]) == {
        "pe",
        "mi",
    }


def test_predict_rejects_garbage_and_bad_params(client):
    r = client.post("/api/predict", files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 415
    r = client.post(
        "/api/predict",
        files={"image": ("f.png", _png(_film()), "image/png")},
        data={"mc_samples": "1"},
    )
    assert r.status_code == 422


def test_torch_predictor_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    from weldsight.models.unet import build_model
    from weldsight.serve.inference import TorchPredictor

    cfg = {
        "experiment": "tiny",
        "seed": 0,
        "model": {
            "arch": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 1,
        },
        "data": {"source": "swrd", "class_names": CLASSES, "patch_size": 64},
        "train": {"normalize": "per_patch", "device": "cpu"},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(
        {"model": build_model(cfg, len(CLASSES)).state_dict(), "epoch": 3}, tmp_path / "best.pt"
    )

    pred = TorchPredictor(tmp_path, device="cpu", batch_size=2)
    img = _film(100, 150)  # not a multiple of 64: exercises shifted last tiles
    out = pred.predict(img, 3)
    assert out.probs.shape == (3, 100, 150) and out.pe.shape == (100, 150)
    assert np.allclose(out.probs.sum(0), 1, atol=1e-4)
    assert 0 <= out.pe.min() and out.pe.max() <= 1 and out.mi.max() <= 1
    small = pred.predict(img[:40, :50], 2)  # smaller than one tile
    assert small.probs.shape == (3, 40, 50)

    j = (
        TestClient(create_app(pred))
        .post("/api/predict", files={"image": ("f.png", _png(img), "image/png")})
        .json()
    )
    assert j["stub"] is False and j["classes"] == CLASSES and "epoch 3" in j["model"]


def test_riawelc_run_is_refused(tmp_path):
    pytest.importorskip("torch")
    from weldsight.serve.inference import TorchPredictor

    cfg = {
        "model": {"encoder_name": "resnet18"},
        "data": {"source": "riawelc", "class_names": CLASSES},
        "train": {},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="untrained"):
        TorchPredictor(tmp_path)

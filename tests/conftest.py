import json
from pathlib import Path

import cv2
import numpy as np
import pytest

CLASSES = ["crack", "porosity", "lack_of_penetration"]
FOLDERS = {
    "Difetto1": "crack",
    "Difetto2": "porosity",
    "Difetto4": "lack_of_penetration",
    "NoDifetto": None,
}


@pytest.fixture
def fake_riawelc(tmp_path: Path) -> Path:
    """Mimics RIAWELC: 8 specimens x 2 films, 227x227 patches, split folders that leak."""
    rng = np.random.default_rng(0)
    root = tmp_path / "riawelc"
    splits = ["training", "validation", "testing"]
    for s_i in range(8):
        spec = f"RRT-{10 + s_i}R"
        for img in (1, 2):
            for folder in FOLDERS:
                for k in range(3):
                    split = splits[(s_i + k) % 3]  # deliberately leaky like the original
                    d = root / "DB - Copy" / split / folder
                    d.mkdir(parents=True, exist_ok=True)
                    arr = rng.integers(0, 255, (227, 227), dtype=np.uint8)
                    cv2.imwrite(str(d / f"{spec}_Img{img}_A80_S{k + 1}_[{k}][{s_i}].png"), arr)
    # an exact duplicate (same bytes) with the " - Copia" suffix in another split
    src = next((root / "DB - Copy" / "training" / "Difetto1").glob("*.png"))
    dst = root / "DB - Copy" / "testing" / "Difetto1" / (src.stem + " - Copia.png")
    dst.write_bytes(src.read_bytes())
    return root


@pytest.fixture
def fake_swrd(tmp_path: Path) -> Path:
    """Six 600x300 radiographs with LabelMe polygons of two classes."""
    rng = np.random.default_rng(1)
    root = tmp_path / "swrd"
    root.mkdir()
    for i in range(6):
        img = rng.integers(60, 200, (300, 600), dtype=np.uint8)
        cv2.imwrite(str(root / f"weld_{i}.png"), img)
        shapes = [
            {
                "label": "porosity",
                "shape_type": "polygon",
                "points": [
                    [50 + 40 * i, 50],
                    [90 + 40 * i, 50],
                    [90 + 40 * i, 90],
                    [50 + 40 * i, 90],
                ],
            },
            {"label": "crack", "shape_type": "rectangle", "points": [[400, 100], [560, 110]]},
        ]
        (root / f"weld_{i}.json").write_text(
            json.dumps({"shapes": shapes, "imagePath": f"weld_{i}.png"})
        )
    return root

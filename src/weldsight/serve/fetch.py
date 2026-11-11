"""Download a released checkpoint and check its sha256.

    uv run weldsight-fetch-model [--name riawelc_cpu_v1]

Saves models/<name>/best.pt next to the committed config.yaml, so the folder can be passed
to `weldsight-serve --run` or `weldsight-eval --run`.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "https://github.com/00monaliza/WeldSight"
MODELS = {
    "riawelc_cpu_v1": {
        "tag": "riawelc-cpu-v1",
        "asset": "riawelc_cpu_v1.pt",
        "sha256": "552b97c1e2c6205aa6e8ea2574f1b268072745955b2fe3922e8d2a58e2777f13",
    }
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, root: Path = Path("models")) -> Path:
    m = MODELS[name]
    out = root / name / "best.pt"
    if out.exists() and sha256(out) == m["sha256"]:
        print(f"[fetch] {out} already present")
        return out
    url = f"{REPO}/releases/download/{m['tag']}/{m['asset']}"
    tmp = out.with_suffix(".part")
    print(f"[fetch] {url}")
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    got = sha256(tmp)
    if got != m["sha256"]:
        tmp.unlink()
        sys.exit(f"sha256 mismatch: expected {m['sha256']}, got {got}")
    tmp.replace(out)
    print(f"[fetch] saved {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", default="riawelc_cpu_v1", choices=sorted(MODELS))
    ap.add_argument("--root", default="models")
    a = ap.parse_args(argv)
    fetch(a.name, Path(a.root))


if __name__ == "__main__":
    main()

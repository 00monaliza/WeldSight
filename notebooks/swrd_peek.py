"""Peek inside the SWRD zip before choosing a subset: sizes per folder, how crops relate
to raw films, label names and annotation format (reads ~600 small json over HTTP Range)."""

import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath as P

from weldsight.data.remote_subset import open_zip

URL = os.environ.get(
    "SWRD_URL",
    "https://www.googleapis.com/drive/v3/files/1Hc9_de5YAdXg46F-GfjcRKMEBRF9XRQi?alt=media&supportsAllDrives=true",
)
zf = open_zip(URL, None, os.environ.get("GDRIVE_TOKEN"))
infos = [i for i in zf.infolist() if not i.is_dir()]
by_dir = defaultdict(list)
for i in infos:
    by_dir[str(P(i.filename).parent)].append(i)

print("== size per folder")
for d, lst in sorted(by_dir.items()):
    tifs = [i.file_size for i in lst if i.filename.lower().endswith(".tif")]
    if tifs:
        print(f"{d:45s} {len(tifs):5d} tif, mean {sum(tifs) / len(tifs) / 1e6:6.1f} MB")
print(
    "other files:",
    [i.filename for i in infos if not i.filename.lower().endswith((".tif", ".json"))],
)

stem = lambda n: P(n).stem
raw = {stem(i.filename) for i in by_dir["Raw_data/images"]}
crop = [stem(i.filename) for d, lst in by_dir.items() if "crop_weld_images" in d for i in lst]
base = Counter(re.sub(r"^[A-Z]_", "", s) for s in crop)
print(
    "\n== crops vs raw: crop stems",
    len(crop),
    "| crops whose base name is a raw image:",
    sum(b in raw for b in base),
    "of",
    len(base),
    "| prefixes:",
    Counter(s[:2] for s in crop).most_common(5),
)
dates = Counter(m.group(1) for s in raw if (m := re.search(r"(\d{8})", s)))
print(
    "raw films:",
    len(raw),
    "| distinct dates:",
    len(dates),
    "| films per date (top):",
    dates.most_common(5),
)
print("raw name patterns:", Counter(re.sub(r"\d", "9", s) for s in raw).most_common(8))

random.seed(0)
for d in sorted(by_dir):
    if not d.endswith(("json", "/1", "/2")) or "images" in d:
        continue
    js = [i for i in by_dir[d] if i.filename.endswith(".json")]
    labels, shapes = Counter(), Counter()
    for i in random.sample(js, min(150, len(js))):
        j = json.loads(zf.read(i))
        for s in j.get("shapes", []):
            labels[s.get("label")] += 1
            shapes[s.get("shape_type")] += 1
    j = json.loads(zf.read(js[0]))
    print(
        f"\n== {d} ({len(js)} json, 150 sampled)\n keys: {sorted(j)}"
        f"\n imagePath: {j.get('imagePath')}  size: {j.get('imageWidth')}x{j.get('imageHeight')}"
        f"\n labels: {labels.most_common()}\n shape types: {dict(shapes)}"
    )

const REQUEST = `POST /api/predict
Content-Type: multipart/form-data

image        file    radiograph (png / jpg / tif), any size up to 100 MP
mc_samples   int     MC Dropout passes T (2–30, default 10)
max_dim      int     longest side of the returned maps (64–4096, default 1024)

GET /api/health      -> { status, model, stub, classes, device, ... }`;

const RESPONSE = `{
  "width": 1024, "height": 384,        // output size: same rule as the client canvas
  "classes": ["porosity", "inclusion", "crack",
              "undercut", "lack_of_fusion",
              "lack_of_penetration"],
  "mask_png": "<base64>",              // grey PNG, value = class idx, 0 = background
  "maps": {                            // grey PNG 0–255 → 0–1, same size
    "pe": "<base64>",                  // predictive entropy / ln(K+1)
    "mi": "<base64>"                   // mutual information, ×4 display gain
  },
  "defects": [{
    "cls": "crack",
    "prob": 0.91,                      // mean foreground probability of the region
    "bbox": [x, y, w, h],              // output pixels
    "uncertainty": { "pe": 0.18, "mi": 0.05 }   // mean over the region
  }],
  "model": "swrd_baseline (epoch 12)", "stub": false, "mc_samples": 10
}`;

export function ApiView() {
  return (
    <>
      <div className="results-head">
        <p className="kicker">Integration</p>
        <h2>One endpoint between this page and a model.</h2>
        <p className="lede">The FastAPI server lives in <code>src/weldsight/serve/</code>. The inference runs tiled MC Dropout exactly like val/test (<code>patch_size</code> tiles,
          per-patch normalisation) and returns the contract below.</p>
      </div>
      <div className="split">
        <section className="block">
          <h3>Run</h3>
          <pre><code>{`# trained on masks (SWRD / GDXray)
uv sync --extra serve
uv run weldsight-serve --run runs/<run_dir>

# no checkpoint yet: heuristic stub
uv run weldsight-serve --stub

# dev UI (proxies /api to :8000)
cd frontend && npm install && npm run dev

# or one process: build the UI, let FastAPI host it
cd frontend && npm run build
uv run weldsight-serve --run runs/<run_dir> --static frontend/dist`}</code></pre>
          <h3>Request</h3>
          <pre><code>{REQUEST}</code></pre>
        </section>
        <section className="block">
          <h3>Response <small>application/json</small></h3>
          <pre><code>{RESPONSE}</code></pre>
        </section>
      </div>
      <section className="block wide">
        <h3>Notes</h3>
        <ul className="notes">
          <li>Class index 0 is background, defects are 1..K in the order of <code>classes</code> (same convention as <code>data/sources.py</code>).</li>
          <li>A region is a <b>detection</b> when its mean foreground probability is ≥ 0.5. Weaker regions (≥ 0.2 pixelwise) come back as <b>suspects</b>, so an uncertain miss can still be referred.</li>
          <li>The page ranks a film by the <b>maximum</b> region uncertainty, matching the radiograph score used for the referral curve.</li>
          <li>A run trained on RIAWELC is refused at start-up: it has patch labels only, so its segmentation decoder is untrained.</li>
          <li>Thresholds on <code>pe</code>/<code>mi</code> are not calibrated. Choose them on the validation split, not by looking at test films.</li>
        </ul>
      </section>
    </>
  );
}

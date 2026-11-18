"""Kagu on a Hugging Face Space (Gradio, free ZeroGPU tier).

Upload a picture, get an editable SVG. No settings: the style is chosen automatically.
The Line Art engine lives in ./engine (lineart + flat2svg); everything runs on the CPU.
Small pictures are enlarged by Real-ESRGAN x4plus on the ZeroGPU card (gpu_upscale.py);
if the card is not available, the same model runs on the CPU through ONNX (slow).
"""
import base64
import os
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import spaces

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "engine"


def _fetch_engine():
    """The engine code is not in this public Space: it is downloaded at start-up from a private Hugging Face repo
    (KAGU_ENGINE_REPO, default Ellenanet/kagu-engine, with folders lineart/ and flat2svg/) using the Space secret
    ENGINE_TOKEN (a read-only token). A local ./engine folder, if present, is used as it is."""
    if (ENGINE / "lineart" / "lineart_assemble.py").exists():
        return
    from huggingface_hub import snapshot_download
    token = os.environ.get("ENGINE_TOKEN")
    if not token:
        raise RuntimeError("ENGINE_TOKEN secret is missing: the engine cannot be downloaded")
    repo = os.environ.get("KAGU_ENGINE_REPO", "Ellenanet/kagu-engine")
    err = None
    for kind in ("model", "space"):                   # the private engine repo may be a model repo or a Space
        try:
            snapshot_download(repo, repo_type=kind, token=token, local_dir=str(ENGINE),
                              allow_patterns=["lineart/**", "flat2svg/**"])
            return
        except Exception as e:
            err = e
    raise RuntimeError(f"engine repo {repo} could not be downloaded: {err}")


_fetch_engine()
sys.path.insert(0, str(ENGINE / "lineart"))


def _upscaler():
    """Path of the x4 ONNX upscale model, downloaded once into the Hugging Face cache."""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(os.environ.get("KAGU_UPSCALER_REPO", "SceneWorks/real-esrgan-onnx"),
                               os.environ.get("KAGU_UPSCALER_FILE", "real_esrgan_x4.onnx"))
    except Exception as e:
        print("Upscale model not available, small pictures use Lanczos:", e)
        return ""


os.environ["LINEART_UPSCAYL"] = ""                   # no Upscayl on the server
os.environ.setdefault("LINEART_ONNX_UPSCALER", _upscaler())

import lineart_assemble as la                         # noqa: E402  (needs the settings above)
from lineart_layers import Params                     # noqa: E402

LAST_UPSCALE = [""]                                   # which upscaler the last build used, for the info line

try:                                                  # x4 upscale on the GPU; the engine calls it in place of ONNX
    import gpu_upscale

    def _gpu_enlarge(bgr, up):
        try:
            big = gpu_upscale.upscale_x4(bgr)
            LAST_UPSCALE[0] = gpu_upscale.NAME
        except Exception as e:                        # GPU quota used up, card busy...: the CPU path takes over
            print("GPU upscale failed, using the CPU:", e)
            LAST_UPSCALE[0] = "ONNX on the CPU"
            return _cpu_enlarge(bgr, up)
        if up != 4:
            import cv2
            h, w = bgr.shape[:2]
            big = cv2.resize(big, (w * up, h * up), interpolation=cv2.INTER_AREA if up < 4 else cv2.INTER_LANCZOS4)
        return big, ""

    _cpu_enlarge = la.onnx_upscale
    la.onnx_upscale = _gpu_enlarge
except Exception as e:
    print("GPU upscale not available, using ONNX on the CPU:", e)


def trace(image_path, progress=gr.Progress()):
    if not image_path:
        raise gr.Error("Upload a picture first.")
    t0 = time.time()
    LAST_UPSCALE[0] = ""

    def step(text, i, n):
        progress((i / n) if n else None, desc=text)

    res = la.assemble_file(image_path, Params(), style="auto", progress=step)
    svg = res["svg"]
    out = Path(tempfile.mkdtemp()) / (Path(image_path).stem + ".svg")
    out.write_text(svg, encoding="utf-8")
    up = res.get("upscale_info") or {}
    info = (f"Style: {res.get('style')}  ·  "
            f"{svg.count('<path')} paths  ·  {time.time() - t0:.0f} s"
            + (f"  ·  enlarged: {LAST_UPSCALE[0] or up.get('method')}" if up.get("method") else ""))
    return _viewer(image_path, svg), str(out), info


def _original_uri(path, max_side=2048):
    """The uploaded picture as a PNG data URI for the left view (reduced so the page stays light)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    k = max_side / max(h, w)
    if k < 1:
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _viewer(image_path, svg):
    """Original on the left, SVG on the right; zoom and pan always move both views together (script in VIEWER_JS)."""
    svg_uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    pane = ('<div class="kv-pane"><div class="kv-label">{label}</div>'
            '<div class="kv-stage"><img src="{src}" draggable="false"></div></div>')
    return ('<div class="kv-wrap">' + pane.format(label="Original", src=_original_uri(image_path))
            + pane.format(label="SVG", src=svg_uri)
            + '</div><div class="kv-hint">Wheel: zoom · drag: move · double-click: fit</div>')


VIEWER_CSS = """
.kv-wrap{display:grid;grid-template-columns:1fr 1fr;gap:8px}
@media (max-width:700px){.kv-wrap{grid-template-columns:1fr}}
.kv-pane{position:relative;height:70vh;overflow:hidden;border-radius:8px;background:#fff;cursor:grab;
  touch-action:none;border:1px solid #e5e7eb}
.kv-pane:active{cursor:grabbing}
.kv-stage{position:absolute;inset:0;transform-origin:0 0}
.kv-stage img{width:100%;height:100%;object-fit:contain;user-select:none;pointer-events:none}
.kv-label{position:absolute;top:6px;left:8px;z-index:2;font-size:12px;color:#555;background:#ffffffcc;
  padding:1px 6px;border-radius:4px}
.kv-hint{font-size:12px;color:#888;margin-top:4px}
"""

VIEWER_JS = """
<script>
(() => {
  // One view state for every .kv-pane: zoom (k) and offset (x, y) in pane pixels.
  const v = {k: 1, x: 0, y: 0};
  let drag = null;
  const apply = () => document.querySelectorAll('.kv-stage').forEach(s =>
    s.style.transform = `translate(${v.x}px,${v.y}px) scale(${v.k})`);
  const fit = () => { v.k = 1; v.x = 0; v.y = 0; apply(); };
  const paneOf = e => e.target.closest && e.target.closest('.kv-pane');
  document.addEventListener('wheel', e => {
    const p = paneOf(e); if (!p) return;
    e.preventDefault();
    const r = p.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = Math.min(64, Math.max(1, v.k * Math.exp(-e.deltaY * 0.0015)));
    v.x = mx - (mx - v.x) * k / v.k; v.y = my - (my - v.y) * k / v.k; v.k = k;
    if (k === 1) { v.x = 0; v.y = 0; }
    apply();
  }, {passive: false});
  document.addEventListener('pointerdown', e => {
    const p = paneOf(e); if (!p) return;
    drag = {x: e.clientX - v.x, y: e.clientY - v.y}; p.setPointerCapture(e.pointerId);
  });
  document.addEventListener('pointermove', e => {
    if (!drag) return; v.x = e.clientX - drag.x; v.y = e.clientY - drag.y; apply();
  });
  document.addEventListener('pointerup', () => { drag = null; });
  document.addEventListener('dblclick', e => { if (paneOf(e)) fit(); });
  // a new result starts fitted
  new MutationObserver(() => { if (document.querySelector('.kv-stage:not([data-kv])')) {
    document.querySelectorAll('.kv-stage').forEach(s => s.dataset.kv = 1); fit(); } })
    .observe(document.documentElement, {childList: true, subtree: true});
})();
</script>
"""


with gr.Blocks(title="Kagu") as demo:
    gr.Markdown("# Kagu\nTurn an illustration into a clean, editable SVG.")
    with gr.Row():
        image = gr.Image(type="filepath", label="Picture (PNG or JPG)", height=220)
        with gr.Column():
            run = gr.Button("Make SVG", variant="primary")
            file = gr.File(label="Download SVG")
            info = gr.Markdown()
    viewer = gr.HTML()
    run.click(trace, [image], [viewer, file, info], concurrency_limit=1)

demo.launch(css=VIEWER_CSS, head=VIEWER_JS)

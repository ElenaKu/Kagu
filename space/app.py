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
    snapshot_download(os.environ.get("KAGU_ENGINE_REPO", "Ellenanet/kagu-engine"), token=token,
                      local_dir=str(ENGINE), allow_patterns=["lineart/**", "flat2svg/**"])


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
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    preview = (f'<div style="background:#fff;border-radius:8px;padding:8px">'
               f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;height:auto"></div>')
    return preview, str(out), info


with gr.Blocks(title="Kagu") as demo:
    gr.Markdown("# Kagu\nTurn an illustration into a clean, editable SVG.")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Picture (PNG or JPG)")
            run = gr.Button("Make SVG", variant="primary")
        with gr.Column():
            preview = gr.HTML(label="Result")
            file = gr.File(label="Download SVG")
            info = gr.Markdown()
    run.click(trace, [image], [preview, file, info], concurrency_limit=1)

demo.launch()

"""Kagu on a Hugging Face Space (Gradio, free ZeroGPU tier).

Upload a picture, get an editable SVG. No settings: the style is chosen automatically.
The Line Art engine lives in ./engine (lineart + flat2svg); everything runs on the CPU.
Small pictures are enlarged by an x4 ONNX model (Real-ESRGAN x4plus) downloaded on the first start.
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
sys.path.insert(0, str(ENGINE / "lineart"))


def _upscaler():
    """Path of the x4 ONNX upscale model, downloaded once into the Hugging Face cache."""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(os.environ.get("KAGU_UPSCALER_REPO", "qualcomm/Real-ESRGAN-x4plus"),
                               os.environ.get("KAGU_UPSCALER_FILE", "Real-ESRGAN-x4plus.onnx"))
    except Exception as e:
        print("Upscale model not available, small pictures use Lanczos:", e)
        return ""


os.environ["LINEART_UPSCAYL"] = ""                   # no Upscayl on the server
os.environ.setdefault("LINEART_ONNX_UPSCALER", _upscaler())

import lineart_assemble as la                         # noqa: E402  (needs the settings above)
from lineart_layers import Params                     # noqa: E402

@spaces.GPU
def _gpu_probe():
    """ZeroGPU needs at least one GPU function; tracing itself runs on the CPU and never calls this."""
    return True


def trace(image_path, progress=gr.Progress()):
    if not image_path:
        raise gr.Error("Upload a picture first.")
    t0 = time.time()

    def step(text, i, n):
        progress((i / n) if n else None, desc=text)

    res = la.assemble_file(image_path, Params(), style="auto", progress=step)
    svg = res["svg"]
    out = Path(tempfile.mkdtemp()) / (Path(image_path).stem + ".svg")
    out.write_text(svg, encoding="utf-8")
    up = res.get("upscale_info") or {}
    info = (f"Style: {res.get('style')}  ·  "
            f"{svg.count('<path')} paths  ·  {time.time() - t0:.0f} s"
            + (f"  ·  enlarged: {up.get('method')}" if up.get("method") else ""))
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

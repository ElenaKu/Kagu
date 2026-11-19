"""Kagu on a Hugging Face Space (free ZeroGPU tier): a custom HTML page (static/index.html) and a small FastAPI
backend in front of a minimal Gradio app, the same layout as the flat2svg Space.

Flow: the page uploads a PNG/JPG to /api/upload and shows it at once; "Make SVG" calls /api/convert; while it runs
the page polls /api/status for the current stage; the result is shown beside the source with synced zoom and pan.
The engine code is not in this public Space: it is downloaded at start-up from a private repo (ENGINE_TOKEN).
Small pictures are enlarged on the ZeroGPU card with the digital-art-4x model (gpu_upscale.py); without GPU quota
the engine uses Lanczos.
"""
import threading
import uuid
from collections import OrderedDict
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["GRADIO_SSR_MODE"] = "False"   # SSR would start its own server and hide our routes

import gradio as gr
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
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

LAST_UPSCALE = [""]


class UpscaleUnavailable(Exception):
    """Neither the GPU nor the processor could enlarge the picture."""


UPSCALE_HELP = ("This picture is small and has to be enlarged x4 before tracing, but enlarging is not available "
                "right now (the free GPU quota is used up and the processor could not do it). You can enlarge it "
                "x4 yourself, for example with Upscayl (free), and upload the enlarged picture: from 2048 px on the "
                "long side no enlarging is needed.")                                   # which upscaler the last build used, for the info line

try:                                                  # x4 upscale on the GPU; the engine calls it in place of ONNX
    import gpu_upscale

    def _gpu_enlarge(bgr, up):
        run = CURRENT_RUN[0] if "CURRENT_RUN" in globals() else None
        try:
            big = gpu_upscale.upscale_x4(bgr)
            LAST_UPSCALE[0] = gpu_upscale.NAME
        except Exception as e:                        # GPU quota used up or card busy: the same model on the CPU
            print("GPU upscale failed, using the CPU:", e)
            if run is not None:
                run["stage"] = ("The free GPU quota is used up, so the picture is enlarged on the processor.\n"
                                "This takes a few minutes.")
                run["notice"] = "GPU quota used up: enlarged on the processor"
            try:
                big = gpu_upscale.upscale_x4_cpu(bgr)
                LAST_UPSCALE[0] = "digital-art-4x on the processor, GPU quota used up"
            except Exception as e2:
                print("CPU upscale failed:", e2)
                raise UpscaleUnavailable() from e2
        if up != 4:
            import cv2
            h, w = bgr.shape[:2]
            big = cv2.resize(big, (w * up, h * up), interpolation=cv2.INTER_AREA if up < 4 else cv2.INTER_LANCZOS4)
        return big, ""

    _cpu_enlarge = la.onnx_upscale
    la.onnx_upscale = _gpu_enlarge
except Exception as e:
    print("GPU upscale not available, using ONNX on the CPU:", e)


CURRENT_RUN = [None]                    # the run being traced (one at a time, under WORK_LOCK)
_engine_enlarge = la._enlarge


def _enlarge_and_keep(bgr, alpha, trace, progress):
    """The engine's enlarge step; the enlarged picture is also saved for the page (view and download)."""
    out = _engine_enlarge(bgr, alpha, trace, progress)
    run = CURRENT_RUN[0]
    if run is not None and out[2] > 1:
        import cv2
        big = Path(run["path"]).with_name(Path(run["path"]).stem + "_x%d.png" % out[2])
        cv2.imwrite(str(big), out[0])
        run["big"], run["factor"] = str(big), out[2]
    return out


la._enlarge = _enlarge_and_keep

MAX_MB = 10
MAX_SIDE = 4000
KEEP_RUNS = 30
RUNS = OrderedDict()                    # run_id -> dict(path, name, width, height, svg, info, stage)
RUNS_LOCK = threading.Lock()
WORK_LOCK = threading.Lock()            # one tracing at a time
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="kagu_uploads_"))


def _get(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "This result has expired. Please upload the picture again.")
    return run


def _put(run_id, run):
    with RUNS_LOCK:
        RUNS[run_id] = run
        while len(RUNS) > KEEP_RUNS:
            _, old = RUNS.popitem(last=False)
            for f in (old["path"], old.get("big")):
                if f:
                    Path(f).unlink(missing_ok=True)


router = APIRouter()


@router.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@router.post("/api/upload")
def api_upload(file: UploadFile = File(...)):
    import cv2
    data = file.file.read(MAX_MB * 1024 * 1024 + 1)
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_MB} MB.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(415, "Please upload a PNG or JPG picture.")
    run_id = uuid.uuid4().hex[:12]
    path = UPLOAD_DIR / f"{run_id}{suffix}"
    path.write_bytes(data)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "This file could not be read as a picture.")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        path.unlink(missing_ok=True)
        raise HTTPException(413, f"Picture is larger than {MAX_SIDE} px on the long side.")
    _put(run_id, {"path": str(path), "name": Path(file.filename or "picture").stem, "width": w, "height": h,
                  "svg": None, "info": "", "stage": ""})
    return {"run_id": run_id, "width": w, "height": h}


@router.post("/api/convert")
def api_convert(run_id: str = Form(...)):
    run = _get(run_id)
    run["stage"] = "Waiting for the previous picture…"
    t0 = time.time()
    with WORK_LOCK:
        LAST_UPSCALE[0] = ""

        def step(text, i, n):
            run["stage"] = text

        run["stage"] = "Starting…"
        CURRENT_RUN[0] = run
        try:
            res = la.assemble_file(run["path"], Params(), style="auto", progress=step)
        except UpscaleUnavailable:
            run["stage"] = ""
            raise HTTPException(503, UPSCALE_HELP)
        except Exception as e:
            run["stage"] = ""
            raise HTTPException(500, f"Tracing failed: {type(e).__name__}")
        finally:
            CURRENT_RUN[0] = None
    svg = res["svg"]
    up = res.get("upscale_info") or {}
    info = (f"{_colour_count(svg)} colours · {svg.count('<path')} shapes · {len(svg.encode('utf-8')) / 1024:.0f} KB · {time.time() - t0:.0f} s"
            + (f" · enlarged x{up.get('factor')} ({LAST_UPSCALE[0] or up.get('engine')})" if up.get("factor", 1) > 1 else ""))
    run.update(svg=svg, info=info, stage="")
    return {"run_id": run_id, "info": info, "factor": run.get("factor", 1) if run.get("big") else 1}


def _colour_count(svg):
    """Distinct fill and stroke colours used in the SVG (attributes and inline styles)."""
    import re
    found = re.findall(r'(?:fill|stroke)\s*[:=]\s*["\']?\s*(#[0-9a-fA-F]{3,8}|rgb\([^)]*\))', svg)
    return len({c.lower() for c in found})


@router.get("/api/status/{run_id}")
def api_status(run_id: str):
    run = _get(run_id)
    return {"stage": run["stage"], "done": run["svg"] is not None}


@router.get("/api/src/{run_id}")
def api_src(run_id: str):
    return FileResponse(_get(run_id)["path"], headers={"Cache-Control": "no-store"})


@router.get("/api/big/{run_id}")
def api_big(run_id: str, download: int = 0):
    """The enlarged picture the SVG was traced from."""
    run = _get(run_id)
    if not run.get("big"):
        raise HTTPException(404, "This picture was not enlarged.")
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{run["name"]}_x{run["factor"]}.png"'
    return FileResponse(run["big"], media_type="image/png", headers=headers)


@router.get("/api/svg/{run_id}")
def api_svg(run_id: str, download: int = 0):
    run = _get(run_id)
    if not run["svg"]:
        raise HTTPException(404, "Not converted yet.")
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{run["name"]}.svg"'
    return Response(run["svg"], media_type="image/svg+xml", headers=headers)


# ZeroGPU only accepts a Space started through Gradio's own launch(): a tiny Gradio app is launched and our routes
# are put in front of Gradio's routes, so visitors only see the custom page at "/".
with gr.Blocks() as demo:
    gr.Markdown("Kagu")

demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
gradio_app = demo.app
_before = list(gradio_app.router.routes)
gradio_app.include_router(router)
_mine = gradio_app.router.routes[len(_before):]
gradio_app.router.routes[:] = _mine + _before
demo.block_thread()

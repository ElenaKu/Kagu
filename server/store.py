"""Projects: scene + undo history + background tracing jobs. In memory, files under KAGU_DATA_DIR."""
import copy
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import engine
from .scene import Scene

DATA = Path(os.environ.get("KAGU_DATA_DIR", Path(tempfile.gettempdir()) / "kagu"))
DATA.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = int(os.environ.get("KAGU_MAX_UPLOAD_MB", "40")) * 1024 * 1024
HISTORY = 40
POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("KAGU_TRACE_WORKERS", "1")))
PROJECTS = {}


class Project:
    def __init__(self, name, orig_path, mime, kind):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.orig_path = orig_path
        self.mime = mime
        self.kind = kind               # "svg" | "raster"
        self.status = "queued"         # queued | running | done | error
        self.message = ""
        self.progress = None
        self.scene = None
        self.base = None               # shapes right after tracing/import (for Reset)
        self.hist, self.redo = [], []
        self.info = {}
        self.lock = threading.RLock()
        self.created = time.time()

    # ---------- history ----------
    def apply(self, fn, *a, **kw):
        with self.lock:
            snap = self.scene.snapshot()
            try:
                msg = fn(*a, **kw)
            except Exception:
                self.scene.shapes = snap
                raise
            self.hist.append(snap)
            del self.hist[:-HISTORY]
            self.redo.clear()
            return msg

    def preview(self, fn, *a, **kw):
        """Run an operation on a throw-away copy; returns (message, scene copy)."""
        with self.lock:
            sc = copy.copy(self.scene)
            sc.shapes = self.scene.snapshot()
            msg = getattr(sc, fn)(*a, **kw)
            return msg, sc

    def undo(self):
        with self.lock:
            if self.hist:
                self.redo.append(self.scene.snapshot())
                self.scene.shapes = self.hist.pop()

    def do_redo(self):
        with self.lock:
            if self.redo:
                self.hist.append(self.scene.snapshot())
                self.scene.shapes = self.redo.pop()

    def reset(self):
        with self.lock:
            self.hist.append(self.scene.snapshot())
            self.redo.clear()
            self.scene.shapes = copy.deepcopy(self.base)


def _finish(p, svg_text, info=None):
    sc = Scene.from_svg(svg_text, Path(p.name).stem or "kagu")
    p.scene = sc
    p.base = copy.deepcopy(sc.shapes)
    p.info.update(info or {})
    p.status = "done"


def _run(p, style, colors):
    p.status = "running"

    def progress(msg, i=None, n=None):
        p.message = str(msg)
        p.progress = (i, n) if i is not None and n else None

    try:
        if p.kind == "svg":
            _finish(p, Path(p.orig_path).read_bytes())
        else:
            svg, info = engine.trace(p.orig_path, style=style, colors=colors, progress=progress)
            (Path(p.orig_path).parent / "traced.svg").write_text(svg, encoding="utf-8")
            _finish(p, svg, info)
        p.message = ""
    except Exception as e:
        p.status = "error"
        p.message = f"{type(e).__name__}: {e}"


def create(filename, data, style="auto", colors=None):
    if len(data) > MAX_UPLOAD:
        raise ValueError(f"File is larger than {MAX_UPLOAD // 1048576} MB")
    head = data[:512].lstrip().lower()
    is_svg = filename.lower().endswith(".svg") or head.startswith(b"<?xml") or b"<svg" in head
    ext = ".svg" if is_svg else (Path(filename).suffix.lower() or ".png")
    if not is_svg and ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        raise ValueError("Upload an SVG, PNG, JPEG or WebP file")
    mime = "image/svg+xml" if is_svg else {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    p = Project(filename, None, mime, "svg" if is_svg else "raster")
    d = DATA / p.id
    d.mkdir(parents=True, exist_ok=True)
    p.orig_path = str(d / ("original" + ext))
    Path(p.orig_path).write_bytes(data)
    PROJECTS[p.id] = p
    _evict()
    POOL.submit(_run, p, style, colors)
    return p


def _evict(keep=30):
    if len(PROJECTS) <= keep:
        return
    for pid in sorted(PROJECTS, key=lambda k: PROJECTS[k].created)[:len(PROJECTS) - keep]:
        PROJECTS.pop(pid, None)
        shutil.rmtree(DATA / pid, ignore_errors=True)


def get(pid):
    p = PROJECTS.get(pid)
    if not p:
        raise KeyError("Project not found (the server may have restarted)")
    return p

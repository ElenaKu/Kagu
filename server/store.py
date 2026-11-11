"""Projects: scene + undo history + a persistent job queue.

Everything a user needs to come back later is on disk under KAGU_DATA_DIR/<id>/:
  original.<ext>   the uploaded file
  meta.json        name, status, style, colours, engine info, timestamps
  base.pkl.gz      the traced result (what Reset returns to)
  scene.pkl.gz     the current edited scene (saved after every edit)
On start-up the server reloads all projects and puts unfinished jobs back in the queue.
"""
import copy
import gzip
import json
import os
import pickle
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
KEEP_DAYS = float(os.environ.get("KAGU_KEEP_DAYS", "14"))
IDLE_UNLOAD = 20 * 60
HISTORY = 40
POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("KAGU_TRACE_WORKERS", "1")))
PROJECTS = {}
_META = ("id", "name", "mime", "kind", "status", "message", "info", "created", "style", "colors", "finished", "owner")


class Project:
    def __init__(self, name, orig_path, mime, kind, style="auto", colors=None, pid=None, owner=None):
        self.id = pid or uuid.uuid4().hex[:12]
        self.name, self.orig_path, self.mime, self.kind = name, orig_path, mime, kind
        self.style, self.colors, self.owner = style, colors, owner
        self.status = "queued"         # queued | running | done | error | cancelled
        self.message = ""
        self.progress = None
        self.scene = None
        self.base = None
        self.hist, self.redo = [], []
        self.info = {}
        self.created = time.time()
        self.finished = None
        self.touched = time.time()
        self.lock = threading.RLock()

    @property
    def dir(self):
        return DATA / self.id

    # ---------- persistence ----------
    def save_meta(self):
        d = {k: getattr(self, k) for k in _META}
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(d), encoding="utf-8")
        tmp.replace(self.dir / "meta.json")

    def save_scene(self, base=False):
        def dump(obj, name):
            tmp = self.dir / (name + ".tmp")
            with gzip.open(tmp, "wb", compresslevel=1) as f:
                pickle.dump(obj, f, protocol=4)
            tmp.replace(self.dir / name)
        if base:
            dump(self.base, "base.pkl.gz")
        dump(self.scene, "scene.pkl.gz")

    def ensure(self):
        """Load the scene from disk if it was unloaded (or after a restart)."""
        with self.lock:
            self.touched = time.time()
            if self.status == "done" and self.scene is None:
                with gzip.open(self.dir / "scene.pkl.gz", "rb") as f:
                    self.scene = pickle.load(f)
                with gzip.open(self.dir / "base.pkl.gz", "rb") as f:
                    self.base = pickle.load(f)
                self.hist, self.redo = [], []

    def unload(self):
        with self.lock:
            if self.status == "done" and self.scene is not None:
                self.scene, self.base, self.hist, self.redo = None, None, [], []

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
            self.save_scene()
            return msg

    def undo(self):
        with self.lock:
            if self.hist:
                self.redo.append(self.scene.snapshot())
                self.scene.shapes = self.hist.pop()
                self.save_scene()

    def do_redo(self):
        with self.lock:
            if self.redo:
                self.hist.append(self.scene.snapshot())
                self.scene.shapes = self.redo.pop()
                self.save_scene()

    def reset(self):
        with self.lock:
            self.hist.append(self.scene.snapshot())
            self.redo.clear()
            self.scene.shapes = copy.deepcopy(self.base.shapes)
            self.save_scene()


# ---------- queue ----------
def waiting_of(owner):
    return sum(1 for q in PROJECTS.values() if q.owner == owner and q.status in ("queued", "running"))


def queue_info(p):
    """(position, total): position 1 = running now, 2 = next, ... None when not waiting."""
    if p.status not in ("queued", "running"):
        return None, 0
    waiting = sorted((q for q in PROJECTS.values() if q.status in ("queued", "running")), key=lambda q: q.created)
    return waiting.index(p) + 1, len(waiting)


def _finish(p, svg_text, info=None):
    sc = Scene.from_svg(svg_text, Path(p.name).stem or "kagu")
    p.scene = sc
    p.base = copy.deepcopy(sc)
    p.info.update(info or {})
    p.save_scene(base=True)


def _run(p):
    if p.status == "cancelled":
        return
    p.status, p.message = "running", ""
    p.save_meta()

    def progress(msg, i=None, n=None):
        p.message = str(msg)
        p.progress = (i, n) if i is not None and n else None

    try:
        if p.kind == "svg":
            _finish(p, Path(p.orig_path).read_bytes())
        else:
            svg, info = engine.trace(p.orig_path, style=p.style, colors=p.colors, progress=progress)
            (p.dir / "traced.svg").write_text(svg, encoding="utf-8")
            _finish(p, svg, info)
        if p.status == "cancelled":
            p.scene = p.base = None
            return
        p.status, p.message = "done", ""
    except Exception as e:
        p.scene = p.base = None
        p.status = "error"
        p.message = f"{type(e).__name__}: {e}"
    p.progress = None
    p.finished = time.time()
    p.save_meta()


def create(filename, data, style="auto", colors=None, owner=None):
    if len(data) > MAX_UPLOAD:
        raise ValueError(f"File is larger than {MAX_UPLOAD // 1048576} MB")
    head = data[:512].lstrip().lower()
    is_svg = filename.lower().endswith(".svg") or head.startswith(b"<?xml") or b"<svg" in head
    ext = ".svg" if is_svg else (Path(filename).suffix.lower() or ".png")
    if not is_svg and ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        raise ValueError("Upload an SVG, PNG, JPEG or WebP file")
    mime = "image/svg+xml" if is_svg else {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    p = Project(filename, None, mime, "svg" if is_svg else "raster", style, colors, owner=owner)
    p.dir.mkdir(parents=True, exist_ok=True)
    p.orig_path = str(p.dir / ("original" + ext))
    Path(p.orig_path).write_bytes(data)
    PROJECTS[p.id] = p
    p.save_meta()
    housekeeping()
    POOL.submit(_run, p)
    return p


def get(pid):
    p = PROJECTS.get(pid)
    if not p:
        raise KeyError("Project not found (it may have been deleted)")
    p.touched = time.time()
    return p


def remove(pid):
    p = get(pid)
    if p.status in ("queued", "running"):
        p.status = "cancelled"
        p.message = "Cancelled"
    PROJECTS.pop(pid, None)
    shutil.rmtree(p.dir, ignore_errors=True)


def cancel(pid):
    p = get(pid)
    if p.status == "queued":
        p.status, p.message = "cancelled", "Cancelled"
        p.finished = time.time()
        p.save_meta()
    elif p.status == "running":
        p.status, p.message = "cancelled", "Cancelled (the engine finishes its current step first)"
        p.save_meta()
    return p


def housekeeping():
    now = time.time()
    for p in list(PROJECTS.values()):
        if now - p.created > KEEP_DAYS * 86400 and p.status not in ("queued", "running"):
            PROJECTS.pop(p.id, None)
            shutil.rmtree(p.dir, ignore_errors=True)
        elif now - p.touched > IDLE_UNLOAD:
            p.unload()


def restore():
    """Reload projects from disk; unfinished jobs go back to the queue (oldest first)."""
    found = []
    for d in DATA.iterdir():
        m = d / "meta.json"
        if not m.exists():
            continue
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
            p = Project(meta["name"], None, meta["mime"], meta["kind"], meta.get("style", "auto"), meta.get("colors"), meta["id"])
            for k in _META:
                if k in meta:
                    setattr(p, k, meta[k])
            origs = list(d.glob("original.*"))
            p.orig_path = str(origs[0]) if origs else None
            p.touched = 0
            if p.status in ("queued", "running"):
                p.status, p.message = "queued", ""
                found.append(p)
            PROJECTS[p.id] = p
        except Exception:
            continue
    for p in sorted(found, key=lambda q: q.created):
        POOL.submit(_run, p)


restore()

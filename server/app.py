"""Kagu server: Starlette app. Run:  uvicorn server.app:app --port 8080"""
import re
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from . import auth, engine, render, store

WEB = Path(__file__).resolve().parent.parent / "web"
OPS = {
    "recolour": lambda sc, a: sc.op_recolour(a["ids"], a["color"]),
    "map_colours": lambda sc, a: sc.op_map_colours(a["mapping"], a.get("ids")),
    "merge": lambda sc, a: sc.op_merge(a["ids"], a.get("into", "largest"), a.get("scope")),
    "auto_merge": lambda sc, a: sc.op_auto_merge(a["delta"], a.get("ids")),
    "limit": lambda sc, a: sc.op_limit(a["n"], a.get("ids")),
    "hsl": lambda sc, a: sc.op_hsl(a["h"], a["s"], a["l"], a.get("ids")),
    "remove_background": lambda sc, a: sc.op_remove_background(),
    "delete": lambda sc, a: sc.op_delete(a["ids"], a.get("mode", "remove")),
}


def payload(p, extra=None):
    pos, total = store.queue_info(p)
    d = {"id": p.id, "name": p.name, "kind": p.kind, "status": p.status, "message": p.message,
         "progress": p.progress, "info": p.info, "created": p.created, "finished": p.finished,
         "queue": pos, "queue_total": total}
    if p.status == "done":
        p.ensure()
        sc = p.scene
        d.update(vb=sc.vb, defs=sc.defs, body=sc.body(), stats=sc.stats(), warnings=sc.warnings,
                 palette=[{"hex": h, "area": a} for h, a in sc.palette()], undo=len(p.hist), redo=len(p.redo))
    if extra:
        d.update(extra)
    return d


def me(request):
    """Owner id of the caller (None when login is off). Raises auth.AuthError when login is on and missing."""
    u = auth.user_of(request)
    return u["id"] if u else None


def err(msg, code=400):
    return JSONResponse({"error": str(msg)}, status_code=code)


async def index(request):
    return FileResponse(WEB / "index.html")


async def config(request):
    return JSONResponse({"auth": auth.ENABLED, "supabase_url": auth.URL, "anon_key": auth.ANON_KEY,
                         "max_queued": auth.MAX_QUEUED_PER_USER})


async def info(request):
    return JSONResponse({"engine": engine.available(), "engine_error": engine.error(), "styles": engine.styles(),
                         "max_upload_mb": store.MAX_UPLOAD // 1048576})


async def create_project(request: Request):
    try:
        owner = me(request)
    except auth.AuthError as e:
        return err(e, 401)
    if owner and store.waiting_of(owner) >= auth.MAX_QUEUED_PER_USER:
        return err(f"You already have {auth.MAX_QUEUED_PER_USER} jobs waiting. Wait for one to finish or stop one.", 429)
    form = await request.form()
    f = form.get("file")
    if f is None or not hasattr(f, "read"):
        return err("No file uploaded")
    data = await f.read()
    cols = form.get("colors")
    try:
        p = store.create(f.filename or "image", data, form.get("style") or "auto", int(cols) if cols else None, owner)
    except ValueError as e:
        return err(e)
    return JSONResponse(payload(p))


async def list_projects(request):
    """Summaries for the project ids this browser remembers (no scene data, cheap to poll)."""
    out = []
    owner = me(request)
    if owner:
        cands = [p for p in store.PROJECTS.values() if p.owner == owner]
    else:
        cands = [store.PROJECTS.get(i.strip()) for i in (request.query_params.get("ids") or "").split(",")[:50]]
    for p in cands:
        if not p:
            continue
        pos, total = store.queue_info(p)
        out.append({"id": p.id, "name": p.name, "kind": p.kind, "status": p.status, "message": p.message,
                    "progress": p.progress, "created": p.created, "finished": p.finished, "queue": pos,
                    "queue_total": total})
    return JSONResponse({"projects": out})


async def delete_project(request):
    try:
        store.remove(_project(request).id)
    except KeyError as e:
        return err(e.args[0], 404)
    return JSONResponse({"ok": True})


async def cancel_project(request):
    try:
        return JSONResponse(payload(store.cancel(_project(request).id)))
    except KeyError as e:
        return err(e.args[0], 404)


def _project(request):
    owner = me(request)
    p = store.get(request.path_params["pid"])
    if owner and p.owner != owner:
        raise KeyError("Project not found (it may have been deleted)")
    return p


async def get_project(request):
    try:
        return JSONResponse(payload(_project(request)))
    except KeyError as e:
        return err(e.args[0], 404)


async def original(request):
    try:
        p = _project(request)
    except KeyError as e:
        return err(e.args[0], 404)
    return FileResponse(p.orig_path, media_type=p.mime)


async def operate(request: Request):
    try:
        p = _project(request)
    except KeyError as e:
        return err(e.args[0], 404)
    if p.status != "done":
        return err("The project is not ready")
    p.ensure()
    a = await request.json()
    op = a.get("op")

    def work():
        if op == "undo":
            p.undo()
            return "Undone", None
        if op == "redo":
            p.do_redo()
            return "Redone", None
        if op == "reset":
            p.reset()
            return "Reset to the traced result", None
        if op == "select_specks":
            return "", {"ids": p.scene.find_specks(float(a.get("max", 20)))}
        if op == "scan":
            return "", {"problems": p.scene.scan()}
        fn = OPS.get(op)
        if not fn:
            raise ValueError("Unknown operation")
        if a.get("preview"):
            import copy
            sc = copy.copy(p.scene)
            sc.shapes = p.scene.snapshot()
            msg = fn(sc, a)
            return msg, {"preview": True, "body": sc.body(), "stats": sc.stats(),
                         "palette": [{"hex": h, "area": x} for h, x in sc.palette()]}
        return p.apply(lambda: fn(p.scene, a)), None

    try:
        msg, extra = await run_in_threadpool(work)
    except (ValueError, KeyError) as e:
        return err(e.args[0] if e.args else e)
    except Exception as e:
        return err(f"{type(e).__name__}: {e}", 500)
    if extra and extra.get("preview"):
        return JSONResponse(dict(extra, message=msg))
    return JSONResponse(payload(p, dict(extra or {}, message=msg)))


async def export(request: Request):
    try:
        p = _project(request)
    except KeyError as e:
        return err(e.args[0], 404)
    if p.status != "done":
        return err("The project is not ready")
    p.ensure()
    q = request.query_params
    fmt = (q.get("fmt") or "svg").lower()
    base = re.sub(r"[^\w.\-]+", "_", q.get("name") or Path(p.name).stem or "kagu")
    base = re.sub(r"\.(svg|png|jpe?g|webp)$", "", base, flags=re.I)

    def work():
        sc = p.scene
        if fmt == "svg":
            b = lambda k, d: q.get(k, str(d)).lower() in ("1", "true", "yes", "on")
            svg = render.export_svg(sc, group=b("group", 1), combine=b("combine", 1), adobe=b("adobe", 1),
                                    stock=b("stock", 0), dec=int(q.get("dec", 2)))
            return svg.encode("utf-8"), "image/svg+xml", "svg"
        w = int(q.get("w") or sc.vb[2])
        h = int(q.get("h") or round(w * sc.vb[3] / sc.vb[2]))
        return render.export_raster(sc, fmt, w, h, q.get("bg", "transp"), q.get("bgc", "#ffffff"),
                                    int(q.get("q", 92)), q.get("dpi", "1") in ("1", "true", "on"))

    try:
        data, mime, ext = await run_in_threadpool(work)
    except (ValueError, RuntimeError) as e:
        return err(e)
    return Response(data, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{base}.{ext}"'})


routes = [
    Route("/", index),
    Route("/api/info", info),
    Route("/api/config", config),
    Route("/api/projects", create_project, methods=["POST"]),
    Route("/api/projects", list_projects),
    Route("/api/projects/{pid}", get_project),
    Route("/api/projects/{pid}", delete_project, methods=["DELETE"]),
    Route("/api/projects/{pid}/cancel", cancel_project, methods=["POST"]),
    Route("/api/projects/{pid}/original", original),
    Route("/api/projects/{pid}/op", operate, methods=["POST"]),
    Route("/api/projects/{pid}/export", export),
]
async def on_auth_error(request, exc):
    return err(exc, 401)


app = Starlette(routes=routes, exception_handlers={auth.AuthError: on_auth_error})

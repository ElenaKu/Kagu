"""Optional mirror of project files in Supabase Storage.

Free hosts (Hugging Face Spaces, Cloud Run) wipe the local disk on restart. When SUPABASE_URL and
SUPABASE_SERVICE_KEY are set, every project file is copied to a private bucket in the background,
and the server pulls missing projects back from it at start-up. Without them nothing changes.

Env: SUPABASE_URL, SUPABASE_SERVICE_KEY (secret, server only), KAGU_BUCKET (default "kagu").
"""
import json
import os
import queue
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
BUCKET = os.environ.get("KAGU_BUCKET", "kagu")
ENABLED = bool(URL and KEY)
FILES = ("meta.json", "base.pkl.gz", "scene.pkl.gz")

_pending = {}               # (pid, name) -> local path; the latest save wins
_lock = threading.Lock()
_wake = queue.Queue()
_started = False


def _req(method, path, body=None, ctype=None, extra=None, timeout=120):
    headers = {"Authorization": "Bearer " + KEY, "apikey": KEY}
    if ctype:
        headers["Content-Type"] = ctype
    headers.update(extra or {})
    r = urllib.request.Request(f"{URL}/storage/v1/{path}", data=body, method=method, headers=headers)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.read()


def _obj(pid, name):
    return f"{BUCKET}/{urllib.parse.quote(pid)}/{urllib.parse.quote(name)}"


def _ensure_bucket():
    try:
        _req("POST", "bucket", json.dumps({"id": BUCKET, "name": BUCKET, "public": False}).encode(), "application/json")
    except urllib.error.HTTPError:
        pass                # already exists


def _worker():
    while True:
        _wake.get()
        while True:
            with _lock:
                if not _pending:
                    break
                key = next(iter(_pending))
                path = _pending.pop(key)
            pid, name = key
            try:
                if path is None:                           # delete the whole project
                    names = [f"{pid}/{n}" for n in _list(pid)]
                    if names:
                        _req("DELETE", BUCKET, json.dumps({"prefixes": names}).encode(), "application/json")
                else:
                    data = Path(path).read_bytes()
                    _req("POST", "object/" + _obj(pid, name), data, "application/octet-stream", {"x-upsert": "true"})
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[kagu] storage sync failed for {pid}/{name}: {e}")


def _start():
    global _started
    if not _started:
        _started = True
        threading.Thread(target=_worker, daemon=True).start()


def push(pid, name, path):
    """Queue a file upload (returns immediately)."""
    if not ENABLED:
        return
    _start()
    with _lock:
        _pending[(pid, name)] = str(path)
    _wake.put(1)


def drop(pid):
    """Queue removal of a project's files."""
    if not ENABLED:
        return
    _start()
    with _lock:
        for k in [k for k in _pending if k[0] == pid]:
            _pending.pop(k)
        _pending[(pid, "*")] = None
    _wake.put(1)


def _list(prefix):
    out, offset = [], 0
    while True:
        body = json.dumps({"prefix": prefix, "limit": 1000, "offset": offset}).encode()
        items = json.loads(_req("POST", "object/list/" + BUCKET, body, "application/json"))
        out += [i["name"] for i in items]
        if len(items) < 1000:
            return out
        offset += 1000


def pull(data_dir):
    """At start-up: download every project that is in the bucket but not on the local disk."""
    if not ENABLED:
        return
    try:
        _ensure_bucket()
        for pid in _list(""):
            d = Path(data_dir) / pid
            if (d / "meta.json").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            for name in _list(pid):
                data = _req("GET", "object/" + _obj(pid, name), timeout=300)
                (d / name).write_bytes(data)
        print("[kagu] projects restored from storage")
    except Exception as e:
        print(f"[kagu] could not restore from storage: {e}")

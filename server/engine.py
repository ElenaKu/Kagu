"""Adapter to the Line Art engine (the `lineart` and `flat2svg` folders of the scripts repository).

Set KAGU_ENGINE_DIR to the folder that contains `lineart/` and `flat2svg/` (default: ../scripts next to this repo).
Without the engine the server still runs: SVG uploads and every edit/export operation work, raster tracing is refused
with a clear message.
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
_DIR = Path(os.environ.get("KAGU_ENGINE_DIR", HERE.parent / "scripts"))
_mod = None
_err = None


def _load():
    global _mod, _err
    if _mod or _err:
        return
    lineart = _DIR / "lineart"
    if not (lineart / "lineart_assemble.py").exists():
        _err = f"Line Art engine not found in {_DIR} (set KAGU_ENGINE_DIR)"
        return
    sys.path.insert(0, str(lineart))
    try:
        import lineart_assemble as la
        import lineart_styles as ls
        from lineart_layers import Params
        _mod = {"la": la, "ls": ls, "Params": Params}
    except Exception as e:  # missing opencv, scikit-image, ...
        _err = f"Line Art engine could not be loaded: {e}"


def available():
    _load()
    return _mod is not None


def error():
    _load()
    return _err


def styles():
    _load()
    if not _mod:
        return []
    ls = _mod["ls"]
    out = [{"key": "auto", "label": "Auto"}]
    for k in ls.style_names():
        try:
            rec = ls.recipe(k)
            if rec.get("route") is None:
                continue
            out.append({"key": k, "label": rec.get("label", k)})
        except Exception:
            continue
    return out


def trace(path, style="auto", colors=None, progress=None):
    """Raster file -> SVG text through the engine."""
    _load()
    if not _mod:
        raise RuntimeError(_err)
    kw = {}
    if colors:
        kw["colors"] = int(colors)
    res = _mod["la"].assemble_file(path, _mod["Params"](), style=style, progress=progress, **kw)
    return res["svg"], {"style": res.get("style"), "route": res.get("route"), "seconds": res.get("seconds")}

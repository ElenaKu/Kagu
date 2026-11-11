"""Rendering and export: SVG serialization, raster export (PNG/JPEG/WebP), neighbour colours."""
import io
import shutil
import subprocess
import tempfile
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from . import colour as C
from . import geom as G

MAX_SIDE = 12000


def svg_document(sc, w=None, h=None, body=None):
    """Plain SVG of the scene for rasterising (no data-* attributes needed, but harmless)."""
    vb = sc.vb
    size = f' width="{w}" height="{h}"' if w and h else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb[0]} {vb[1]} {vb[2]} {vb[3]}"{size}>'
            f"{sc.defs}{body if body is not None else sc.body()}</svg>")


def export_svg(sc, group=True, combine=True, adobe=True, stock=False, dec=2):
    items = []
    for s in sc.shapes:
        extra = "".join(f" {k}={quoteattr(v)}" for k, v in s["extra"].items())
        items.append({"fill": s["fill"], "d": G.to_d(G.parse_d(s["d"]), dec), "ex": extra,
                      "tr": s["tr"], "plain": s["tr"] is None and "stroke" not in s["extra"]})
        if s["tr"]:
            items[-1]["ex"] += ' transform="matrix(%s)"' % ",".join(f"{v:.6g}" for v in s["tr"])
    paths = []
    for it in items:
        last = paths[-1] if paths else None
        if combine and last and it["plain"] and last["plain"] and last["fill"] == it["fill"] and last["ex"] == it["ex"]:
            last["d"] += it["d"]
        else:
            paths.append(dict(it))
    nl = "\n" if adobe else ""

    def ps(p, with_fill):
        return f'<path{" fill=" + quoteattr(p["fill"]) if with_fill else ""}{p["ex"]} d="{p["d"]}"/>'

    body, used, k = "", {}, 0
    if group:
        while k < len(paths):
            j = k
            while j < len(paths) and paths[j]["fill"] == paths[k]["fill"]:
                j += 1
            run, f = paths[k:j], paths[k]["fill"]
            if len(run) == 1:
                body += ps(run[0], True) + nl
            else:
                base = "c_" + "".join(ch for ch in f if ch.isalnum())
                used[base] = used.get(base, 0) + 1
                gid = f' id="{base}{"_" + str(used[base]) if used[base] > 1 else ""}"' if adobe else ""
                body += f'<g{gid} fill={quoteattr(f)}>{nl}{nl.join(ps(p, False) for p in run)}{nl}</g>{nl}'
            k = j
    else:
        body = nl.join(ps(p, True) for p in paths)
    x, y, w, h = sc.vb
    head = '<?xml version="1.0" encoding="UTF-8"?>\n' if adobe else ""
    head += (f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="{x:g} {y:g} {w:g} {h:g}"'
             + (f' width="{w:g}px" height="{h:g}px"' if adobe else "") + f">{nl}")
    if stock:
        head += f"<title>{escape(sc.name)}</title>{nl}"
    if sc.defs and "url(" in body:
        head += sc.defs + nl
    return head + body + "</svg>"


def _rsvg(svg, w, h, bg=None):
    exe = shutil.which("rsvg-convert")
    if exe:
        cmd = [exe, "-f", "png", "-w", str(w), "-h", str(h)]
        if bg:
            cmd += ["-b", bg]
        r = subprocess.run(cmd, input=svg.encode("utf-8"), capture_output=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError("rsvg-convert failed: " + r.stderr.decode("utf-8", "replace")[:300])
        return r.stdout
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=w, output_height=h, background_color=bg)
    except ImportError:
        raise RuntimeError("No SVG renderer found: install librsvg (rsvg-convert) or cairosvg")


def export_raster(sc, fmt, w, h, bg="transp", bg_colour="#ffffff", quality=92, dpi=True):
    from PIL import Image
    fmt = fmt.lower()
    w, h = int(w), int(h)
    if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE):
        raise ValueError(f"Size must be between 1 and {MAX_SIDE} px")
    colour = {"white": "#ffffff", "black": "#000000", "custom": bg_colour}.get(bg)
    if fmt == "jpeg" and not colour:
        colour = "#ffffff"
    png = _rsvg(svg_document(sc, w, h), w, h, colour)
    if fmt == "png":
        im = Image.open(io.BytesIO(png))
        buf = io.BytesIO()
        im.save(buf, "PNG", dpi=(300, 300) if dpi else None, optimize=False)
        return buf.getvalue(), "image/png", "png"
    im = Image.open(io.BytesIO(png))
    buf = io.BytesIO()
    if fmt == "jpeg":
        im.convert("RGB").save(buf, "JPEG", quality=int(quality), dpi=(300, 300) if dpi else (72, 72), subsampling=0)
        return buf.getvalue(), "image/jpeg", "jpg"
    if fmt == "webp":
        im.save(buf, "WEBP", quality=int(quality))
        return buf.getvalue(), "image/webp", "webp"
    raise ValueError("Unknown format")


def neighbour_colours(sc, ids, work_side=1400, ring=3):
    """For each shape id: the fill of the most common visible shape around it (excluding `ids`).

    Renders an id-map (every shape painted with a unique flat colour, no anti-aliasing) and reads the ring around
    each deleted shape, so the answer is what is really visible next to it.
    """
    from scipy import ndimage as ndi
    from PIL import Image
    W, H = sc.vb[2], sc.vb[3]
    sc_ = work_side / max(W, H)
    pw, ph = max(1, round(W * sc_)), max(1, round(H * sc_))
    parts = []
    for s in sc.shapes:
        code = s["id"] + 1
        col = "#%02x%02x%02x" % (code >> 16 & 255, code >> 8 & 255, code & 255)
        tr = f' transform="matrix({",".join(f"{v:.6g}" for v in s["tr"])})"' if s["tr"] else ""
        parts.append(f'<path d={quoteattr(s["d"])} fill="{col}" stroke="none"{tr}/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{sc.vb[0]} {sc.vb[1]} {W} {H}" width="{pw}" height="{ph}" '
           f'shape-rendering="crispEdges">{"".join(parts)}</svg>')
    img = np.array(Image.open(io.BytesIO(_rsvg(svg, pw, ph))).convert("RGBA"))
    labels = np.where(img[..., 3] > 0, (img[..., 0].astype(np.int64) << 16) | (img[..., 1].astype(np.int64) << 8) | img[..., 2], 0) - 0
    by_id = sc.by_id()
    skip = set(i + 1 for i in ids)
    out = {}
    for sid in ids:
        mask = labels == sid + 1
        if not mask.any():
            continue
        band = ndi.binary_dilation(mask, iterations=ring) & ~mask
        vals = labels[band]
        vals = vals[(vals > 0)]
        if not len(vals):
            continue
        keep = vals[~np.isin(vals, list(skip))]
        if not len(keep):
            continue
        u, n = np.unique(keep, return_counts=True)
        best = by_id.get(int(u[np.argmax(n)]) - 1)
        if best and C.is_hex(best["fill"]):
            out[sid] = best["fill"]
    return out

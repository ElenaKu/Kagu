"""Scene model: a list of filled shapes parsed from an SVG, plus all editing operations.

The scene is the single source of truth on the server. The browser only shows `body()` (the SVG markup of the
shapes) and asks for operations by shape id.
"""
import copy
import re
import subprocess
from xml.sax.saxutils import quoteattr

import numpy as np

from . import colour as C
from . import geom as G

SHAPES = {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line"}


def _local(tag):
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def _num(v, default=0.0):
    try:
        return float(re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v)).group(0))
    except Exception:
        return default


def _style(el):
    out = {}
    for part in (el.get("style") or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


_NAMED = {"black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
          "yellow": "#ffff00", "gray": "#808080", "grey": "#808080", "orange": "#ffa500", "purple": "#800080"}


def _color(v):
    if not v or v in ("none", "transparent"):
        return "none"
    v = v.strip().lower()
    if v.startswith("#"):
        return "#" + "".join(c * 2 for c in v[1:4]) if len(v) == 4 else v[:7]
    m = re.match(r"rgba?\(\s*([\d.]+%?)[ ,]+([\d.]+%?)[ ,]+([\d.]+%?)", v)
    if m:
        vals = [float(x[:-1]) * 2.55 if x.endswith("%") else float(x) for x in m.groups()]
        return C.rgb2hex(*vals)
    if v.startswith("url("):
        return v.split(")")[0] + ")"
    return _NAMED.get(v, "none")


class Scene:
    def __init__(self):
        self.vb = [0.0, 0.0, 100.0, 100.0]
        self.defs = ""
        self.shapes = []      # dicts: id, d, fill, extra, tr, bb, area, n, g
        self.next_id = 0
        self.name = "kagu"
        self.warnings = []

    # ---------- import ----------
    @classmethod
    def from_svg(cls, text, name="kagu"):
        from lxml import etree
        sc = cls()
        sc.name = name
        root = etree.fromstring(text.encode("utf-8") if isinstance(text, str) else text,
                                etree.XMLParser(remove_blank_text=True, resolve_entities=False, huge_tree=True))
        if _local(root.tag) != "svg":
            raise ValueError("The file is not an SVG")
        vb = [float(t) for t in re.split(r"[\s,]+", (root.get("viewBox") or "").strip()) if t]
        if len(vb) != 4 or vb[2] <= 0 or vb[3] <= 0:
            vb = [0, 0, _num(root.get("width"), 800), _num(root.get("height"), 600)]
        sc.vb = vb
        skipped = {"image": 0, "text": 0, "use": 0, "clip": 0}
        gid = [0]
        sc.defs = "".join(etree.tostring(e, encoding="unicode") for e in root.iter() if _local(e.tag) == "defs")

        def walk(el, tr, inh, g):
            tag = _local(el.tag)
            if tag in ("defs", "clipPath", "mask", "symbol", "pattern", "marker", "style", "title", "desc", "metadata"):
                return
            if tag in skipped:
                skipped[tag] += 1
                return
            st = _style(el)

            def get(k, dflt=None):
                return st.get(k, el.get(k, inh.get(k, dflt)))

            cur = dict(inh)
            for k in ("fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "fill-rule", "fill-opacity",
                      "stroke-opacity"):
                v = st.get(k, el.get(k))
                if v is not None:
                    cur[k] = v
            op = float(st.get("opacity", el.get("opacity", 1)) or 1) * float(inh.get("_op", 1))
            cur["_op"] = op
            t2 = G.mat_mul(tr, G.parse_transform(el.get("transform")))
            if el.get("clip-path") or el.get("mask"):
                skipped["clip"] += 1
            if tag == "g" or tag == "svg":
                gid[0] += 1
                for ch in el:
                    walk(ch, t2, cur, f"g{gid[0]}" if tag == "g" else g)
                return
            if tag not in SHAPES:
                return
            if (st.get("display") or el.get("display")) == "none" or (st.get("visibility") or el.get("visibility")) == "hidden":
                return
            d = el.get("d", "") if tag == "path" else sc._prim_d(el, tag)
            if not d:
                return
            fill = _color(cur.get("fill", "#000000"))
            stroke = _color(cur.get("stroke", "none"))
            if fill == "none" and stroke == "none" or op <= 0:
                return
            sc._add(d, fill, stroke, cur, op, t2, g or f"s{sc.next_id}")

        walk(root, (1, 0, 0, 1, 0, 0), {}, None)
        if not sc.shapes:
            raise ValueError("No drawable shapes found in the SVG")
        for k, label in (("image", "embedded raster images"), ("text", "text (convert it to outlines first)"),
                         ("use", "<use> instances"), ("clip", "clip paths and masks")):
            if skipped[k]:
                sc.warnings.append(f"{skipped[k]} {label} ignored")
        return sc

    @staticmethod
    def _prim_d(el, tag):
        g = lambda n: _num(el.get(n))
        if tag == "rect":
            w, h = g("width"), g("height")
            if w <= 0 or h <= 0:
                return ""
            rx, ry = g("rx"), g("ry")
            rx, ry = (rx, rx) if el.get("ry") is None else (rx, ry)
            if el.get("rx") is None:
                rx = ry
            return G.rect_d(g("x"), g("y"), w, h, rx, ry)
        if tag == "circle":
            return G.ellipse_d(g("cx"), g("cy"), g("r"), g("r")) if g("r") > 0 else ""
        if tag == "ellipse":
            return G.ellipse_d(g("cx"), g("cy"), g("rx"), g("ry")) if g("rx") > 0 and g("ry") > 0 else ""
        if tag in ("polygon", "polyline"):
            n = [float(t) for t in re.split(r"[\s,]+", (el.get("points") or "").strip()) if t]
            if len(n) < 4:
                return ""
            d = f"M{n[0]} {n[1]}" + "".join(f"L{n[i]} {n[i + 1]}" for i in range(2, len(n) - 1, 2))
            return d + ("Z" if tag == "polygon" else "")
        if tag == "line":
            return f"M{g('x1')} {g('y1')}L{g('x2')} {g('y2')}"
        return ""

    def _add(self, d, fill, stroke, cur, op, tr, g):
        extra = {}
        for k in ("fill-opacity", "fill-rule"):
            if cur.get(k) is not None and not (k == "fill-rule" and cur[k] == "nonzero"):
                extra[k] = str(cur[k])
        if stroke != "none":
            extra["stroke"] = stroke
            extra["stroke-width"] = str(_num(cur.get("stroke-width"), 1.0))
            for k in ("stroke-linecap", "stroke-linejoin", "stroke-opacity"):
                if cur.get(k) is not None and cur[k] not in ("butt", "miter"):
                    extra[k] = str(cur[k])
        if op < 1:
            extra["opacity"] = f"{op:.3f}"
        segs = G.parse_d(d)
        if not segs:
            return
        ident = all(abs(a - b) < 1e-9 for a, b in zip(tr, (1, 0, 0, 1, 0, 0)))
        if not ident:
            baked = G.apply_matrix(segs, tr)
            if baked is not None:
                segs, d, ident = baked, G.to_d(baked, 4), True
        bb, area, n = G.measure(segs)
        if not ident:
            # arcs under a transform: measure the transformed chord polygon for the bounding box only
            pts = [(tr[0] * x + tr[2] * y + tr[4], tr[1] * x + tr[3] * y + tr[5]) for s in G.flatten(segs) for x, y in s]
            if pts:
                bb = (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
            area *= abs(tr[0] * tr[3] - tr[1] * tr[2])
        self.shapes.append({"id": self.next_id, "d": d, "fill": fill, "extra": extra, "tr": None if ident else tr,
                            "bb": bb, "area": area, "n": n, "g": g})
        self.next_id += 1

    # ---------- views ----------
    def by_id(self):
        return {s["id"]: s for s in self.shapes}

    def body(self):
        out = []
        for s in self.shapes:
            a = f'<path d={quoteattr(s["d"])} fill={quoteattr(s["fill"])}'
            for k, v in s["extra"].items():
                a += f" {k}={quoteattr(v)}"
            if s["tr"]:
                a += ' transform="matrix(%s)"' % ",".join(f"{v:.6g}" for v in s["tr"])
            a += (f' data-lid="{s["id"]}" data-bb="{",".join(f"{v:.2f}" for v in s["bb"])}" data-a="{s["area"]:.2f}"'
                  f' data-n="{s["n"]}" data-g="{s["g"]}"/>')
            out.append(a)
        return "".join(out)

    def palette(self, ids=None):
        pool = self._pool(ids)
        m = {}
        for s in pool:
            if C.is_hex(s["fill"]):
                o = m.setdefault(s["fill"], [0, 0.0])
                o[0] += 1
                o[1] += s["area"]
        return sorted(((h, a) for h, (n, a) in m.items()), key=lambda t: -t[1])

    def stats(self):
        return {"shapes": len(self.shapes), "nodes": sum(s["n"] for s in self.shapes),
                "colours": len({s["fill"] for s in self.shapes if C.is_hex(s["fill"])})}

    def _pool(self, ids):
        if ids is None:
            return self.shapes
        ss = set(ids)
        return [s for s in self.shapes if s["id"] in ss]

    # ---------- operations (all mutate self; the store snapshots before calling) ----------
    def op_recolour(self, ids, color):
        n = 0
        for s in self._pool(ids):
            s["fill"] = C._color(color) if not C.is_hex(color) else color
            n += 1
        return f"{n} shape(s) recoloured"

    def op_map_colours(self, mapping, ids=None):
        n = 0
        for s in self._pool(ids):
            t = mapping.get(s["fill"])
            if t and t != s["fill"]:
                s["fill"] = t
                n += 1
        return f"{n} shape(s) recoloured"

    def op_merge(self, ids, into="largest", scope=None):
        cols = list({s["fill"] for s in self._pool(ids) if C.is_hex(s["fill"])})
        if len(cols) < 2:
            raise ValueError("Select shapes of two or more colours")
        area = dict(self.palette())
        if into == "light":
            t = max(cols, key=C.lightness)
        elif into == "dark":
            t = min(cols, key=C.lightness)
        else:
            t = max(cols, key=lambda c: area.get(c, 0))
        n = 0
        for s in self._pool(scope):
            if s["fill"] in cols and s["fill"] != t:
                s["fill"] = t
                n += 1
        return f"{len(cols)} colours merged into {t} ({n} shapes)"

    def op_auto_merge(self, delta, ids=None):
        pal = self.palette(ids)
        mp = C.map_by_threshold(pal, float(delta))
        before = len(pal)
        self.op_map_colours(mp, ids)
        return f"{before} -> {len(set(mp.values()))} colours"

    def op_limit(self, n, ids=None):
        pal = self.palette(ids)
        mp = C.map_by_limit(pal, int(n))
        self.op_map_colours(mp, ids)
        return f"{len(pal)} -> {len(set(mp.values()))} colours"

    def op_hsl(self, h, s, l, ids=None):
        cache = {}
        for sh in self._pool(ids):
            if C.is_hex(sh["fill"]):
                f = sh["fill"]
                if f not in cache:
                    cache[f] = C.hsl_shift(f, float(h), float(s), float(l))
                sh["fill"] = cache[f]
        return "Colour shift applied"

    def op_remove_background(self):
        W, H = self.vb[2], self.vb[3]
        best = None
        for s in self.shapes:
            b = s["bb"]
            if b[2] - b[0] >= W * .97 and b[3] - b[1] >= H * .97 and C.is_hex(s["fill"]) and (best is None or s["area"] > best["area"]):
                best = s
        if not best:
            raise ValueError("No full-size background shape found")
        self.shapes.remove(best)
        return f"Background removed ({best['fill']})"

    def op_delete(self, ids, mode="remove"):
        ids = set(ids)
        if not ids:
            raise ValueError("Nothing selected")
        recol = removed = 0
        if mode == "fill":
            from .render import neighbour_colours
            plan = neighbour_colours(self, ids)
        else:
            plan = {}
        for s in list(self.shapes):
            if s["id"] in ids:
                c = plan.get(s["id"])
                if c:
                    s["fill"] = c
                    recol += 1
                else:
                    self.shapes.remove(s)
                    removed += 1
        if mode == "fill":
            return f"{recol} filled with the neighbour colour, {removed} removed (hidden under other shapes or no neighbour)"
        return f"{removed} shape(s) removed"

    def find_specks(self, max_area):
        return [s["id"] for s in self.shapes if s["area"] <= max_area and "stroke" not in s["extra"]]

    def scan(self):
        out, seen = [], {}
        for s in self.shapes:
            w, h = s["bb"][2] - s["bb"][0], s["bb"][3] - s["bb"][1]
            key = s["d"] + "|" + s["fill"]
            if key in seen:
                out.append({"id": s["id"], "kind": "Duplicate", "text": f"same as #{seen[key]}"})
                continue
            seen[key] = s["id"]
            st = "stroke" in s["extra"]
            if not st and s["area"] < 2:
                out.append({"id": s["id"], "kind": "Speck", "text": f"{s['area']:.1f} px2"})
            elif not st and min(w, h) < .8 and max(w, h) / max(min(w, h), .01) > 12:
                out.append({"id": s["id"], "kind": "Sliver", "text": f"{w:.0f}x{h:.1f}"})
            elif s["n"] > 40 and s["n"] / (2 * (w + h) + 1) > 1.2:
                out.append({"id": s["id"], "kind": "Dense nodes", "text": f"{s['n']} nodes"})
        return out

    def snapshot(self):
        return copy.deepcopy(self.shapes)

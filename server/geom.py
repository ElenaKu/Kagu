"""SVG path helpers: parse, flatten, measure, serialize. Pure Python, no extra dependencies."""
import math
import re

_TOK = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")
_ARGN = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}


def parse_d(d):
    """Path data -> absolute segments [(cmd, [numbers])], cmd in M L C Q A Z."""
    toks = []
    for m in _TOK.finditer(d):
        toks.append(m.group(1) if m.group(1) else float(m.group(2)))
    out, i, cmd = [], 0, ""
    x = y = sx = sy = 0.0
    lc = lq = None
    while i < len(toks):
        if isinstance(toks[i], str):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                out.append(("Z", []))
                x, y = sx, sy
                lc = lq = None
                continue
        if not cmd or cmd in "Zz":
            i += 1
            continue
        c = cmd.upper()
        rel = cmd != c
        n = _ARGN[c]
        if i + n > len(toks) or any(isinstance(t, str) for t in toks[i:i + n]):
            i += 1
            continue
        a = toks[i:i + n]
        i += n
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if c == "M":
            x, y = a[0] + ox, a[1] + oy
            sx, sy = x, y
            out.append(("M", [x, y]))
            cmd = "l" if rel else "L"
            lc = lq = None
        elif c == "L":
            x, y = a[0] + ox, a[1] + oy
            out.append(("L", [x, y]))
            lc = lq = None
        elif c == "H":
            x = a[0] + ox
            out.append(("L", [x, y]))
            lc = lq = None
        elif c == "V":
            y = a[0] + oy
            out.append(("L", [x, y]))
            lc = lq = None
        elif c == "C":
            p = [a[0] + ox, a[1] + oy, a[2] + ox, a[3] + oy, a[4] + ox, a[5] + oy]
            out.append(("C", p))
            lc, lq = (p[2], p[3]), None
            x, y = p[4], p[5]
        elif c == "S":
            c1 = (2 * x - lc[0], 2 * y - lc[1]) if lc else (x, y)
            p = [c1[0], c1[1], a[0] + ox, a[1] + oy, a[2] + ox, a[3] + oy]
            out.append(("C", p))
            lc, lq = (p[2], p[3]), None
            x, y = p[4], p[5]
        elif c == "Q":
            p = [a[0] + ox, a[1] + oy, a[2] + ox, a[3] + oy]
            out.append(("Q", p))
            lq, lc = (p[0], p[1]), None
            x, y = p[2], p[3]
        elif c == "T":
            c1 = (2 * x - lq[0], 2 * y - lq[1]) if lq else (x, y)
            p = [c1[0], c1[1], a[0] + ox, a[1] + oy]
            out.append(("Q", p))
            lq, lc = c1, None
            x, y = p[2], p[3]
        elif c == "A":
            x, y = a[5] + ox, a[6] + oy
            out.append(("A", [a[0], a[1], a[2], a[3], a[4], x, y]))
            lc = lq = None
    return out


def _fmt(v, dec):
    s = f"{v:.{dec}f}".rstrip("0").rstrip(".") if dec > 0 else str(int(round(v)))
    return "0" if s in ("-0", "") else s


def to_d(segs, dec=2):
    o = []
    for c, p in segs:
        if c == "Z":
            o.append("Z")
        elif c == "A":
            o.append("A" + " ".join([_fmt(p[0], dec), _fmt(p[1], dec), _fmt(p[2], dec), str(int(p[3])), str(int(p[4])),
                                      _fmt(p[5], dec), _fmt(p[6], dec)]))
        else:
            o.append(c + " ".join(_fmt(v, dec) for v in p))
    return "".join(o)


def count_nodes(segs):
    return sum(1 for c, _ in segs if c != "Z")


def apply_matrix(segs, m):
    """Bake an affine matrix (a,b,c,d,e,f) into the segments. Returns None when arcs are present."""
    a, b, c, d, e, f = m
    out = []
    for cmd, p in segs:
        if cmd == "A":
            return None
        q = []
        for i in range(0, len(p), 2):
            q += [a * p[i] + c * p[i + 1] + e, b * p[i] + d * p[i + 1] + f]
        out.append((cmd, q))
    return out


def flatten(segs, steps=8):
    """Segments -> list of polylines (list of (x, y)); arcs are approximated by their chord."""
    subs, cur = [], []
    px = py = 0.0
    for c, p in segs:
        if c == "M":
            if len(cur) > 1:
                subs.append(cur)
            cur = [(p[0], p[1])]
            px, py = p[0], p[1]
        elif c == "L":
            cur.append((p[0], p[1]))
            px, py = p[0], p[1]
        elif c == "C":
            for k in range(1, steps + 1):
                t = k / steps
                u = 1 - t
                cur.append((u**3 * px + 3 * u * u * t * p[0] + 3 * u * t * t * p[2] + t**3 * p[4],
                            u**3 * py + 3 * u * u * t * p[1] + 3 * u * t * t * p[3] + t**3 * p[5]))
            px, py = p[4], p[5]
        elif c == "Q":
            for k in range(1, steps + 1):
                t = k / steps
                u = 1 - t
                cur.append((u * u * px + 2 * u * t * p[0] + t * t * p[2], u * u * py + 2 * u * t * p[1] + t * t * p[3]))
            px, py = p[2], p[3]
        elif c == "A":
            cur.append((p[5], p[6]))
            px, py = p[5], p[6]
        elif c == "Z" and cur:
            cur.append(cur[0])
    if len(cur) > 1:
        subs.append(cur)
    return subs


def measure(segs):
    """(bbox, area, nodes) of a path in its own coordinates. Area is the absolute sum of subpath areas."""
    subs = flatten(segs)
    xs = [q[0] for s in subs for q in s]
    ys = [q[1] for s in subs for q in s]
    if not xs:
        return (0, 0, 0, 0), 0.0, 0
    area = 0.0
    for s in subs:
        a = 0.0
        for i in range(len(s) - 1):
            a += s[i][0] * s[i + 1][1] - s[i + 1][0] * s[i][1]
        area += abs(a) / 2
    return (min(xs), min(ys), max(xs), max(ys)), area, count_nodes(segs)


def rect_d(x, y, w, h, rx=0.0, ry=0.0):
    if rx > 0 and ry > 0:
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        return (f"M{x + rx} {y}H{x + w - rx}A{rx} {ry} 0 0 1 {x + w} {y + ry}V{y + h - ry}A{rx} {ry} 0 0 1 {x + w - rx} {y + h}"
                f"H{x + rx}A{rx} {ry} 0 0 1 {x} {y + h - ry}V{y + ry}A{rx} {ry} 0 0 1 {x + rx} {y}Z")
    return f"M{x} {y}H{x + w}V{y + h}H{x}Z"


def ellipse_d(cx, cy, rx, ry):
    return f"M{cx - rx} {cy}A{rx} {ry} 0 1 0 {cx + rx} {cy}A{rx} {ry} 0 1 0 {cx - rx} {cy}Z"


def mat_mul(m, n):
    a1, b1, c1, d1, e1, f1 = m
    a2, b2, c2, d2, e2, f2 = n
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2, a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def parse_transform(s):
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(t) for t in re.split(r"[\s,]+", args.strip()) if t]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate":
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            r = math.radians(v[0])
            co, si = math.cos(r), math.sin(r)
            t = (co, si, -si, co, 0, 0)
            if len(v) == 3:
                t = mat_mul(mat_mul((1, 0, 0, 1, v[1], v[2]), t), (1, 0, 0, 1, -v[1], -v[2]))
        elif name == "skewX":
            t = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif name == "skewY":
            t = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else:
            continue
        m = mat_mul(m, t)
    return m

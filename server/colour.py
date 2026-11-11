"""Colour maths: hex <-> RGB, Lab distance, HSL shift, palette reduction."""
import colorsys
import math

_LAB = {}


def hex2rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    n = int(h, 16)
    return n >> 16 & 255, n >> 8 & 255, n & 255


def rgb2hex(r, g, b):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v)))) for v in (r, g, b))


def is_hex(f):
    return isinstance(f, str) and f.startswith("#")


def lab(h):
    v = _LAB.get(h)
    if v:
        return v

    def lin(c):
        c /= 255
        return ((c + .055) / 1.055) ** 2.4 if c > .04045 else c / 12.92

    r, g, b = (lin(c) for c in hex2rgb(h))

    def f(t):
        return t ** (1 / 3) if t > .008856 else 7.787 * t + 16 / 116

    x = f((r * .4124 + g * .3576 + b * .1805) / .95047)
    y = f(r * .2126 + g * .7152 + b * .0722)
    z = f((r * .0193 + g * .1192 + b * .9505) / 1.08883)
    v = (116 * y - 16, 500 * (x - y), 200 * (y - z))
    _LAB[h] = v
    return v


def delta_e(a, b):
    p, q = lab(a), lab(b)
    return math.dist(p, q)


def lightness(h):
    return lab(h)[0]


def hsl_shift(hex_, dh, ds, dl):
    r, g, b = (c / 255 for c in hex2rgb(hex_))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + dh / 360) % 1
    s = max(0.0, min(1.0, s * (1 + ds / 100)))
    l = l + (1 - l) * dl / 100 if dl >= 0 else l * (1 + dl / 100)
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l)), s)
    return rgb2hex(r * 255, g * 255, b * 255)


def map_by_threshold(pal, th):
    """pal: list of (hex, area) sorted by area desc. Returns {hex: representative}."""
    centers, out = [], {}
    for hx, _ in pal:
        best, bd = None, 1e9
        for k in centers:
            d = delta_e(hx, k)
            if d < bd:
                best, bd = k, d
        if best is not None and bd <= th:
            out[hx] = best
        else:
            centers.append(hx)
            out[hx] = hx
    return out


def map_by_limit(pal, n):
    """Agglomerative merge of the two closest colours (area-weighted keep) until n remain."""
    src = pal
    pre = None
    if len(src) > 400:
        pre = map_by_threshold(pal, 2.5)
        agg = {}
        for hx, a in pal:
            agg[pre[hx]] = agg.get(pre[hx], 0) + a
        src = list(agg.items())
    cl = [{"hex": h, "area": a, "mem": [h]} for h, a in src]
    while len(cl) > max(1, n):
        bi, bj, bd = 0, 1, 1e9
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                d = delta_e(cl[i]["hex"], cl[j]["hex"])
                if d < bd:
                    bi, bj, bd = i, j, d
        big, sm = (bi, bj) if cl[bi]["area"] >= cl[bj]["area"] else (bj, bi)
        cl[big]["mem"] += cl[sm]["mem"]
        cl[big]["area"] += cl[sm]["area"]
        cl.pop(sm)
    out = {}
    for c in cl:
        for m in c["mem"]:
            out[m] = c["hex"]
    if pre is not None:
        out = {h: out.get(pre[h], h) for h, _ in pal}
    return out

#!/usr/bin/env python3
"""Reflection_Eter — procedural, audio-reactive music video renderer.

Usage:
  python3 mv.py song.mp3 out.mp4                 # full render
  python3 mv.py song.mp3 stills/ --stills 5,30   # PNG stills at given seconds

Everything is rendered from code + the audio file: no external footage.
Scenes are drawn in luminance and graded through a teal-grey LUT taken
from the single's cover art.
"""
import argparse
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, signal

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

W, H = 1920, 1080
CH = 804                      # 2.39:1 content area
BAR_Y = (H - CH) // 2         # letterbox bar height
HZ = CH // 2                  # horizon / mirror line (the recurring light line)
FPS = 24

# ---- musical timeline (measured from the track) ---------------------------
BPM = 92.34
BEAT = 60.0 / BPM
BAR = 4 * BEAT
T0 = 0.307                    # first downbeat phase


def bar_t(n):
    return T0 + n * BAR


SECTIONS = [                  # (name, chapter label, start, end)
    ("intro",  "I.   mist",       0.0,        bar_t(8)),
    ("lake",   "II.  surface",    bar_t(8),   bar_t(24)),
    ("flower", "III. bloom",      bar_t(24),  bar_t(40)),
    ("film",   "IV.  frames",     bar_t(40),  143.9),
    ("void",   "V.   stillness",  143.9,      bar_t(64)),
    ("finale", "VI.  ether",      bar_t(64),  203.4),
    ("outro",  "VII. reflection", 203.4,      999.0),
]
SILENCE = (61.0, bar_t(24))   # the breath before the first drop
BUILD = (161.4, bar_t(64))    # riser into the finale

# ---- grading LUT: cover-art palette ----------------------------------------
_STOPS = np.array([0.0, 0.12, 0.3, 0.55, 0.8, 1.0])
_COLS = np.array([
    [4, 7, 9], [14, 24, 28], [36, 56, 62], [96, 122, 127],
    [176, 196, 197], [238, 245, 243]], dtype=np.float32)
LUT_N = 2048
_lx = np.linspace(0, 1, LUT_N)
LUT = np.stack([np.interp(_lx, _STOPS, _COLS[:, c]) for c in range(3)], 1).astype(np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def sstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def hash01(*k):
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5bd1e995) & 0xFFFFFFFF
    h ^= h >> 15
    return (h & 0xFFFFFF) / float(0x1000000)


# ============================================================================
# Audio analysis
# ============================================================================
def analyse(path, n_frames):
    ff = ffmpeg_exe()
    raw = subprocess.run([ff, "-v", "error", "-i", path, "-ac", "1", "-ar", "22050",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    sr, hop, n = 22050, 256, 2048
    f, t, S = signal.stft(x, sr, nperseg=n, noverlap=n - hop, boundary=None, padded=False)
    M = np.abs(S).astype(np.float32)
    afps = sr / hop

    def band(a, b):
        return np.sqrt((M[(f >= a) & (f < b)] ** 2).mean(0))

    sub = band(25, 130)
    rms = np.sqrt((M ** 2).mean(0))
    hi = band(2500, 9000)

    # kick onsets: peaks of positive sub-band log flux
    ls = np.log1p(sub / (np.percentile(sub, 95) + 1e-9) * 20)
    fl = np.maximum(0, np.diff(ls, prepend=ls[0]))
    fl = np.convolve(fl, np.ones(3) / 3, "same")
    thr = np.convolve(fl, np.ones(int(afps)) / int(afps), "same") * 1.6 + 0.02
    peaks, _ = signal.find_peaks(fl, height=thr, distance=int(0.28 * afps))
    kicks = t[peaks]
    kstr = fl[peaks] / (np.percentile(fl[peaks], 90) + 1e-9) if len(peaks) else np.zeros(0)

    # 56 log bands for the spectrum line
    edges = np.geomspace(45, 11000, 57)
    lo = np.searchsorted(f, edges[:-1])
    hi_ = np.maximum(np.searchsorted(f, edges[1:]), lo + 1)
    spec = np.stack([np.sqrt((M[lo[i]:hi_[i]] ** 2).mean(0) + 1e-12) for i in range(56)], 1)
    db = 20 * np.log10(spec + 1e-9)
    ref = np.percentile(db, 99, axis=0)
    spec = np.clip((db - (ref - 38)) / 38, 0, 1)

    vt = np.arange(n_frames) / FPS
    def at(arr):
        return np.interp(vt, t, arr)

    level = at(rms / np.percentile(rms, 97))
    level = np.clip(ndimage.gaussian_filter1d(level, FPS * 0.6), 0, 1.2)
    subn = np.clip(at(sub / np.percentile(sub, 97)), 0, 1.3)
    hin = np.clip(at(hi / np.percentile(hi, 97)), 0, 1.3)
    sp = np.stack([at(spec[:, i]) for i in range(56)], 1)
    # asymmetric smoothing: quick attack, slow release
    out = np.zeros_like(sp)
    prev = np.zeros(56)
    for i in range(n_frames):
        a = np.where(sp[i] > prev, 0.55, 0.12)
        prev = prev + (sp[i] - prev) * a
        out[i] = prev
    # kick pulse envelope
    pulse = np.zeros(n_frames)
    for k, s in zip(kicks, kstr):
        # peak on the frame nearest the kick (slightly early reads as "on" the beat)
        i0 = max(0, int(round((k - 0.02) * FPS)))
        d = np.maximum(0, np.arange(i0, min(n_frames, i0 + FPS * 2)) / FPS - k)
        pulse[i0:i0 + len(d)] = np.maximum(pulse[i0:i0 + len(d)], min(s, 1.2) * np.exp(-d / 0.24))
    return dict(level=level, sub=subn, hi=hin, spec=out, pulse=np.clip(pulse, 0, 1.2),
                kicks=kicks, kstr=kstr)


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


# ============================================================================
# Precomputed assets (built once per worker)
# ============================================================================
A = {}


def fbm(h, w, beta, seed):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((h, w)))
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    k[0, 0] = 1
    F *= k ** (-beta / 2)
    F[0, 0] = 0
    r = np.real(np.fft.ifft2(F))
    r = (r - r.mean()) / (r.std() + 1e-9)
    return r.astype(np.float32)


def fbm1d(n, beta, seed):
    rng = np.random.default_rng(seed)
    F = np.fft.rfft(rng.standard_normal(n))
    k = np.fft.rfftfreq(n)
    k[0] = 1
    F *= k ** (-beta / 2)
    F[0] = 0
    r = np.fft.irfft(F, n)
    return ((r - r.mean()) / r.std()).astype(np.float32)


def resizeF(a, w, h):
    return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((w, h), Image.BICUBIC))


def gauss_sprite(r, sigma):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return np.exp(-(x * x + y * y) / (2 * sigma * sigma)).astype(np.float32)


def bokeh_sprite(r):
    s = 2 * r + 3
    y, x = np.mgrid[:s, :s] - (s - 1) / 2
    d = np.sqrt(x * x + y * y)
    disc = np.clip(r - d + 0.5, 0, 1)
    rim = np.exp(-((d - r + 1.2) ** 2) / (2 * 0.9 ** 2)) * 0.6
    hl = np.exp(-((x + r * 0.35) ** 2 + (y + r * 0.35) ** 2) / (2 * (r * 0.22 + 0.6) ** 2)) * 0.8
    return (disc * 0.45 + rim * disc + hl * disc).astype(np.float32)


def blit_add(dst, spr, cx, cy, gain):
    h, w = spr.shape
    x0, y0 = int(round(cx)) - w // 2, int(round(cy)) - h // 2
    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(dst.shape[1], x1), min(dst.shape[0], y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    dst[dy0:dy1, dx0:dx1] += spr[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0] * gain


def text_mask(txt, font, size, tracking=0):
    f = ImageFont.truetype(os.path.join(FONT_DIR, font), size)
    ws = [f.getbbox(c)[2] if c != " " else size * 0.3 for c in txt]
    tw = int(sum(ws) + tracking * (len(txt) - 1)) + 20
    th = int(size * 2.2)
    im = Image.new("L", (tw, th), 0)
    d = ImageDraw.Draw(im)
    x = 10
    for c, w_ in zip(txt, ws):
        d.text((x, size * 0.15), c, font=f, fill=255)
        x += w_ + tracking
    a = np.asarray(im).astype(np.float32) / 255
    ys, xs = np.nonzero(a > 0.01)
    return a[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5]


def draw_flower(scale, sway, seed=3, droop=0.0):
    """Silhouette of a single-stem poppy (like the cover). Returns (alpha, anchor_x, anchor_y)
    where anchor is the stem base inside the returned array."""
    ss = 2
    Wc, Hc = int(560 * scale), int(900 * scale)
    im = Image.new("L", (Wc * ss, Hc * ss), 0)
    d = ImageDraw.Draw(im)
    bx, by = Wc * ss * 0.5, Hc * ss - 1
    L = 640 * scale * ss
    rng = np.random.default_rng(seed)

    def stem(x, y, length, a0, bend, w0, w1, n=60):
        pts = []
        ang = a0
        for i in range(n + 1):
            s = i / n
            pts.append((x, y))
            ang = a0 + bend * s ** 1.6
            x += math.sin(ang) * length / n
            y -= math.cos(ang) * length / n
        for i in range(n):
            w_ = w0 + (w1 - w0) * i / n
            d.line([pts[i], pts[i + 1]], fill=255, width=max(1, int(w_)))
            d.ellipse([pts[i][0] - w_ / 2, pts[i][1] - w_ / 2, pts[i][0] + w_ / 2, pts[i][1] + w_ / 2], fill=255)
        return pts[-1], ang

    def petal(cx, cy, rx, ry, rot):
        pts = []
        for k in range(48):
            a = 2 * math.pi * k / 48
            px, py = rx * math.cos(a), ry * math.sin(a)
            # flatten the top edge a little: poppy petals are wavy, cup-like
            py -= 0.08 * ry * math.sin(3 * a) ** 2
            pts.append((cx + px * math.cos(rot) - py * math.sin(rot),
                        cy + px * math.sin(rot) + py * math.cos(rot)))
        d.polygon(pts, fill=255)

    # side buds (thin stems)
    for k, (a0, bend, ln) in enumerate([(-0.25, -0.35, 0.55), (0.3, 0.5, 0.42)]):
        (hx, hy), ang = stem(bx + (k * 2 - 1) * 6 * ss, by, L * ln, a0 + sway * 0.7, bend + sway * 0.5,
                             4.0 * scale * ss, 2.2 * scale * ss)
        r = 13 * scale * ss
        petal(hx + math.sin(ang) * r * 0.5, hy - math.cos(ang) * r * 0.5, r * 0.7, r * 1.05, ang)
    # leaves
    for k, (h_, side) in enumerate([(0.18, -1), (0.3, 1)]):
        ly = by - L * h_
        pts = []
        for i in range(30):
            s = i / 29
            pts.append((bx + side * (s * 70 * scale * ss), ly - s * 60 * scale * ss + math.sin(s * math.pi) * 12 * scale * ss))
        for i in range(29, -1, -1):
            s = i / 29
            pts.append((bx + side * (s * 70 * scale * ss), ly - s * 60 * scale * ss - math.sin(s * math.pi) * 10 * scale * ss))
        d.polygon(pts, fill=255)
    # main stem + head
    (hx, hy), ang = stem(bx, by, L, 0.05 + sway, 0.55 + droop + sway * 0.8, 7.5 * scale * ss, 4.5 * scale * ss)
    r = 46 * scale * ss
    ux, uy = math.sin(ang), -math.cos(ang)
    cx, cy = hx + ux * r * 0.55, hy + uy * r * 0.55
    petal(hx + ux * 6 * ss, hy + uy * 6 * ss, r * 0.28, r * 0.35, ang)     # receptacle
    petal(cx - uy * r * 0.33, cy + ux * r * 0.33, r * 0.55, r * 0.85, ang - 0.45)
    petal(cx + uy * r * 0.33, cy - ux * r * 0.33, r * 0.55, r * 0.85, ang + 0.45)
    petal(cx + ux * r * 0.12, cy + uy * r * 0.12, r * 0.5, r * 0.9, ang)
    petal(cx - uy * r * 0.62 + ux * r * 0.05, cy + ux * r * 0.62 + uy * r * 0.05, r * 0.3, r * 0.55, ang - 0.95)
    im = im.resize((Wc, Hc), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255, Wc // 2, Hc - 1


def init_assets(feat):
    if A:
        return
    A["feat"] = feat
    fogA = fbm(CH // 2, W, 3.0, 11)
    fogB = fbm(CH // 2, W, 3.3, 12)
    A["fogA"] = np.tile(resizeF(fogA, W, CH), (1, 2))
    A["fogB"] = np.tile(resizeF(fogB, W, CH), (1, 2))
    cl = fbm(CH // 2, W // 2, 2.6, 13)
    A["cloud"] = np.tile(resizeF(cl, W, CH), (1, 2))
    rng = np.random.default_rng(5)
    ns = 520
    A["stars"] = dict(x=rng.integers(0, W, ns), y=(rng.random(ns) ** 1.4 * (HZ - 30)).astype(int),
                      b=rng.random(ns) ** 3 * 0.9 + 0.05, ph=rng.random(ns) * 6.28,
                      sp=rng.random(ns) * 2 + 0.5)
    A["ridge_far"] = 26 + 14 * fbm1d(W, 2.2, 21)
    A["ridge_near"] = 12 + 8 * fbm1d(W, 2.0, 22)
    A["moon"] = moon_sprite()
    A["glint"] = np.random.default_rng(7).random((CH - HZ, W * 2)).astype(np.float32)
    A["grain"] = [np.random.default_rng(100 + i).standard_normal((H, W)).astype(np.float16)
                  for i in range(12)]
    yy, xx = np.mgrid[:CH, :W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - CH / 2) / (CH * 0.78)) ** 2)
    A["vig"] = np.clip(1 - 0.55 * r ** 2.2, 0.25, 1).astype(np.float32)
    A["xx"] = np.arange(W, dtype=np.float32)
    d = np.arange(CH - HZ, dtype=np.float32)
    A["refl_d"] = d
    A["refl_src"] = (HZ - 1 - d).astype(np.int32)
    A["bokeh"] = [bokeh_sprite(r) for r in range(2, 30)]
    A["dot"] = [gauss_sprite(r, r / 2.2) for r in range(1, 12)]
    # film-strip tiles (small stills in the language of the cover)
    A["tiles"] = make_tiles()
    # typography
    A["title"] = text_mask("REFLECTION", "InstrumentSerif-Regular.ttf", 128, tracking=26)
    A["title2"] = text_mask("eter", "InstrumentSerif-Italic.ttf", 92, tracking=10)
    A["artist"] = text_mask("wanderingnonet286", "DMMono-Regular.ttf", 22, tracking=8)
    A["small_title"] = text_mask("REFLECTION_ETER", "DMMono-Regular.ttf", 15, tracking=6)
    A["chapters"] = {s[0]: text_mask(s[1], "DMMono-Regular.ttf", 17, tracking=5) for s in SECTIONS}
    A["end1"] = text_mask("Reflection_Eter", "InstrumentSerif-Italic.ttf", 72, tracking=4)
    A["end2"] = text_mask("music  wanderingnonet286", "DMMono-Regular.ttf", 20, tracking=6)
    A["end3"] = text_mask("visuals  generated from the sound", "DMMono-Regular.ttf", 16, tracking=5)


def moon_sprite():
    r = 120
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y)
    disc = np.clip(26 - d + 0.5, 0, 1)
    tex = 1 - 0.18 * np.clip(fbm(2 * r + 1, 2 * r + 1, 2.4, 31), -2, 2) * 0.5
    glow = np.exp(-d / 26) * 0.55 + np.exp(-d / 70) * 0.18
    glow *= np.clip(1 - (d / r) ** 2, 0, 1) ** 2
    return (disc * 0.95 * tex + glow * (1 - disc)).astype(np.float32)


def make_tiles():
    tw, th = 440, 330
    tiles = []
    yy = np.linspace(0, 1, th)[:, None]
    # 1. flower against pale sky
    t1 = 0.62 + 0.18 * yy + 0.07 * resizeF(fbm(64, 96, 2.8, 41), tw, th)
    m, ax, ay = draw_flower(0.4, -0.05, droop=0.35)
    t1 = paste_mask(t1, m, tw * 0.55, th + 40, 0.05)
    tiles.append(t1)
    # 2. raindrops on dark foliage
    t2 = 0.13 + 0.08 * resizeF(fbm(60, 80, 2.2, 42), tw, th)
    rng = np.random.default_rng(43)
    for _ in range(26):
        blit_add(t2, bokeh_sprite(int(rng.integers(3, 9))), rng.random() * tw, rng.random() * th, 0.7)
    tiles.append(t2)
    # 3. clouds
    t3 = np.clip(0.45 + 0.22 * resizeF(fbm(60, 80, 3.0, 44), tw, th), 0, 1)
    tiles.append(t3)
    # 4. moon over mirror water
    t4 = np.full((th, tw), 0.16, np.float32) + 0.08 * (1 - np.abs(yy - 0.5) * 2)
    t4[th // 2:th // 2 + 1] += 0.6
    blit_add(t4, gauss_sprite(10, 4) * 1.2, tw * 0.66, th * 0.28, 0.9)
    blit_add(t4, gauss_sprite(10, 4) * 0.5, tw * 0.66, th * 0.72, 0.8)
    tiles.append(t4)
    # 5. single falling drop
    t5 = 0.1 + 0.3 * np.exp(-((np.linspace(-1, 1, tw)[None, :]) ** 2) / 0.1) * np.exp(-((yy - 0.8) ** 2) / 0.02)
    blit_add(t5, bokeh_sprite(9), tw * 0.5, th * 0.4, 1.0)
    tiles.append(t5.astype(np.float32))
    # 6. mist + light line
    t6 = 0.2 + 0.12 * resizeF(fbm(60, 80, 3.1, 45), tw, th)
    t6 += 0.7 * np.exp(-((yy - 0.5) ** 2) / 0.0004) * np.exp(-(np.linspace(-1, 1, tw)[None, :] ** 2) / 0.5)
    tiles.append(t6.astype(np.float32))
    return [np.clip(t, 0, 1.2).astype(np.float32) for t in tiles]


def paste_mask(dst, m, cx_base, cy_base, val):
    h, w = m.shape
    x0 = int(cx_base - w / 2)
    y0 = int(cy_base - h)
    out = dst.copy()
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(dst.shape[0], y0 + h), min(dst.shape[1], x0 + w)
    if ys1 > ys0 and xs1 > xs0:
        mm = m[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        out[ys0:ys1, xs0:xs1] = out[ys0:ys1, xs0:xs1] * (1 - mm) + val * mm
    return out


# ============================================================================
# Scene building blocks (luminance, CH x W)
# ============================================================================
def fog_layer(t, speed=10, which="fogA"):
    off = int(t * speed) % W
    return A[which][:, off:off + W]


def sky_base(t, top=0.06, bottom=0.34):
    y = np.linspace(0, 1, HZ, dtype=np.float32)[:, None]
    g = top + (bottom - top) * y ** 2.2
    fog = fog_layer(t, 9)[:HZ] * 0.045 + fog_layer(t + 40, 16, "fogB")[:HZ] * 0.03 * y
    return np.broadcast_to(g, (HZ, W)) + fog


def add_stars(sky, t, amount):
    if amount <= 0:
        return
    s = A["stars"]
    tw = 0.65 + 0.35 * np.sin(t * s["sp"] + s["ph"])
    b = s["b"] * tw * amount
    np.add.at(sky, (s["y"], s["x"]), b)
    big = s["b"] > 0.5
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        np.add.at(sky, (np.clip(s["y"][big] + dy, 0, HZ - 1), np.clip(s["x"][big] + dx, 0, W - 1)),
                  b[big] * 0.3)


def add_ridges(sky, t, far_l=0.14, near_l=0.06):
    y = np.arange(HZ, dtype=np.float32)[:, None]
    rf = np.roll(A["ridge_far"], int(t * 2))[None, :]
    rn = np.roll(A["ridge_near"], int(t * 4))[None, :]
    mf = np.clip((y - (HZ - rf)) * 0.9, 0, 1)
    mn = np.clip((y - (HZ - rn)) * 0.9, 0, 1)
    sky[:] = sky * (1 - mf) + far_l * mf
    sky[:] = sky * (1 - mn) + near_l * mn


def reflect(sky, t, calm=1.0, gain=0.62):
    """Mirror the sky into the water with animated ripples."""
    d = A["refl_d"][:, None]
    x = A["xx"][None, :]
    amp = (0.5 + d * 0.03) * calm
    dx = amp * (0.6 * np.sin(x * 0.011 + d * 0.33 - t * 1.25) + 0.4 * np.sin(x * 0.0037 - d * 0.11 + t * 0.6))
    dy = (d * 0.012 * calm) * np.sin(d * 0.45 + t * 1.9 + x * 0.002)
    src_y = np.clip(A["refl_src"][:, None] + dy.astype(np.int32), 0, HZ - 1)
    src_x = (x + dx).astype(np.int32) % W
    r = sky[src_y, src_x]
    r = (r + np.roll(r, 1, 0) + np.roll(r, 2, 0)) / 3
    fade = gain * (1 - d / (CH - HZ) * 0.35)
    return r * fade


def moon_glint(water, t, mx, strength):
    d = A["refl_d"][:, None]
    width = 18 + d * 0.22
    band = np.exp(-((A["xx"][None, :] - mx) ** 2) / (2 * width ** 2))
    off = int(t * 37) % W
    g = A["glint"][:, off:off + W]
    sp = np.clip((g - 0.93) * 14, 0, 1) * band * strength * (1 - d / (CH - HZ)) ** 0.7
    water += sp + band * 0.05 * strength


class Rings:
    """Ripple rings on the mirror plane."""

    @staticmethod
    def draw(water, t, events, gain=0.3):
        for (te, cx, cy, s) in events:
            a = t - te
            if a < 0 or a > 3.2:
                continue
            R = 12 + 230 * math.sqrt(a / 3.2) * s
            depth = (cy - HZ) / (CH - HZ)
            ecc = 0.07 + 0.3 * depth
            wdt = 2.0 + 5 * a
            fade = (1 - a / 3.2) ** 1.6 * gain
            x0, x1 = int(max(0, cx - R - 20)), int(min(W, cx + R + 20))
            yl0 = int(max(HZ, cy - R * ecc - 12)); yl1 = int(min(CH, cy + R * ecc + 12))
            if x1 <= x0 or yl1 <= yl0:
                continue
            yy = np.arange(yl0, yl1, dtype=np.float32)[:, None]
            xx = np.arange(x0, x1, dtype=np.float32)[None, :]
            dist = np.sqrt((xx - cx) ** 2 + ((yy - cy) / ecc) ** 2)
            v = np.exp(-((dist - R) / wdt) ** 2) + 0.5 * np.exp(-((dist - R * 0.62) / (wdt * 0.8)) ** 2)
            water[yl0 - HZ:yl1 - HZ, x0:x1] += v * fade


def ring_events(t, t_from, t_to, every=1, seed=0, strength=1.0):
    ev = []
    if t_to <= t_from:
        return ev
    b0 = max(0, int((max(t_from, t - 3.3) - T0) / BEAT))
    b1 = int((min(t, t_to) - T0) / BEAT) + 1
    for b in range(b0, b1):
        tb = T0 + b * BEAT
        if tb < t_from or tb > t_to or tb > t or b % every:
            continue
        cx = W * (0.08 + 0.84 * hash01(b, seed, 1))
        cy = HZ + 24 + (CH - HZ - 60) * hash01(b, seed, 2) ** 1.3
        s = (0.6 + 0.6 * hash01(b, seed, 3)) * strength * (1.25 if b % 4 == 0 else 1.0)
        ev.append((tb, cx, cy, s))
    return ev


def rain(img, t, n, seed, speed=(40, 160), size=(3, 22), gain=0.28, y_min=0, y_max=CH):
    rng = np.random.default_rng(seed)
    xs = rng.random(n) * W
    y0 = rng.random(n) * (y_max - y_min)
    sp = rng.uniform(*speed, n)
    sz = rng.uniform(0, 1, n) ** 2.2
    for i in range(n):
        r = int(size[0] + (size[1] - size[0]) * sz[i])
        y = y_min + (y0[i] + sp[i] * (0.4 + 0.6 * sz[i]) * t) % (y_max - y_min + 60) - 30
        x = xs[i] + 14 * math.sin(t * 0.3 + i)
        spr = A["bokeh"][min(r, len(A["bokeh"]) + 1) - 2]
        blit_add(img, spr, x, y, gain * (0.35 + 0.65 * (1 - sz[i]) if r > 12 else gain))


def light_line(img, t, strength, spec=None, width=1.0, y=HZ, spread=0.55):
    """The signature horizontal light line; its halo breathes with the spectrum."""
    if strength <= 0:
        return
    x = (A["xx"] - W / 2) / (W / 2)
    base = np.exp(-(x ** 2) / (2 * spread ** 2)) * 0.85 + 0.15 * np.exp(-(x ** 2) / 1.8)
    if spec is not None:
        idx = np.clip((np.abs(x) ** 0.8) * 55, 0, 55)
        s = np.interp(idx, np.arange(56), spec)
        halo = 5 + 26 * s * width
    else:
        halo = np.full(W, 6.0) * width
        s = np.zeros(W)
    r = 70
    yy = np.arange(-r, r + 1, dtype=np.float32)[:, None]
    core = np.exp(-(yy ** 2) / (2 * (0.8 * width + 0.3) ** 2)) * 1.25
    hal = np.exp(-np.abs(yy) / halo[None, :]) * (0.2 + 0.35 * s[None, :])
    v = (core + hal) * base[None, :] * strength
    y0, y1 = max(0, y - r), min(img.shape[0], y + r + 1)
    img[y0:y1] += v[y0 - (y - r):y1 - (y - r)]


def ether(img, t, n, seed, rate, gain):
    """Particles rising from the horizon into the sky (the 'ether')."""
    rng = np.random.default_rng(seed)
    xs = rng.random(n) * W
    ph = rng.random(n)
    sp = rng.uniform(0.6, 1.4, n)
    sz = rng.integers(1, 5, n)
    for i in range(n):
        life = (ph[i] + t * rate * sp[i] / (HZ)) % 1.0
        y = HZ - 4 - life * (HZ + 20)
        x = xs[i] + 22 * math.sin(t * 0.6 * sp[i] + i * 1.7) * life
        a = math.sin(life * math.pi) ** 1.5 * gain * (0.5 + 0.5 * math.sin(t * 3 + i))
        blit_add(img, A["dot"][sz[i]], x, y, a)


# ============================================================================
# Scenes
# ============================================================================
def scene_lake(t, f, i, variant="lake"):
    sky = np.array(sky_base(t), dtype=np.float32)
    add_stars(sky, t, 0.9)
    mx, my = W * 0.7 - t * 0.9, HZ * 0.34 + t * 0.12
    blit_add(sky, A["moon"], mx, my, 1.0)
    fin = variant == "finale"
    if fin:
        aurora(sky, t, f, i)
        ether(sky, t, 140, 77, 26 + 30 * f["level"][i], 0.55 + 0.5 * f["pulse"][i])
    cloud = fog_layer(t, 6, "cloud")[:HZ]
    sky += np.clip(cloud - 0.6, 0, 3) * 0.05
    add_ridges(sky, t)
    water = reflect(sky, t, calm=1.0 + 0.25 * f["pulse"][i])
    moon_glint(water, t, mx, 0.6)
    t0, t1 = (SECTIONS[1][2], SILENCE[0]) if not fin else (SECTIONS[5][2], SECTIONS[5][3])
    Rings.draw(water, t, ring_events(t, t0, t1, every=1 if fin else 2, seed=1 if fin else 0,
                                     strength=1.1 if fin else 1.0), gain=0.34 if fin else 0.28)
    img = np.concatenate([sky, water], 0)
    if fin:
        fl, ax, ay = flower_cached(i, 1.15, 0.03 * math.sin(t * 0.55) + 0.012 * f["pulse"][i], droop=0.25)
        img = paste_mask(img, fl, W * 0.13, CH + 90, 0.015)
    return img


def aurora(sky, t, f, i):
    x = A["xx"]
    n = A["ridge_far"]  # reuse a periodic 1-D noise
    k1 = np.roll(n, int(t * 25)) * 0.5 + np.roll(A["ridge_near"], -int(t * 14)) * 0.5
    curtain = np.clip(0.5 + 0.35 * np.sin(x * 0.006 + t * 0.2) + 0.03 * (k1 - 20), 0, 1) ** 2
    base_y = HZ * 0.62 + 40 * np.sin(x * 0.0021 + t * 0.15) + 18 * np.sin(x * 0.0071 - t * 0.3)
    y = np.arange(HZ, dtype=np.float32)[:, None]
    dy = base_y[None, :] - y
    prof = np.where(dy > 0, np.exp(-dy / 150), np.exp(dy / 9))
    streak = 0.75 + 0.25 * np.sin(x * 0.09 + np.sin(x * 0.013 + t * 0.5) * 4)[None, :]
    amt = (0.16 + 0.12 * f["level"][i] + 0.06 * f["pulse"][i])
    sky += prof * curtain[None, :] * streak * amt


_FL_CACHE = {}


def flower_cached(i, scale, sway, droop=0.0):
    key = (scale, round(sway, 3), round(droop, 3))
    if key not in _FL_CACHE:
        if len(_FL_CACHE) > 64:
            _FL_CACHE.clear()
        _FL_CACHE[key] = draw_flower(scale, sway, droop=droop)
    return _FL_CACHE[key]


def scene_flower(t, f, i):
    # pale misty sky like the cover's top frame, flower silhouette in front
    y = np.linspace(0, 1, CH, dtype=np.float32)[:, None]
    g = 0.36 + 0.12 * y + 0.1 * np.exp(-((y - 0.5) ** 2) / 0.02)
    img = np.broadcast_to(g, (CH, W)).astype(np.float32)
    img = img + (fog_layer(t, 14, "cloud") - 0.0) * 0.06 + fog_layer(t, 30, "fogB") * 0.035
    ts = t - SECTIONS[2][2]
    # slow push: scale grows 1.0 -> 1.12 over the section
    sc = 1.0 + 0.06 * ts / (SECTIONS[2][3] - SECTIONS[2][2])
    sway = 0.045 * math.sin(2 * math.pi * t / (4 * BAR)) + 0.02 * math.sin(t * 1.3) + 0.018 * f["pulse"][i]
    fl, ax, ay = flower_cached(i, round(sc, 2), sway, droop=0.3)
    # distant soft flowers (depth)
    bg, _, _ = flower_cached(i, 0.55, -sway * 0.7, droop=0.1)
    img = paste_mask(img, bg * 0.35, W * 0.2, CH + 40, 0.3)
    img = paste_mask(img, bg * 0.3, W * 0.82, CH + 80, 0.3)
    img = paste_mask(img, fl, W * 0.56, CH + 150, 0.03)
    # water surface at the bottom reflects the light line
    lower = np.clip((y - 0.86) * 12, 0, 1)
    img = img * (1 - 0.35 * lower)
    rain(img, t, 42, 9, speed=(25, 70), size=(4, 26), gain=0.12 + 0.05 * f["pulse"][i])
    rain(img, t, 90, 10, speed=(120, 240), size=(2, 5), gain=0.2)
    return img


def scene_film(t, f, i):
    ts = t - SECTIONS[3][2]
    tiles = A["tiles"]
    th, tw = tiles[0].shape
    pitch = th + 34
    SW = tw + 110
    strip_h = CH + 2 * pitch
    off = (ts * pitch / (2 * BAR) + 8 * f["pulse"][i])
    base = int(off) % pitch
    k0 = int(off) // pitch
    strip = np.full((strip_h, SW), 0.025, np.float32)
    for j in range(strip_h // pitch + 1):
        tile = tiles[(k0 + j) % len(tiles)]
        y0 = j * pitch + 17
        if y0 + th <= strip_h:
            kb = int(6 * math.sin(t * 0.3 + j))
            strip[y0:y0 + th, 55:55 + tw] = np.roll(tile, kb, 1) * (1.05 + 0.08 * math.sin(t * 2 + j))
    # sprocket holes
    for yh in range(10, strip_h, 38):
        for xh in (18, SW - 32):
            strip[yh:yh + 16, xh:xh + 14] = 0.55
    strip = strip[base:base + CH]
    # background: blurred magnified strip + mist (like the cover)
    small = strip[::6, ::6]
    small = ndimage.gaussian_filter(small, 2.5)
    bgw = int(SW * 3.4)
    bg = resizeF(small, bgw, CH)
    img = np.full((CH, W), 0.05, np.float32) + fog_layer(t, 12, "fogA") * 0.05 + fog_layer(t, 7, "cloud") * 0.04
    x0 = (W - bgw) // 2
    xs0, xs1 = max(0, x0), min(W, x0 + bgw)
    img[:, xs0:xs1] += bg[:, xs0 - x0:xs1 - x0] * 0.45
    img += 0.13 * np.exp(-((A["xx"][None, :] - W / 2) / (W * 0.3)) ** 2)
    sx = (W - SW) // 2
    shadow = np.zeros_like(img)
    shadow[:, sx - 20:sx + SW + 20] = 1
    shadow = ndimage.uniform_filter1d(shadow, 41, 1)
    img = img * (1 - 0.5 * shadow)
    img[:, sx:sx + SW] = strip
    rain(img, t, 26, 31, speed=(20, 60), size=(6, 28), gain=0.08)
    return img


def scene_void(t, f, i):
    img = np.full((CH, W), 0.018, np.float32)
    fog = fog_layer(t, 5, "fogA") * 0.035 + fog_layer(t, 9, "fogB") * 0.02
    img += fog
    sky = img[:HZ]
    ts = t - SECTIONS[4][2]
    add_stars(sky, t, 0.2 + 0.5 * sstep(0, 12, ts))
    # a single droplet falls every 2 bars, meeting its reflection on the downbeat
    period = 2 * BAR
    k = math.floor((t - T0) / period)
    for kk in (k, k - 1):
        t_hit = T0 + (kk + 1) * period
        tf = t - (t_hit - period * 0.8)
        cx = W * (0.3 + 0.4 * hash01(kk, 55))
        if 0 <= tf < period * 0.8:
            p = tf / (period * 0.8)
            yd = HZ - (1 - p ** 2.2) * (HZ + 30)
            for m in (1, -1):
                yy = yd if m == 1 else 2 * HZ - yd
                blit_add(img, A["bokeh"][6], cx, yy, 0.8 * (1 if m == 1 else 0.5))
                for s in range(1, 7):
                    blit_add(img, A["dot"][2], cx, yy - m * s * 7 * (0.3 + p), 0.12 * (1 - s / 7))
        # impact ring on the mirror
        a = t - t_hit
        if 0 <= a < 4.5:
            R = 8 + 360 * math.sqrt(a / 4.5)
            yy = np.arange(HZ - 26, HZ + 27, dtype=np.float32)[:, None]
            xx = A["xx"][None, :]
            dist = np.sqrt((xx - cx) ** 2 + ((yy - HZ) / 0.06) ** 2)
            v = np.exp(-((dist - R) / (3 + 5 * a)) ** 2) * (1 - a / 4.5) ** 1.5 * 0.5
            img[HZ - 26:HZ + 27] += v
    water_dim = np.linspace(1, 0.7, CH - HZ, dtype=np.float32)[:, None]
    img[HZ:] = img[HZ:] * water_dim
    # build-up: rising particles accelerate
    b = sstep(BUILD[0], BUILD[1], t)
    if b > 0:
        ether(img, t, int(40 + 120 * b), 91, 40 + 260 * b ** 2, 0.8 * b)
    return img


def scene_intro(t, f, i):
    img = np.full((CH, W), 0.02, np.float32)
    fog = fog_layer(t, 11, "fogA") * 0.06 + fog_layer(t + 17, 19, "fogB") * 0.04 + fog_layer(t, 5, "cloud") * 0.05
    img += fog * sstep(0, 6, t) + 0.05 * sstep(0, 8, t)
    rain(img, t, 30, 3, speed=(15, 40), size=(5, 26), gain=0.1 * sstep(2, 10, t))
    return img


def scene_outro(t, f, i):
    return scene_lake(t, f, i)


# ============================================================================
# Frame composition
# ============================================================================
FADE_IN = dict(intro=0.0, lake=2.2, flower=0.0, film=1.2, void=1.6, finale=0.0, outro=2.0)  # 0 = hard cut


def section_weights(t):
    """Crossfade weights between scenes; the drop at bar 24 and the finale are hard cuts."""
    w = {}
    for idx, (name, _, a, b) in enumerate(SECTIONS):
        fi = FADE_IN[name]
        rin = (1.0 if t >= a else 0.0) if fi == 0 else sstep(a - fi / 2, a + fi / 2, t)
        if idx + 1 < len(SECTIONS):
            fo = FADE_IN[SECTIONS[idx + 1][0]]
            rout = (1.0 if t < b else 0.0) if fo == 0 else 1 - sstep(b - fo / 2, b + fo / 2, t)
        else:
            rout = 1.0
        if rin * rout > 1e-3:
            w[name] = rin * rout
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


SCENES = dict(intro=scene_intro, lake=scene_lake, flower=scene_flower, film=scene_film,
              void=scene_void, finale=lambda t, f, i: scene_lake(t, f, i, "finale"), outro=scene_outro)


def exposure(t, f, i):
    e = 1.0 + 0.05 * f["pulse"][i] + 0.04 * (f["level"][i] - 0.5)
    if SILENCE[0] - 0.3 < t < SILENCE[1]:            # hold the breath
        e *= 1 - 0.55 * sstep(SILENCE[0] - 0.3, SILENCE[0] + 0.9, t)
    if SECTIONS[-1][2] < t:                          # outro fade
        e *= 1 - sstep(209.5, 213.0, t)
    e *= sstep(0.0, 3.0, t) * 0.9 + 0.1 if t < 3 else 1.0
    return e


def flash(t):
    v = 0.0
    for tc, amt in ((SILENCE[1], 0.18), (SECTIONS[5][2], 0.32)):
        if t >= tc:
            v = max(v, amt * math.exp(-(t - tc) / 0.35))
    b = sstep(BUILD[1] - 2.2, BUILD[1], t) if t < BUILD[1] else 0.0
    return max(v, 0.22 * b ** 2)


def line_strength(t, f, i):
    s = 0.55 + 0.35 * f["pulse"][i]
    if t < SECTIONS[1][2]:
        s = 0.9 * sstep(2.0, 9.0, t)
    if SILENCE[0] < t < SILENCE[1]:
        s *= 0.35
    if SECTIONS[4][2] < t < SECTIONS[4][3]:
        s = 0.35 + 0.6 * sstep(BUILD[0], BUILD[1], t) ** 2
    if t > SECTIONS[5][2]:
        s *= 1.25
    if t > SECTIONS[6][2]:
        s *= 1 - sstep(206, 212, t)
    return s


def line_spread(t):
    if t < SECTIONS[1][2]:
        return 0.04 + 0.55 * sstep(2.0, 12.0, t)
    if t > SECTIONS[6][2]:
        return 0.55 - 0.5 * sstep(205, 211.5, t)
    return 0.55


def render_frame(i):
    f = A["feat"]
    t = i / FPS
    ws = section_weights(t)
    lum = None
    for name, w in ws.items():
        s = SCENES[name](t, f, i)
        lum = s * w if lum is None else lum + s * w
    lum = np.ascontiguousarray(lum, dtype=np.float32)
    ln = line_strength(t, f, i)
    light_line(lum, t, ln, spec=f["spec"][i], width=1.0 + 0.5 * f["pulse"][i], spread=line_spread(t))
    lum *= exposure(t, f, i)
    lum += flash(t)
    # bloom
    sm = lum.reshape(CH // 4, 4, W // 8, 8).mean((1, 3))
    br = np.clip(sm - 0.55, 0, None)
    bl = ndimage.gaussian_filter(br, 6) * 0.9 + ndimage.gaussian_filter(br, 22) * 0.9
    lum += resizeF(bl, W, CH)
    lum *= A["vig"]
    # grade
    full = np.zeros((H, W), np.float32)
    full[BAR_Y:BAR_Y + CH] = lum
    g = A["grain"][i % len(A["grain"])].astype(np.float32)
    full[BAR_Y:BAR_Y + CH] += g[BAR_Y:BAR_Y + CH] * (0.022 + 0.03 * np.clip(lum, 0, 1) * (1 - np.clip(lum, 0, 1)))
    dust(full, i)
    idx = np.clip(np.nan_to_num(full) * (LUT_N - 1), 0, LUT_N - 1).astype(np.int32)
    rgb = LUT[idx]
    # chromatic aberration on kicks (stronger in the finale)
    ca = int(round((1.0 + (2.5 if SECTIONS[5][2] < t < SECTIONS[5][3] else 0.8) * f["pulse"][i])))
    if ca:
        rgb[:, :, 0] = np.roll(rgb[:, :, 0], ca, 1)
        rgb[:, :, 2] = np.roll(rgb[:, :, 2], -ca, 1)
    rgb[:BAR_Y] = 0
    rgb[BAR_Y + CH:] = 0
    typography(rgb, t)
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def dust(full, i):
    rng = np.random.default_rng(9000 + i)
    for _ in range(rng.integers(0, 3)):
        x, y = rng.integers(40, W - 40), rng.integers(BAR_Y + 20, BAR_Y + CH - 20)
        r = int(rng.integers(1, 4))
        full[y - r:y + r + 1, x - r:x + r + 1] += gauss_sprite(r, r * 0.6) * float(rng.uniform(0.08, 0.2))
    if rng.random() < 0.06:
        x = int(rng.integers(100, W - 100))
        full[BAR_Y:BAR_Y + CH, x] += 0.05


def put_text(rgb, m, cx, cy, alpha, color=(226, 238, 236), anchor="c"):
    if alpha <= 0.003:
        return
    h, w = m.shape
    if anchor == "c":
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
    elif anchor == "l":
        x0, y0 = int(cx), int(cy - h / 2)
    else:
        x0, y0 = int(cx - w), int(cy - h / 2)
    reg = rgb[y0:y0 + h, x0:x0 + w]
    a = (m[:reg.shape[0], :reg.shape[1]] * alpha)[..., None]
    reg[:] = reg * (1 - a) + np.array(color, np.float32) * a


def typography(rgb, t):
    # opening title
    a = sstep(5.0, 8.5, t) * (1 - sstep(16.5, 20.0, t))
    if a > 0:
        rise = (1 - sstep(5.0, 10.0, t)) * 10
        put_text(rgb, A["title"], W / 2, H / 2 - 70 + rise, a)
        put_text(rgb, A["title2"], W / 2, H / 2 + 60 + rise, a * sstep(6.5, 9.5, t))
        put_text(rgb, A["artist"], W / 2, H / 2 + 150, a * sstep(8.5, 11.0, t) * 0.8)
    # chapter labels in the lower letterbox bar
    for name, _, s, e in SECTIONS[1:]:
        ca = sstep(s + 0.6, s + 2.0, t) * (1 - sstep(s + 7.0, s + 9.0, t))
        if ca > 0:
            put_text(rgb, A["chapters"][name], 96, H - BAR_Y / 2, ca * 0.75, anchor="l")
    # persistent small title in the top bar after the intro
    if 21 < t < 206:
        put_text(rgb, A["small_title"], W - 96, BAR_Y / 2, 0.35 * sstep(22, 25, t) * (1 - sstep(203.5, 205.5, t)),
                 anchor="r")
    # end card
    e = sstep(205.0, 207.0, t) * (1 - sstep(211.0, 213.0, t))
    if e > 0:
        put_text(rgb, A["end1"], W / 2, H / 2 - 30, e)
        put_text(rgb, A["end2"], W / 2, H / 2 + 50, e * 0.8)
        put_text(rgb, A["end3"], W / 2, H / 2 + 88, e * 0.5)


# ============================================================================
def _init(feat):
    init_assets(feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--stills", default=None, help="comma separated seconds -> PNGs in OUT dir")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--video-only", action="store_true")
    args = ap.parse_args()

    ff = ffmpeg_exe()
    dur = float(subprocess.run([ff, "-i", args.audio], capture_output=True, text=True).stderr
                .split("Duration: ")[1].split(",")[0].split(":")[-1]) + \
        60 * int(subprocess.run([ff, "-i", args.audio], capture_output=True, text=True).stderr
                 .split("Duration: ")[1].split(",")[0].split(":")[1])
    n_frames = int(math.ceil(dur * FPS))
    print(f"duration {dur:.2f}s -> {n_frames} frames", flush=True)
    feat = analyse(args.audio, n_frames)
    print(f"kicks detected: {len(feat['kicks'])}", flush=True)

    if args.stills:
        os.makedirs(args.out, exist_ok=True)
        init_assets(feat)
        for s in args.stills.split(","):
            i = int(float(s) * FPS)
            Image.fromarray(render_frame(i)).save(os.path.join(args.out, f"still_{float(s):07.2f}.png"))
            print("still", s, flush=True)
        return

    i0 = int(args.start * FPS)
    i1 = n_frames if args.end is None else min(n_frames, int(args.end * FPS))
    vcodec = ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-tune", "grain", "-pix_fmt", "yuv420p"]
    if args.video_only:   # segment renders; joined + muxed with the audio afterwards
        cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-"] + vcodec + [args.out]
    else:
        cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-ss", str(i0 / FPS), "-t", str((i1 - i0) / FPS), "-i", args.audio,
               "-map", "0:v", "-map", "1:a"] + vcodec + ["-c:a", "aac", "-b:a", "320k",
                                                         "-movflags", "+faststart", "-shortest", args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    import time
    t_start = time.time()
    with Pool(args.workers, initializer=_init, initargs=(feat,)) as pool:
        for k, fr in enumerate(pool.imap(render_frame, range(i0, i1), chunksize=4)):
            enc.stdin.write(fr.tobytes())
            if k % 120 == 0:
                el = time.time() - t_start
                print(f"frame {i0 + k}/{i1}  {el:.0f}s elapsed  eta {el / (k + 1) * (i1 - i0 - k - 1):.0f}s",
                      flush=True)
    enc.stdin.close()
    enc.wait()
    print("done", args.out)


if __name__ == "__main__":
    main()

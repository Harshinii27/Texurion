"""Generate the Texurion pixel-T brand assets.

The mark is a bold capital "T" built from a deliberate grid of small square
pixels in layered shades of pink. This script renders clean SVG (vector) and
transparent PNG versions into this directory. It is deterministic: running it
again reproduces the exact same assets.

Usage:
    python assets/branding/generate_brand_assets.py
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Palette: soft blush -> deep magenta --------------------------------
PALE = "#FFE3F0"       # soft blush highlight
BLUSH = "#FFC7DE"
PASTEL = "#FF9EC4"
ROSE = "#FF6BA6"
HOT = "#F72585"
FUCHSIA = "#E5177F"
MAGENTA = "#C2185B"
DEEP = "#A61E4D"

# Ordered light -> deep. Kept mostly saturated so the mark stays visible over
# the existing pink website gradient.
SHADES = [PALE, BLUSH, PASTEL, ROSE, HOT, FUCHSIA, MAGENTA, DEEP]
HIGHLIGHT = PALE

INK = "#1B2438"
TAGLINE_INK = "#5B6B8C"
X_ACCENT = "#A61E4D"

PITCH = 16
PIXEL = 14
PAD = 14
RADIUS = 1.5


def pixel_color(col: int, row: int, rows: int) -> str:
    """Deterministic shade with a soft top->bottom depth gradient."""

    if (col * 3 + row * 5) % 13 == 0:
        return HIGHLIGHT
    t = row / max(rows - 1, 1)
    base = 2.0 + t * 4.0
    jitter = ((col * 7 + row * 11) % 3) - 1
    idx = int(round(base + jitter))
    idx = max(1, min(len(SHADES) - 1, idx))
    return SHADES[idx]


def t_cells(cols: int, rows: int, bar_h: int, stem_w: int):
    """Return the (col, row) cells that make the solid capital T."""

    centre = cols // 2
    half = stem_w // 2
    stem_cols = set(range(centre - half, centre - half + stem_w))
    cells = set()
    for r in range(bar_h):
        for c in range(cols):
            cells.add((c, r))
    for r in range(bar_h, rows):
        for c in stem_cols:
            cells.add((c, r))
    return cells, sorted(stem_cols)


def dispersion_specs(cols: int, rows: int, bar_h: int, stem_cols):
    """A small, deliberate set of outward-drifting pixels at chosen edges."""

    left, right = min(stem_cols), max(stem_cols)
    specs = []
    # Top bar left / right edges.
    for r, dx in ((1, -1), (2, -1), (0, -1)):
        specs.append((0, r, dx, 0, 0.70, "soft"))
    for r, dx in ((1, 1), (3, 1), (2, 1)):
        specs.append((cols - 1, r, dx, 0, 0.70, "soft"))
    # Bottom of the stem.
    specs.append((left, rows - 1, 0, 1, 0.62, "deep"))
    specs.append((right, rows - 1, 0, 1, 0.62, "deep"))
    # Outer corners of the bar.
    specs.append((0, bar_h - 1, -1, 1, 0.55, "soft"))
    specs.append((cols - 1, bar_h - 1, 1, 1, 0.55, "soft"))
    return specs


def _rect(x, y, size, fill, opacity=None, radius=RADIUS):
    opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
    r = f' rx="{radius}"' if radius else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" '
            f'height="{size:.1f}"{r} fill="{fill}"{opacity_attr}/>')


def build_mark(cols, rows, bar_h, stem_w, dispersion=True):
    """Return (svg_inner, width, height) for the pixel T mark."""

    cells, stem_cols = t_cells(cols, rows, bar_h, stem_w)
    width = PAD * 2 + cols * PITCH
    height = PAD * 2 + rows * PITCH

    parts = []
    for col, row in sorted(cells, key=lambda c: (c[1], c[0])):
        x = PAD + col * PITCH
        y = PAD + row * PITCH
        parts.append(_rect(x, y, PIXEL, pixel_color(col, row, rows)))

    if dispersion:
        for col, row, dx, dy, op, tone in dispersion_specs(
                cols, rows, bar_h, stem_cols):
            cx = PAD + col * PITCH + PIXEL / 2 + dx * PITCH
            cy = PAD + row * PITCH + PIXEL / 2 + dy * PITCH
            size = PIXEL * 0.62
            fill = DEEP if tone == "deep" else HOT
            parts.append(_rect(cx - size / 2, cy - size / 2, size, fill,
                               opacity=f"{op:.2f}"))
    return "".join(parts), width, height


def icon_svg():
    inner, w, h = build_mark(13, 16, 4, 3)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" '
            f'aria-label="Texurion pixel T icon">{inner}</svg>')


def favicon_svg():
    inner, w, h = build_mark(7, 8, 2, 3, dispersion=True)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" '
            f'aria-label="Texurion favicon">{inner}</svg>')


def _wordmark_parts(dark=False):
    base = "#FFFFFF" if dark else INK
    x_col = "#FF7FB8" if dark else X_ACCENT
    return [("TE", base), ("X", x_col), ("URION", base)]


def logo_svg(dark=False):
    inner, iw, ih = build_mark(13, 16, 4, 3)
    target_h = 220.0
    s = target_h / ih
    scaled_w = iw * s
    pad = 24.0
    gap = 34.0
    total_w = pad * 2 + scaled_w + gap + 430
    total_h = pad * 2 + target_h
    word = "TEXURION"
    tag = "TEXTURE INTELLIGENCE STUDIO"
    base = "#FFFFFF" if dark else INK
    tagline = "#F6D7E6" if dark else TAGLINE_INK
    x_col = "#FF7FB8" if dark else X_ACCENT
    tx = pad + scaled_w + gap
    ty = pad + target_h * 0.56
    ty2 = ty + 46
    word_markup = (
        f'<text x="{tx:.1f}" y="{ty:.1f}" font-family="Segoe UI, Inter, '
        f'Helvetica Neue, Arial, sans-serif" font-size="72" font-weight="800" '
        f'letter-spacing="7" fill="{base}">'
        f'TE<tspan fill="{x_col}">X</tspan>URION</text>'
    )
    tag_markup = (
        f'<text x="{tx + 2:.1f}" y="{ty2:.1f}" font-family="Segoe UI, Inter, '
        f'Helvetica Neue, Arial, sans-serif" font-size="21" font-weight="600" '
        f'letter-spacing="6" fill="{tagline}">{tag}</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0f} '
        f'{total_h:.0f}" width="{total_w:.0f}" height="{total_h:.0f}" '
        f'role="img" aria-label="Texurion - Texture Intelligence Studio">'
        f'<g transform="translate({pad:.1f},{pad:.1f}) scale({s:.4f})">'
        f'{inner}</g>{word_markup}{tag_markup}</svg>'
    )


def _png_mark(cols, rows, bar_h, stem_w, out_px, path):
    from PIL import Image, ImageDraw

    inner_w = PAD * 2 + cols * PITCH
    inner_h = PAD * 2 + rows * PITCH
    scale = out_px / max(inner_w, inner_h)
    W, H = int(round(inner_w * scale)), int(round(inner_h * scale))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cells, stem_cols = t_cells(cols, rows, bar_h, stem_w)

    def box(col, row, dx=0.0, dy=0.0, size=PIXEL):
        cx = (PAD + col * PITCH + dx * PITCH + PIXEL / 2) * scale
        cy = (PAD + row * PITCH + dy * PITCH + PIXEL / 2) * scale
        r = size / 2 * scale
        return [cx - r, cy - r, cx + r, cy + r]

    for col, row in cells:
        draw.rounded_rectangle(
            box(col, row), radius=max(1, int(RADIUS * scale)),
            fill=pixel_color(col, row, rows))
    for col, row, dx, dy, op, tone in dispersion_specs(cols, rows, bar_h, stem_cols):
        fill = DEEP if tone == "deep" else HOT
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(layer).rounded_rectangle(
            box(col, row, dx, dy, PIXEL * 0.62),
            radius=max(1, int(RADIUS * scale)), fill=fill)
        img = Image.alpha_composite(
            img, Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 0)),
                             layer, op))
    img.save(path)
    return W, H


def _load_font(size, bold=True):
    from PIL import ImageFont

    candidates = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_tracked(draw, x, y, parts, font, tracking):
    for text, fill in parts:
        for ch in text:
            draw.text((x, y), ch, font=font, fill=fill, anchor="ls")
            x += draw.textlength(ch, font=font) + tracking
    return x


def logo_png(path, dark=False):
    from PIL import Image, ImageDraw

    tmp_icon = os.path.join(HERE, "_icon_tmp.png")
    _png_mark(13, 16, 4, 3, 220, tmp_icon)
    icon = Image.open(tmp_icon).convert("RGBA")
    os.remove(tmp_icon)

    pad, gap = 48, 56
    word_font = _load_font(132)
    tag_font = _load_font(40)

    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    word = "TEXURION"
    word_w = sum(probe.textlength(ch, font=word_font) for ch in word) + 12 * 7
    tag = "TEXTURE INTELLIGENCE STUDIO"
    tag_w = sum(probe.textlength(ch, font=tag_font) for ch in tag) + 10 * (len(tag) - 1)

    W = int(pad + icon.width + gap + max(word_w, tag_w) + pad)
    H = int(pad * 2 + max(icon.height, 210))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    img.alpha_composite(icon, (int(pad), int((H - icon.height) / 2)))

    tx = int(pad + icon.width + gap)
    word_y = int(H * 0.50)
    base = (255, 255, 255, 255) if dark else (27, 36, 56, 255)
    x_col = (255, 127, 184, 255) if dark else (166, 30, 77, 255)
    tag_col = (246, 215, 230, 255) if dark else (91, 107, 140, 255)

    parts = [("TE", base), ("X", x_col), ("URION", base)]
    _draw_tracked(draw, tx, word_y, parts, word_font, 12)
    _draw_tracked(draw, tx + 2, word_y + 74, [(tag, tag_col)], tag_font, 10)
    img.save(path)


def main():
    files = {
        "texurion_icon.svg": icon_svg(),
        "texurion_favicon.svg": favicon_svg(),
        "texurion_logo.svg": logo_svg(dark=False),
        "texurion_logo_dark.svg": logo_svg(dark=True),
    }
    for name, content in files.items():
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as handle:
            handle.write(content)
        print("wrote", name)

    _png_mark(13, 16, 4, 3, 512, os.path.join(HERE, "texurion_icon.png"))
    _png_mark(7, 8, 2, 3, 256, os.path.join(HERE, "texurion_favicon.png"))
    logo_png(os.path.join(HERE, "texurion_logo.png"), dark=False)
    logo_png(os.path.join(HERE, "texurion_logo_dark.png"), dark=True)
    print("wrote PNG assets")

    cells, _ = t_cells(13, 16, 4, 3)
    shades = {}
    for col, row in cells:
        shades[pixel_color(col, row, 16)] = shades.get(pixel_color(col, row, 16), 0) + 1
    print("T pixels:", len(cells))
    for shade in SHADES:
        print(f"  {shade}: {shades.get(shade, 0)}")


if __name__ == "__main__":
    main()

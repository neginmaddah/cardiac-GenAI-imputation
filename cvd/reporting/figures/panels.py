"""Two drawing helpers shared by the endpoint figures.

Lifted verbatim out of the module that used to hold them, which also
resolved directories belonging to a retired benchmark family at import
time and so could not be shipped as-is.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

def _save_300dpi(img: Image.Image, out: Path) -> None:
    img.save(out, dpi=(300, 300))


# PIL's default bitmap font is ~11 px tall, which on a 4,196 px-wide 300-dpi
# composite renders the (a)/(b) panel labels as invisible specks -- the caption
# referred to panels the figure did not visibly label. Draw them with a real
# TrueType face sized to the image instead.
_LABEL_FONTS = ("/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf",
                "/usr/share/fonts/dejavu-serif-fonts/DejaVuSerif-Bold.ttf")


def _label_font(size: int):
    for path in _LABEL_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _panel_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int,
                 size: int = 90) -> None:
    draw.text((x, y), text, fill=(0, 0, 0), font=_label_font(size))


# ── Figure 4: binary class distributions, three endpoints ─────────────────

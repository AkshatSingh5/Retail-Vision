"""Convert the user's logo image into favicon assets.

Reads the source logo (a wide banner), crops its content box, and renders
it centered on a square canvas using the logo's own background color so it
works as a browser favicon / app icon.

Outputs:
  frontend/favicon.png            favicon (32x32)
  frontend/favicon.ico            multi-size ICO (16, 32, 48)
  frontend/apple-touch-icon.png   iOS home-screen icon (180x180)
  backend/app/static/pos/favicon.png  static POS UI favicon (256x256)

Run:  .venv\\Scripts\\python.exe scripts/generate_favicon.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

SRC = ROOT / "logo.png"  # user-provided logo (wide banner)


def load_content() -> Image.Image:
    """Crop the source image to its non-background content box."""
    img = Image.open(SRC).convert("RGB")
    bg = img.getpixel((0, 0))
    w, h = img.size
    px = img.load()

    min_x, max_x, min_y, max_y = w, 0, h, 0
    step = max(1, w // 400)
    for x in range(0, w, step):
        for y in range(h):
            c = px[x, y]
            if any(abs(c[i] - bg[i]) > 18 for i in range(3)):
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    return img.crop((min_x, min_y, max_x + 1, max_y + 1))


def square_canvas(size: int, content: Image.Image, pad_ratio: float = 0.06) -> Image.Image:
    """Center the content on a square canvas filled with the logo bg color."""
    bg = content.getpixel((0, 0))
    canvas = Image.new("RGB", (size, size), bg)
    inner = int(size * (1 - 2 * pad_ratio))
    content.thumbnail((inner, inner), Image.LANCZOS)
    x = (size - content.width) // 2
    y = (size - content.height) // 2
    canvas.paste(content, (x, y))
    return canvas


def save_sizes(master: Image.Image) -> None:
    frontend = ROOT / "frontend"
    static = ROOT / "backend" / "app" / "static" / "pos"

    master.resize((32, 32), Image.LANCZOS).save(frontend / "favicon.png", "PNG")
    master.resize((180, 180), Image.LANCZOS).save(frontend / "apple-touch-icon.png", "PNG")
    master.resize((256, 256), Image.LANCZOS).save(static / "favicon.png", "PNG")

    ico = Image.new("RGB", (256, 256), (0, 0, 0))
    ico.paste(master.resize((48, 48), Image.LANCZOS), (0, 0))
    ico.save(frontend / "favicon.ico", format="ICO", sizes=[(48, 48), (32, 32), (16, 16)])

    for name in ("favicon.png", "favicon.ico", "apple-touch-icon.png"):
        print(f"wrote {frontend / name}")
    print(f"wrote {static / 'favicon.png'}")


if __name__ == "__main__":
    if not SRC.exists():
        raise SystemExit(f"Missing source logo: {SRC}")
    content = load_content()
    print(f"content box: {content.size}")
    save_sizes(square_canvas(1024, content))
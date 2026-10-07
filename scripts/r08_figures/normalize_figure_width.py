"""Normalise every figure master to the width it is actually printed at.

Problem
-------
The manuscript includes all body figures at ``[width=0.99\\linewidth]``, which
is 142.5 mm on Elsevier preprint A4. Several figure masters were authored on
much wider canvases (up to 396 mm), so LaTeX shrank them by up to 2.8x. Type
that was designed at 7-8 pt therefore printed at roughly 2.6-5 pt, which is not
legible.

Fix
---
Rescale each vector master so its page is exactly the printed width, keeping
the aspect ratio. Because the masters are vector, the glyphs stay vector: this
is a geometric scaling of the page, not a raster resample, so no quality is
lost and the effective point size comes out at the intended value.

PDF: rebuilt with PyMuPDF ``show_pdf_page`` into a page of the target size.
SVG: ``width``/``height``/``viewBox`` rewritten; the drawing scales with it.
PNG: re-rendered from the rescaled PDF at 300 dpi so the preview matches.

Run from anywhere; paths are resolved from this file.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pymupdf

PKG = Path(__file__).resolve().parent.parent
FIGDIR = PKG / "figures"
TARGET_MM = 142.5
TARGET_PT = TARGET_MM / 25.4 * 72.0


def resize_pdf(src: Path) -> dict:
    doc = pymupdf.open(src)
    page = doc[0]
    w, h = page.rect.width, page.rect.height
    if abs(w - TARGET_PT) < 0.5:
        doc.close()
        return {"file": src.name, "action": "already target width",
                "from_mm": round(w / 72 * 25.4, 1),
                "to_mm": round(w / 72 * 25.4, 1),
                "scale": 1.0}

    scale = TARGET_PT / w
    new_h = h * scale
    out = pymupdf.open()
    newpage = out.new_page(width=TARGET_PT, height=new_h)
    newpage.show_pdf_page(newpage.rect, doc, 0)
    tmp = src.with_suffix(".resized.pdf")
    out.save(tmp, garbage=4, deflate=True)
    out.close()
    doc.close()
    tmp.replace(src)

    return {"file": src.name, "action": "rescaled",
            "from_mm": round(w / 72 * 25.4, 1),
            "to_mm": round(TARGET_PT / 72 * 25.4, 1),
            "scale": round(scale, 4)}


def resize_svg(src: Path) -> dict:
    text = src.read_text(encoding="utf-8")
    m = re.search(r'width="([0-9.]+)pt"\s+height="([0-9.]+)pt"', text)
    if not m:
        return {"file": src.name, "action": "skipped (no pt width)"}
    w_pt = float(m.group(1))
    h_pt = float(m.group(2))
    if abs(w_pt - TARGET_PT) < 0.5:
        return {"file": src.name, "action": "already target width"}
    scale = TARGET_PT / w_pt
    text = text.replace(m.group(0),
                        f'width="{TARGET_PT:.4f}pt" '
                        f'height="{h_pt * scale:.4f}pt"', 1)
    # matplotlib also writes a viewBox in pt units; if present, scale it.
    vb = re.search(r'viewBox="([0-9.\- ]+)"', text)
    if vb:
        nums = [float(v) for v in vb.group(1).split()]
        if len(nums) == 4:
            scaled = " ".join(f"{v * scale:.4f}" for v in nums)
            text = text.replace(vb.group(0), f'viewBox="{scaled}"', 1)
    src.write_text(text, encoding="utf-8")
    return {"file": src.name, "action": "rescaled",
            "from_mm": round(w_pt / 72 * 25.4, 1),
            "to_mm": round(TARGET_PT / 72 * 25.4, 1),
            "scale": round(scale, 4)}


def rerender_png(pdf: Path, png: Path) -> dict:
    doc = pymupdf.open(pdf)
    page = doc[0]
    zoom = 300.0 / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    pix.save(png)
    doc.close()
    return {"file": png.name, "pixels": f"{pix.width}x{pix.height}"}


def main() -> None:
    report = []
    for pdf in sorted(FIGDIR.glob("*.pdf")):
        if pdf.name == "fig_graphical_abstract.pdf":
            report.append({"file": pdf.name,
                           "action": "skipped (standalone asset)"})
            continue
        report.append(resize_pdf(pdf))
        png = pdf.with_suffix(".png")
        if png.exists():
            report.append(rerender_png(pdf, png))
        svg = pdf.with_suffix(".svg")
        if svg.exists():
            report.append(resize_svg(svg))

    print(f"{'file':<34s} {'action':<24s} {'from mm':>8s} {'to mm':>8s} "
          f"{'scale':>7s}")
    for r in report:
        print(f"{r['file']:<34s} {r.get('action', 'png re-rendered'):<24s} "
              f"{r.get('from_mm', ''):>8} {r.get('to_mm', ''):>8} "
              f"{r.get('scale', ''):>7}")

    print("\nfinal page widths (mm):")
    for pdf in sorted(FIGDIR.glob("*.pdf")):
        doc = pymupdf.open(pdf)
        r = doc[0].rect
        print(f"  {pdf.name:<34s} {r.width / 72 * 25.4:7.2f} x "
              f"{r.height / 72 * 25.4:6.2f}")
        doc.close()


if __name__ == "__main__":
    main()

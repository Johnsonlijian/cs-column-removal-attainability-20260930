"""Measure the printed point size of every figure at the manuscript include width.

A figure is included at [width=0.99\\linewidth] = 142.5 mm. If a master page is
wider than that, LaTeX scales it down and the printed type scales with it, so the
effective size is (internal pt) x (142.5 / master width in mm). This script
reports that number for every master, plus a clipping check on the page edges.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pymupdf

PKG = Path(__file__).resolve().parent.parent
FIGDIR = PKG / "figures"
INCLUDE_MM = 142.5
TARGET_PT = INCLUDE_MM / 25.4 * 72.0


def edge_ink(page) -> int:
    """Count non-white pixels in the outermost 2 rows/columns at 300 dpi."""
    zoom = 300.0 / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    w, h, n = pix.width, pix.height, pix.n
    samples = pix.samples
    bad = 0
    for y in list(range(2)) + list(range(h - 2, h)):
        row = y * pix.stride
        for x in range(w):
            off = row + x * n
            if samples[off] < 250 or samples[off + 1] < 250 or samples[off + 2] < 250:
                bad += 1
    for x in list(range(2)) + list(range(w - 2, w)):
        for y in range(h):
            off = y * pix.stride + x * n
            if samples[off] < 250 or samples[off + 1] < 250 or samples[off + 2] < 250:
                bad += 1
    return bad


def main() -> None:
    rows = []
    for path in sorted(glob.glob(str(FIGDIR / "*.pdf"))):
        doc = pymupdf.open(path)
        page = doc[0]
        rect = page.rect
        width_mm = rect.width / 72.0 * 25.4
        scale = TARGET_PT / rect.width
        sizes = [s["size"]
                 for b in page.get_text("dict")["blocks"] if b["type"] == 0
                 for l in b["lines"] for s in l["spans"]]
        n = len(sizes)
        eff_min = (min(sizes) if sizes else 0.0) * scale
        eff_mean = (sum(sizes) / n if n else 0.0) * scale
        rows.append({
            "file": os.path.basename(path),
            "pages": doc.page_count,
            "width_mm": round(width_mm, 1),
            "scale": round(scale, 3),
            "spans": n,
            "internal_min_pt": round(min(sizes), 2) if sizes else 0.0,
            "effective_min_pt": round(eff_min, 2),
            "effective_mean_pt": round(eff_mean, 2),
            "edge_ink_px": edge_ink(page),
        })
        doc.close()

    print(f"{'figure':<32s} {'pg':>3s} {'w_mm':>7s} {'scale':>6s} "
          f"{'eff_min':>8s} {'eff_mean':>9s} {'edge':>6s} {'verdict':>10s}")
    worst = 99.0
    for r in rows:
        ok = r["effective_min_pt"] >= 7.0 and r["edge_ink_px"] == 0
        worst = min(worst, r["effective_min_pt"])
        print(f"{r['file']:<32s} {r['pages']:>3d} {r['width_mm']:>7.1f} "
              f"{r['scale']:>6.3f} {r['effective_min_pt']:>8.2f} "
              f"{r['effective_mean_pt']:>9.2f} {r['edge_ink_px']:>6d} "
              f"{'OK' if ok else 'CHECK':>10s}")
    print()
    print(f"worst effective minimum point size: {worst:.2f} pt")
    print("body figures meet the 7 pt floor at the include width:",
          all(r["effective_min_pt"] >= 7.0
              for r in rows if "graphical_abstract" not in r["file"]))


if __name__ == "__main__":
    main()

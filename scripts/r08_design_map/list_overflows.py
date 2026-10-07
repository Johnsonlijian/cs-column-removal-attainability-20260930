"""Report every text span that overflows the figure page, with its content.

Overflow means ink reaches the canvas boundary, which LaTeX will either clip or
which indicates the layout no longer fits the intended printed size.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pymupdf

PKG = Path(__file__).resolve().parent.parent
FIGDIR = PKG / "figures"


def main() -> None:
    for path in sorted(glob.glob(str(FIGDIR / "*.pdf"))):
        doc = pymupdf.open(path)
        page = doc[0]
        W, H = page.rect.width, page.rect.height
        over = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for s in line["spans"]:
                    x0, y0, x1, y1 = s["bbox"]
                    if x0 < 1 or y0 < 1 or x1 > W - 1 or y1 > H - 1:
                        over.append((round(x0), round(y0), round(x1), round(y1),
                                     round(s["size"], 1), s["text"][:44]))
        print(f"=== {os.path.basename(path)} "
              f"({W:.0f}x{H:.0f} pt = {W/72*25.4:.1f}x{H/72*25.4:.1f} mm), "
              f"overflows: {len(over)}")
        for o in over[:8]:
            print(f"    x0={o[0]:5d} y0={o[1]:5d} x1={o[2]:5d} y1={o[3]:5d} "
                  f"{o[4]:5.1f}pt  {o[5]!r}")
        doc.close()


if __name__ == "__main__":
    main()

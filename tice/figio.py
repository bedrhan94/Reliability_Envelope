"""One place that decides how a figure reaches disk.

Springer wants line art as vector: a chart is lines and text, and a vector file stores
them as geometry rather than as a pixel grid, so it stays sharp at any size and prints
correctly. Every figure here is line art, so each one is written twice -- a PDF for the
typesetter and a PNG for anything that cannot read PDF (previews, the Word build).

`\\includegraphics{path-without-extension}` makes pdflatex prefer the PDF, so the
manuscript can name a figure once and get the vector copy automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

# Raster fallback resolution. Only the PNG uses it; the PDF is resolution-independent.
FIG_DPI = int(os.environ.get("TICE_FIG_DPI", "600"))


def save_figure(fig, out: str | Path, dpi: int | None = None, **kwargs) -> Path:
    """Write `fig` as both PDF (vector, for typesetting) and PNG (raster fallback).

    `out` may carry any extension or none; both siblings are written beside it. Returns
    the PDF path, which is the one the manuscript should reference.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")

    pdf = stem.with_suffix(".pdf")
    # metadata=... keeps the creation date out of the file so rebuilding an unchanged
    # figure produces an identical PDF and does not show up as a spurious diff.
    fig.savefig(pdf, format="pdf", metadata={"CreationDate": None}, **kwargs)

    png = stem.with_suffix(".png")
    fig.savefig(png, format="png", dpi=dpi or FIG_DPI, **kwargs)
    return pdf

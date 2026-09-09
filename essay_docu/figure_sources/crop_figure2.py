"""Remove the embedded banner from Figure 2 while preserving vector content.

The paper scales the cropped page to the text width, so trimming the unused
banner and margins also makes the remaining labels materially larger.
"""

from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figure_sources" / "legacy_02_hex_multiscale_multiangle_kernels.pdf"
OUTPUT = ROOT / "figs" / "02_hex_multiscale_multiangle_kernels.pdf"


def main() -> None:
    reader = PdfReader(SOURCE)
    page = reader.pages[0]
    # Original MediaBox: 1122.48 x 485.402 pt.  The title occupies the top
    # ~39 pt; the bottom line is an explanatory banner duplicated by the
    # manuscript caption.  Tight side margins enlarge the remaining diagram.
    page.cropbox.lower_left = (18, 31)
    page.cropbox.upper_right = (1105, 456)
    page.mediabox.lower_left = page.cropbox.lower_left
    page.mediabox.upper_right = page.cropbox.upper_right

    writer = PdfWriter()
    writer.add_page(page)
    with OUTPUT.open("wb") as handle:
        writer.write(handle)


if __name__ == "__main__":
    main()

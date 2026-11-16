"""A tiny pure-Python PDF builder (no dependencies) for the docling scenario.

Produces a valid single-font, text-only PDF 1.4 whose pages carry real text
operators, so a text extractor (Docling, MarkItDown) can read the words back.
"""

from __future__ import annotations


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(lines: list[str], *, title: str = "acceptance probe") -> bytes:
    """One US-Letter page, Helvetica 12pt, one text line per entry in ``lines``."""
    ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
    for i, line in enumerate(lines):
        ops.append(f"({_esc(line)}) Tj" if i == 0 else f"T* ({_esc(line)}) Tj")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Title (" + _esc(title).encode("latin-1", "replace") + b") >>",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def chart_pixels(width: int = 320, height: int = 200) -> bytes:
    """Raw RGB bytes of a plain candlestick-looking figure (white page, green/red bars).

    Deterministic and dependency-free, so a PDF built from it is byte-stable.
    """
    rows = []
    bars = [(0.30, 0.55, True), (0.40, 0.70, True), (0.50, 0.65, False), (0.35, 0.60, True),
            (0.25, 0.50, False), (0.45, 0.80, True), (0.55, 0.75, True)]
    slot = width // (len(bars) + 1)
    for y in range(height):
        row = bytearray(b"\xff" * (width * 3))
        # axis + a support line
        for x in range(width):
            if y == height - 6 or x == 5:
                row[x * 3:x * 3 + 3] = b"\x30\x30\x30"
            if y == int(height * 0.62):
                row[x * 3:x * 3 + 3] = b"\x90\x90\xd0"
        for i, (lo, hi, up) in enumerate(bars):
            x0 = slot * (i + 1) - slot // 4
            top, bot = int(height * (1 - hi)), int(height * (1 - lo))
            if top <= y <= bot:
                color = b"\x1a\xa0\x4a" if up else b"\xd0\x30\x30"
                for x in range(max(x0, 0), min(x0 + slot // 2, width)):
                    row[x * 3:x * 3 + 3] = color
        rows.append(bytes(row))
    return b"".join(rows)


def make_pdf_with_image(
    lines: list[str], *, width: int = 320, height: int = 200, title: str = "acceptance probe"
) -> bytes:
    """A one-page PDF: the text ``lines`` plus an embedded RGB figure (Flate-compressed XObject).

    The figure is what a figure extractor (Docling with image extraction on) should
    pick up; the text keeps the page parseable when it does not.
    """
    import zlib

    pixels = zlib.compress(chart_pixels(width, height))
    ops = ["BT", "/F1 12 Tf", "14 TL", "72 740 Td"]
    for i, line in enumerate(lines):
        ops.append(f"({_esc(line)}) Tj" if i == 0 else f"T* ({_esc(line)}) Tj")
    ops.append("ET")
    # draw the figure at 1.5x, below the text block
    ops += ["q", f"{width * 1.5:.0f} 0 0 {height * 1.5:.0f} 72 380 cm", "/Im1 Do", "Q"]
    content = "\n".join(ops).encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> /XObject << /Im1 7 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Title (" + _esc(title).encode("latin-1", "replace") + b") >>",
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(pixels)} >>\nstream\n"
        ).encode() + pixels + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)

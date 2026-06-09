#!/usr/bin/env python3
"""
build_pdf.py — Convert report/report.md into a polished PDF with the six figures
actually embedded, using reportlab (per the 'pdf' skill: reportlab Platypus).

Design notes
------------
* Body text uses DejaVu Serif, headings/captions DejaVu Sans, inline code
  DejaVu Sans Mono. All are TrueType with wide Unicode coverage, so glyphs such
  as sigma, multiplication sign, approx, minus, arrows and dashes render
  correctly instead of as the black boxes the 'pdf' skill warns about.
* Every character used in the document is checked against the registered fonts'
  cmaps; any glyph with no coverage is substituted with a safe ASCII fallback so
  the PDF can never contain a missing-glyph box.
* Figures are embedded from analysis/figures/ via Image flowables (guaranteed
  embedding), sized to the text width with an aspect-ratio-preserving height cap,
  and kept together with their caption.

This script only READS analysis/figures/*.png and report/report.md and WRITES
under report/. It never touches anything under data/.
"""

import os
import re
import sys

from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
    KeepTogether, HRFlowable, ListFlowable, ListItem,
)
from reportlab.pdfbase.ttfonts import TTFError

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MD_PATH = os.path.join(HERE, "report.md")
PDF_PATH = os.path.join(HERE, "Rohan_Balu_Research_Report.pdf")

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
import matplotlib  # noqa: E402  (only used to locate bundled DejaVu TTFs)
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")

FONT_FILES = {
    "Body":          "DejaVuSerif.ttf",
    "Body-Bold":     "DejaVuSerif-Bold.ttf",
    "Body-Italic":   "DejaVuSerif-Italic.ttf",
    "Body-BoldItalic": "DejaVuSerif-BoldItalic.ttf",
    "Head":          "DejaVuSans.ttf",
    "Head-Bold":     "DejaVuSans-Bold.ttf",
    "Head-Italic":   "DejaVuSans-Oblique.ttf",
    "Mono":          "DejaVuSansMono.ttf",
    "Mono-Bold":     "DejaVuSansMono-Bold.ttf",
}

_registered_cmaps = {}


def register_fonts():
    """Register DejaVu fonts and remember their cmaps for glyph checking."""
    from fontTools.ttLib import TTFont as FT_TTFont  # bundled with matplotlib deps
    for name, fname in FONT_FILES.items():
        path = os.path.join(FONT_DIR, fname)
        pdfmetrics.registerFont(TTFont(name, path))
        ft = FT_TTFont(path)
        cmap = set()
        for table in ft["cmap"].tables:
            cmap.update(table.cmap.keys())
        _registered_cmaps[name] = cmap
        ft.close()

    # Map families so <b>/<i> markup resolves to the right TTF.
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Body-Bold",
        italic="Body-Italic", boldItalic="Body-BoldItalic",
    )
    pdfmetrics.registerFontFamily(
        "Head", normal="Head", bold="Head-Bold", italic="Head-Italic", boldItalic="Head-Bold",
    )
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold")


# Characters that are not guaranteed in the fonts get a readable ASCII fallback.
GLYPH_FALLBACK = {
    "⊙": "(x)",   # circled dot operator -> element-wise product
    "⊗": "(x)",   # circled times
    "→": "->",    # rightwards arrow
    "−": "-",     # minus sign -> hyphen
    "≈": "~",     # almost equal to
    "≤": "<=",    # less-than or equal
    "≥": ">=",    # greater-than or equal
    "×": "x",     # multiplication sign
    "·": ".",     # middle dot
    "—": "-",     # em dash
    "–": "-",     # en dash
    "σ": "sigma", # greek small sigma
    "…": "...",   # ellipsis
    "’": "'",     # right single quote
    "“": '"',     # left double quote
    "”": '"',     # right double quote
    "‘": "'",     # left single quote
}

_missing_reported = set()


def covered(ch, fonts=("Body", "Body-Bold", "Body-Italic", "Head", "Head-Bold", "Mono")):
    """True if every listed font has a glyph for ch."""
    cp = ord(ch)
    return all(cp in _registered_cmaps[f] for f in fonts)


def sanitize(text):
    """Replace any character lacking glyph coverage with a safe fallback."""
    out = []
    for ch in text:
        if ch in ("\n", "\t") or ord(ch) < 128 or covered(ch):
            out.append(ch)
        elif ch in GLYPH_FALLBACK:
            out.append(GLYPH_FALLBACK[ch])
        else:
            if ch not in _missing_reported:
                _missing_reported.add(ch)
                print(f"  [warn] no glyph for U+{ord(ch):04X} {ch!r} -> '?'")
            out.append("?")
    return "".join(out)


# ---------------------------------------------------------------------------
# Inline markdown -> reportlab mini-markup
# ---------------------------------------------------------------------------
def inline(text):
    """Convert inline **bold**, *italic*, `code` to reportlab markup, XML-safe."""
    # 1. Protect inline code spans first so * and _ inside them are literal.
    code_spans = []

    def _stash_code(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash_code, text)

    # 2. XML-escape the remaining prose.
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 3. Bold then italic (bold first so ** is consumed before single *).
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)

    # 4. Restore code spans as monospace, XML-escaped.
    def _restore_code(m):
        raw = code_spans[int(m.group(1))]
        raw = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<font face="Mono" size="9">{raw}</font>'

    text = re.sub(r"\x00CODE(\d+)\x00", _restore_code, text)
    return sanitize(text)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
ACCENT = colors.HexColor("#1F3A5F")      # deep navy
ACCENT2 = colors.HexColor("#2E6171")     # teal
LIGHT = colors.HexColor("#EEF2F6")       # callout / header fill
RULE = colors.HexColor("#9AA7B4")

STYLES = {
    "Title": ParagraphStyle("Title", fontName="Head-Bold", fontSize=21, leading=26,
                             textColor=ACCENT, spaceAfter=4, alignment=TA_LEFT),
    "Subtitle": ParagraphStyle("Subtitle", fontName="Head-Italic", fontSize=12.5,
                               leading=16, textColor=ACCENT2, spaceAfter=10),
    "Meta": ParagraphStyle("Meta", fontName="Body", fontSize=9.5, leading=14,
                           textColor=colors.HexColor("#333333"), spaceAfter=2),
    "H2": ParagraphStyle("H2", fontName="Head-Bold", fontSize=14.5, leading=18,
                         textColor=ACCENT, spaceBefore=14, spaceAfter=6),
    "H3": ParagraphStyle("H3", fontName="Head-Bold", fontSize=11.5, leading=15,
                         textColor=ACCENT2, spaceBefore=9, spaceAfter=4),
    "Body": ParagraphStyle("Body", fontName="Body", fontSize=10, leading=14.5,
                           alignment=TA_JUSTIFY, spaceAfter=7,
                           textColor=colors.HexColor("#1A1A1A")),
    "Bullet": ParagraphStyle("Bullet", fontName="Body", fontSize=10, leading=14.5,
                             alignment=TA_LEFT, spaceAfter=3,
                             textColor=colors.HexColor("#1A1A1A")),
    "Caption": ParagraphStyle("Caption", fontName="Head-Italic", fontSize=8.6,
                              leading=11.5, alignment=TA_CENTER,
                              textColor=colors.HexColor("#444444"),
                              spaceBefore=4, spaceAfter=6),
    "Callout": ParagraphStyle("Callout", fontName="Body", fontSize=9.6, leading=13.5,
                              alignment=TA_LEFT, textColor=colors.HexColor("#20303f")),
    "TblHead": ParagraphStyle("TblHead", fontName="Head-Bold", fontSize=8.8,
                              leading=11, textColor=colors.white, alignment=TA_LEFT),
    "TblCell": ParagraphStyle("TblCell", fontName="Body", fontSize=8.7, leading=11,
                              textColor=colors.HexColor("#1A1A1A"), alignment=TA_LEFT),
}

PAGE_W, PAGE_H = letter
LMARGIN = RMARGIN = 0.85 * inch
TMARGIN = 0.8 * inch
BMARGIN = 0.8 * inch
USABLE_W = PAGE_W - LMARGIN - RMARGIN

FIG_DIR = os.path.join(REPO, "analysis", "figures")


# ---------------------------------------------------------------------------
# Figure flowable
# ---------------------------------------------------------------------------
def make_figure(alt, path):
    """Return a KeepTogether[Image, caption] sized to the text column."""
    # Resolve path relative to the markdown file, then fall back to FIG_DIR.
    candidates = [
        os.path.normpath(os.path.join(HERE, path)),
        os.path.join(FIG_DIR, os.path.basename(path)),
    ]
    src = next((c for c in candidates if os.path.exists(c)), None)
    if src is None:
        raise FileNotFoundError(f"Figure not found for {path!r}: tried {candidates}")

    with PILImage.open(src) as im:
        iw, ih = im.size
    max_w = USABLE_W
    max_h = 4.15 * inch
    w = max_w
    h = w * ih / iw
    if h > max_h:
        h = max_h
        w = h * iw / ih
    img = RLImage(src, width=w, height=h)
    img.hAlign = "CENTER"
    cap = Paragraph(inline(alt), STYLES["Caption"])
    return KeepTogether([Spacer(1, 4), img, cap, Spacer(1, 2)]), os.path.basename(src)


# ---------------------------------------------------------------------------
# Table flowable
# ---------------------------------------------------------------------------
def _strip_md(text):
    """Approximate the visible text of a cell (drop markup) for width measuring."""
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "").replace("*", "")
    return text


def make_table(rows):
    """rows: list of list-of-str cells (header is rows[0])."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    pad = 2 * 6 + 2  # left+right padding plus slack

    # Per column: a minimum that fits the longest unbreakable WORD (so short
    # headers like "Params" never wrap mid-word), and a "desired" width that
    # would hold the longest full cell on one line.
    min_col = [0.0] * ncols
    des_col = [0.0] * ncols
    for ri, row in enumerate(rows):
        font, size = ("Head-Bold", 8.8) if ri == 0 else ("Body", 8.7)
        for c in range(ncols):
            vis = sanitize(_strip_md(row[c]))
            words = vis.split()
            longest_word = max((stringWidth(w, font, size) for w in words), default=0)
            full = stringWidth(vis, font, size)
            min_col[c] = max(min_col[c], longest_word)
            des_col[c] = max(des_col[c], full)
    min_col = [w + pad for w in min_col]
    des_col = [w + pad for w in des_col]

    if sum(min_col) >= USABLE_W:
        scale = USABLE_W / sum(min_col)
        widths = [w * scale for w in min_col]
    else:
        widths = list(min_col)
        remaining = USABLE_W - sum(widths)
        demand = [des_col[c] - min_col[c] for c in range(ncols)]
        td = sum(demand)
        if td > 0:
            widths = [widths[c] + remaining * demand[c] / td for c in range(ncols)]
        else:
            widths = [w + remaining / ncols for w in widths]

    data = []
    for ri, row in enumerate(rows):
        style = STYLES["TblHead"] if ri == 0 else STYLES["TblCell"]
        data.append([Paragraph(inline(cell), style) for cell in row])

    tbl = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
    ]
    for ri in range(1, len(rows)):
        if ri % 2 == 0:
            ts.append(("BACKGROUND", (0, ri), (-1, ri), LIGHT))
    tbl.setStyle(TableStyle(ts))
    return tbl


def make_callout(text):
    """A left-bordered shaded callout box for blockquotes."""
    p = Paragraph(inline(text), STYLES["Callout"])
    tbl = Table([[p]], colWidths=[USABLE_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT2),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether([Spacer(1, 2), tbl, Spacer(1, 6)])


# ---------------------------------------------------------------------------
# Markdown parser -> flowables
# ---------------------------------------------------------------------------
IMG_RE = re.compile(r"^!\[(.*)\]\((.*)\)\s*$")


def parse_markdown(md):
    lines = md.split("\n")
    flow = []
    embedded = []      # figure basenames actually added
    i = 0
    n = len(lines)
    para_buf = []
    bullet_buf = []    # list of (kind, text): kind in {'ul','ol'}

    def flush_para():
        nonlocal para_buf
        if para_buf:
            text = "<br/>".join(para_buf)
            flow.append(Paragraph(inline(text), STYLES["Body"]))
            para_buf = []

    def flush_bullets():
        nonlocal bullet_buf
        if not bullet_buf:
            return
        kind = bullet_buf[0][0]
        items = []
        for _, t in bullet_buf:
            items.append(ListItem(Paragraph(inline(t), STYLES["Bullet"]),
                                  leftIndent=18, value=None))
        if kind == "ol":
            lf = ListFlowable(items, bulletType="1", bulletFontName="Body",
                              bulletFontSize=10, leftIndent=20, bulletColor=ACCENT)
        else:
            lf = ListFlowable(items, bulletType="bullet", bulletChar="•",
                              bulletFontName="Body", bulletFontSize=10,
                              leftIndent=18, bulletColor=ACCENT2)
        flow.append(lf)
        flow.append(Spacer(1, 4))
        bullet_buf = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # blank line: paragraph / list boundary
        if stripped == "":
            flush_para()
            flush_bullets()
            i += 1
            continue

        # image / figure
        m = IMG_RE.match(stripped)
        if m:
            flush_para(); flush_bullets()
            fig, base = make_figure(m.group(1), m.group(2))
            flow.append(fig)
            embedded.append(base)
            i += 1
            continue

        # table block: consecutive lines starting with '|'
        if stripped.startswith("|"):
            flush_para(); flush_bullets()
            tbl_lines = []
            while i < n and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tl in tbl_lines:
                cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                # skip the |---|---| separator row
                if all(set(c) <= set("-: ") and c != "" for c in cells):
                    continue
                if all(c == "" for c in cells):
                    continue
                rows.append(cells)
            if rows:
                flow.append(Spacer(1, 2))
                flow.append(make_table(rows))
                flow.append(Spacer(1, 8))
            continue

        # horizontal rule (standalone --- / *** )
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush_para(); flush_bullets()
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                   spaceBefore=2, spaceAfter=8))
            i += 1
            continue

        # blockquote (one or more leading '>' lines)
        if stripped.startswith(">"):
            flush_para(); flush_bullets()
            q_lines = []
            while i < n and lines[i].strip().startswith(">"):
                q_lines.append(lines[i].strip()[1:].strip())
                i += 1
            flow.append(make_callout(" ".join(q_lines)))
            continue

        # headings
        if stripped.startswith("#"):
            flush_para(); flush_bullets()
            hashes = len(stripped) - len(stripped.lstrip("#"))
            content = stripped[hashes:].strip()
            if hashes == 1:
                flow.append(Paragraph(inline(content), STYLES["Title"]))
            elif hashes == 2:
                flow.append(Paragraph(inline(content), STYLES["H2"]))
                flow.append(HRFlowable(width="100%", thickness=1.1, color=ACCENT,
                                       spaceBefore=1, spaceAfter=6))
            else:
                flow.append(Paragraph(inline(content), STYLES["H3"]))
            i += 1
            continue

        # bullet list item
        mb = re.match(r"^[-*]\s+(.*)$", stripped)
        if mb:
            flush_para()
            if bullet_buf and bullet_buf[0][0] != "ul":
                flush_bullets()
            bullet_buf.append(("ul", mb.group(1)))
            i += 1
            continue

        # ordered list item
        mo = re.match(r"^\d+\.\s+(.*)$", stripped)
        if mo:
            flush_para()
            if bullet_buf and bullet_buf[0][0] != "ol":
                flush_bullets()
            bullet_buf.append(("ol", mo.group(1)))
            i += 1
            continue

        # plain paragraph text (sub-title line handled specially)
        flush_bullets()
        para_buf.append(stripped)
        i += 1

    flush_para()
    flush_bullets()
    return flow, embedded


# ---------------------------------------------------------------------------
# Page furniture (footer with page number + running title)
# ---------------------------------------------------------------------------
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Head", 8)
    canvas.setFillColor(colors.HexColor("#7A8794"))
    # footer rule
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(LMARGIN, BMARGIN - 12, PAGE_W - RMARGIN, BMARGIN - 12)
    canvas.drawString(LMARGIN, BMARGIN - 24,
                      "Fetal-Head Biometry Landmark Localization · Rohan Balu")
    canvas.drawRightString(PAGE_W - RMARGIN, BMARGIN - 24, f"Page {doc.page}")
    canvas.restoreState()


def main():
    register_fonts()
    with open(MD_PATH, "r", encoding="utf-8") as fh:
        md = fh.read()
    flow, embedded = parse_markdown(md)

    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=letter,
        leftMargin=LMARGIN, rightMargin=RMARGIN,
        topMargin=TMARGIN, bottomMargin=BMARGIN,
        title="Fetal-Head Biometry Landmark Localization",
        author="Rohan Balu",
    )
    doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)

    size = os.path.getsize(PDF_PATH)
    print(f"\nPDF written: {PDF_PATH}")
    print(f"Size: {size/1024:.1f} KB ({size} bytes)")
    print(f"Figures embedded (in order): {len(embedded)}")
    for b in embedded:
        print(f"   - {b}")
    if _missing_reported:
        print(f"[warn] characters with no glyph: {sorted(_missing_reported)}")
    else:
        print("Glyph check: all characters covered, no fallback boxes.")


if __name__ == "__main__":
    main()

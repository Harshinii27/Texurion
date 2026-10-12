"""Export helpers: save analysis artefacts and build PDF reports."""

import json
import os
from datetime import datetime

import numpy as np

# matplotlib is rendered head-less so exports work without a display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _to_uint8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if array.max() <= 1.0 and array.min() >= 0.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype("uint8")


def _save_rgb(array, path):
    Image.fromarray(_to_uint8(array)).save(path)


def render_histogram_png(hist, path, title="LBP Histogram", color="#4f8cff"):
    """Render a styled LBP histogram to a PNG file."""

    hist = np.asarray(hist, dtype="float64")
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    fig.patch.set_facecolor("#161a23")
    ax.set_facecolor("#161a23")

    x = np.arange(hist.size)
    ax.bar(x, hist, width=0.85, color=color, edgecolor="none")
    ax.plot(x, hist, color="#ffd166", linewidth=1.2, alpha=0.9)

    ax.set_title(title, color="#e8ecf5", fontsize=11)
    ax.set_xlabel("LBP pattern bin", color="#9aa4b2", fontsize=8)
    ax.set_ylabel("Frequency", color="#9aa4b2", fontsize=8)
    ax.tick_params(colors="#9aa4b2", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a3140")
    ax.grid(axis="y", color="#2a3140", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_bundle(result, outputs_dir, original_rgb, gray_rgb, lbp_rgb, hist_png=None):
    """Save images, histogram and a JSON summary for an analysis result."""

    _ensure_dir(outputs_dir)
    name = result.name

    paths = {
        "original": os.path.join(outputs_dir, f"{name}_original.png"),
        "gray": os.path.join(outputs_dir, f"{name}_grayscale.png"),
        "lbp": os.path.join(outputs_dir, f"{name}_lbp.png"),
    }

    _save_rgb(original_rgb, paths["original"])
    _save_rgb(gray_rgb, paths["gray"])
    _save_rgb(lbp_rgb, paths["lbp"])

    if hist_png is None:
        hist_png = os.path.join(outputs_dir, f"{name}_histogram.png")
        render_histogram_png(result.hist, hist_png)
    paths["histogram"] = hist_png

    summary = {
        "image": result.name,
        "source": result.path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "radius": result.radius,
            "points": result.points,
            "method": result.method,
            "bins": result.nbins,
        },
        "classification": {
            "label": result.label,
            "confidence": round(result.confidence, 2),
            "roughness": round(result.classification.get("roughness", 0.0), 4),
        },
        "features": {k: round(float(v), 6) for k, v in result.features.items()},
    }

    json_path = os.path.join(outputs_dir, f"{name}_result.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    paths["json"] = json_path
    return paths


def _pdf_available():
    try:
        import reportlab  # noqa: F401
    except Exception:
        return False
    return True


def build_pdf_report(result, image_paths, reports_dir, histogram_png=None):
    """Build a one page PDF report.

    Returns the path to the PDF, or ``None`` when reportlab is unavailable.
    """

    if not _pdf_available():
        return None

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    _ensure_dir(reports_dir)
    pdf_path = os.path.join(reports_dir, f"{result.name}_report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Texture Analysis - {result.name}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleX", parent=styles["Title"], textColor=colors.HexColor("#1b2a4a"),
        fontSize=20, spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "SubX", parent=styles["Normal"], textColor=colors.HexColor("#5b6b8c"),
        fontSize=10, spaceAfter=10,
    )
    head_style = ParagraphStyle(
        "HeadX", parent=styles["Heading2"], textColor=colors.HexColor("#1b2a4a"),
        fontSize=12, spaceBefore=10, spaceAfter=6,
    )

    story = [
        Paragraph("Texture Analysis Report", title_style),
        Paragraph(
            f"Image: <b>{result.name}</b> &nbsp;|&nbsp; "
            f"{datetime.now().strftime('%d %b %Y, %H:%M')}",
            sub_style,
        ),
    ]

    info_rows = [
        ["Parameter", "Value"],
        ["LBP method", result.method],
        ["Radius", str(result.radius)],
        ["Points", str(result.points)],
        ["Histogram bins", str(result.nbins)],
        ["Predicted class", result.label],
        ["Confidence", f"{result.confidence:.1f}%"],
        ["Roughness", f"{result.classification.get('roughness', 0.0):.3f}"],
    ]
    info_table = Table(info_rows, colWidths=[60 * mm, 100 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2a4a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [Paragraph("Analysis Summary", head_style), info_table]

    # Image strip.
    def _thumb(path, width=52 * mm):
        return RLImage(path, width=width, height=width * 0.72)

    thumbs = []
    labels = []
    for key, caption in (("original", "Original"), ("lbp", "LBP map")):
        if image_paths.get(key):
            thumbs.append(_thumb(image_paths[key]))
            labels.append(caption)
    if thumbs:
        img_table = Table([labels, thumbs], colWidths=[56 * mm] * len(thumbs))
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#5b6b8c")),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ]))
        story += [Paragraph("Visual Comparison", head_style), img_table]

    if histogram_png and os.path.isfile(histogram_png):
        story += [
            Paragraph("LBP Histogram", head_style),
            RLImage(histogram_png, width=150 * mm, height=71 * mm),
        ]

    # Feature table (two columns).
    feature_items = list(result.features.items())
    rows = [["Feature", "Value", "Feature", "Value"]]
    half = (len(feature_items) + 1) // 2
    for i in range(half):
        left = feature_items[i]
        right = feature_items[i + half] if i + half < len(feature_items) else ("", "")
        rows.append([
            left[0], f"{left[1]:.4f}",
            right[0], f"{right[1]:.4f}" if right[0] else "",
        ])

    feat_table = Table(rows, colWidths=[42 * mm, 30 * mm, 42 * mm, 30 * mm])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f8cff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story += [
        Spacer(1, 6),
        Paragraph("Texture Features", head_style),
        feat_table,
        Spacer(1, 10),
        Paragraph(
            f"<i>{result.interpretation}</i>",
            ParagraphStyle("ItX", parent=styles["Normal"],
                           textColor=colors.HexColor("#3d4f74"), fontSize=9),
        ),
    ]

    doc.build(story)
    return pdf_path


def batch_csv(results, reports_dir,
              filename=None):
    """Write a CSV summary for a batch of results."""

    import csv

    if filename is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_summary_{stamp}.csv"

    _ensure_dir(reports_dir)
    path = os.path.join(reports_dir, filename)

    feature_keys = sorted({k for r in results for k in r.features})
    header = ["image", "label", "confidence", "roughness",
              "radius", "points", "method", "bins"] + feature_keys

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for r in results:
            row = [
                r.name, r.label, round(r.confidence, 2),
                round(r.classification.get("roughness", 0.0), 4),
                r.radius, r.points, r.method, r.nbins,
            ]
            row += [round(float(r.features.get(k, 0.0)), 6) for k in feature_keys]
            writer.writerow(row)

    return path

"""Branded PDF and CSV report builders for Texurion.

Reports contain only real, computed values from the saved analysis record.
"""

import os
from datetime import datetime

from . import branding, runtime, store

ACCENT = "#A61E4D"
ACCENT_SOFT = "#C2185B"
INK = "#1b2438"
MUTED = "#5b6b8c"


def _available():
    try:
        import reportlab  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _styles():
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TxTitle", parent=base["Title"], fontSize=20,
                                textColor=colors.HexColor(ACCENT), spaceAfter=2),
        "sub": ParagraphStyle("TxSub", parent=base["Normal"], fontSize=10,
                              textColor=colors.HexColor(MUTED), spaceAfter=10),
        "head": ParagraphStyle("TxHead", parent=base["Heading2"], fontSize=12,
                               textColor=colors.HexColor(INK), spaceBefore=10,
                               spaceAfter=6),
        "body": ParagraphStyle("TxBody", parent=base["Normal"], fontSize=9,
                               textColor=colors.HexColor(INK), spaceAfter=4),
        "small": ParagraphStyle("TxSmall", parent=base["Normal"], fontSize=8,
                                textColor=colors.HexColor(MUTED)),
    }


def _logo_flowable(height_mm=15):
    """Scaled pixel-T mark for the report letterhead, or None."""

    path = branding.icon_path()
    if not path:
        return None
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage

    icon_w, icon_h = 425, 512
    height = height_mm * mm
    return RLImage(path, width=height * icon_w / icon_h, height=height)


def _title_block(title, subtitle, styles):
    """Return a brand letterhead: pixel-T mark beside the report title."""

    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Table, TableStyle

    text_cell = [Paragraph(title, styles["title"]),
                 Paragraph(subtitle, styles["sub"])]
    logo = _logo_flowable(15)
    if logo is None:
        return text_cell
    table = Table([[logo, text_cell]], colWidths=[26 * mm, None])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 8),
        ("LEFTPADDING", (1, 0), (1, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return [table]


def _meta_table(rows, headers=("Field", "Value")):
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    data = [list(headers)] + rows
    table = Table(data, colWidths=[55 * mm, 105 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e3c8d4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fdf3f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _metric_table(metrics):
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    items = list(metrics.items())
    half = (len(items) + 1) // 2
    rows = [["Metric", "Value", "Metric", "Value"]]
    for i in range(half):
        left = items[i]
        right = items[i + half] if i + half < len(items) else ("", "")
        rows.append([left[0], f"{left[1]:.4f}",
                     right[0], f"{right[1]:.4f}" if right[0] else ""])
    table = Table(rows, colWidths=[38 * mm, 26 * mm, 38 * mm, 26 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT_SOFT)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e3c8d4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fdf3f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def build_analysis_pdf(record):
    """Build a Texurion PDF for one saved analysis. Returns bytes."""

    if not _available():
        raise RuntimeError("reportlab is not installed; PDF export unavailable.")

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage, Paragraph, SimpleDocTemplate, Spacer)

    styles = _styles()
    buffer_path = os.path.join(store.data_dir(), f"_export_{record['id']}.pdf")
    doc = SimpleDocTemplate(
        buffer_path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Texurion report - {record.get('name', 'sample')}")

    story = _title_block(
        "Texurion - Texture Analysis Report",
        f"Sample: <b>{record.get('name', '-')}</b> &nbsp;|&nbsp; "
        f"{record.get('created_at', datetime.now().isoformat(timespec='seconds'))}"
        " &nbsp;|&nbsp; Texture Intelligence Studio",
        styles)

    prediction = record.get("model_prediction") or {}
    rows = [
        ["Analysis ID", record.get("id", "-")],
        ["Workflow", record.get("workflow") or "General"],
        ["Descriptor label", record.get("descriptor_label", "-")],
        ["Roughness (rule-based)", f"{record.get('roughness', 0):.3f}"],
        ["ROI pixels", f"{record.get('roi_pixels', 0):,}"],
        ["Feature vector length", str(record.get("vector_length", 0))],
        ["Predicted class (trained model)", prediction.get("label", "n/a")],
        ["Predicted confidence", (f"{prediction.get('confidence'):.1%}"
                                  if prediction.get("confidence") is not None
                                  else "n/a")],
    ]
    story += [Paragraph("Summary", styles["head"]), _meta_table(rows)]

    cfg = record.get("config", {})
    cfg_rows = [
        ["LBP method", str(cfg.get("method", "-"))],
        ["LBP engine", str(cfg.get("engine", "-"))],
        ["LBP scales", str(cfg.get("lbp_scales", "-"))],
        ["Gabor block", "on" if cfg.get("include_gabor") else "off"],
        ["Opponent-colour block", "on" if cfg.get("include_color_opponent") else "off"],
        ["Working resolution", str(cfg.get("max_side", "-"))],
    ]
    story += [Paragraph("Analysis settings", styles["head"]), _meta_table(cfg_rows)]

    # Visual strip.
    thumbs, labels = [], []
    for key, caption in (("original", "Original"), ("lbp", "LBP map")):
        path = store.asset_path(record, key)
        if path:
            thumbs.append(RLImage(path, width=52 * mm, height=52 * mm * 0.72))
            labels.append(caption)
    if thumbs:
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle

        table = Table([labels, thumbs], colWidths=[56 * mm] * len(thumbs))
        table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(MUTED)),
        ]))
        story += [Paragraph("Visual analysis", styles["head"]), table]

    hist_path = store.asset_path(record, "histogram")
    if hist_path:
        story += [Paragraph("Fused LBP histogram", styles["head"]),
                  RLImage(hist_path, width=150 * mm, height=64 * mm)]

    story += [Paragraph("Texture metrics", styles["head"]),
              _metric_table(record.get("metrics") or {})]

    if record.get("summary"):
        story += [Paragraph("Interpretation", styles["head"]),
                  Paragraph(record["summary"], styles["body"])]

    story += [
        Spacer(1, 6),
        Paragraph("Methodology & limitations", styles["head"]),
        Paragraph(
            "Metrics are computed live from the uploaded image: GLCM statistics "
            "on a 32-level quantised grayscale ROI and histogram statistics on "
            "the fused LBP histogram. Gabor and opponent-colour blocks are "
            "included only when enabled in the settings. Pixel-level texture "
            "indicators are NOT calibrated physical measurements, and the "
            "descriptor label is a rule-based heuristic, not a trained "
            "classifier. Predicted classes appear only when a trained model "
            "was loaded.", styles["small"]),
        Spacer(1, 4),
        Paragraph("Generated by Texurion (Texture Intelligence Studio).",
                  styles["small"]),
    ]

    doc.build(story)
    try:
        with open(buffer_path, "rb") as handle:
            data = handle.read()
    finally:
        try:
            os.remove(buffer_path)
        except OSError:
            pass
    return data


def build_comparison_pdf(result, name_a, name_b):
    """Build a Texurion comparison PDF. Returns bytes."""

    if not _available():
        raise RuntimeError("reportlab is not installed; PDF export unavailable.")

    import tempfile

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage, Paragraph, SimpleDocTemplate, Spacer)

    styles = _styles()
    tmp_pdf = os.path.join(tempfile.gettempdir(), "texurion_compare.pdf")
    doc = SimpleDocTemplate(tmp_pdf, pagesize=A4, leftMargin=18 * mm,
                            rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Texurion comparison")

    story = _title_block(
        "Texurion - Image Comparison Report",
        f"Image A: <b>{name_a}</b> &nbsp;|&nbsp; Image B: <b>{name_b}</b> "
        f"&nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y, %H:%M')}",
        styles)

    rows = [
        ["Pixel similarity (SSIM)", f"{result['ssim']:.4f}"],
        ["Mean absolute pixel difference", f"{result['mae']:.3f} / 255"],
        ["LBP histogram chi-square", f"{result['lbp_chi_square']:.5f}"],
        ["LBP histogram cosine distance", f"{result['lbp_cosine']:.5f}"],
        ["Feature-vector cosine distance", f"{result['feature_cosine']:.5f}"],
        ["Feature similarity (1 - cosine)", f"{result['feature_similarity']:.4f}"],
    ]
    story += [Paragraph("Similarity measures", styles["head"]), _meta_table(rows)]

    story += [Paragraph("Texture metrics", styles["head"]),
              _metric_table(result["deltas"])]

    # Difference heatmap.
    try:
        import cv2
        import tempfile

        diff_path = os.path.join(tempfile.gettempdir(), "texurion_diff.png")
        cv2.imwrite(diff_path, result["diff_image"])
        story += [Paragraph("Difference map (B - A)", styles["head"]),
                  RLImage(diff_path, width=150 * mm, height=80 * mm)]
    except Exception:  # noqa: BLE001
        pass

    story += [
        Spacer(1, 6),
        Paragraph("Interpretation", styles["head"]),
        Paragraph(
            "Pixel similarity (SSIM / MAE) is computed on aligned grayscale "
            "images and measures structure, not texture statistics. "
            "The LBP histogram and feature-vector distances operate on the "
            "fused texture features and are independent of pixel alignment. "
            "These measures are related but not interchangeable.",
            styles["small"]),
        Spacer(1, 4),
        Paragraph("Generated by Texurion (Texture Intelligence Studio).",
                  styles["small"]),
    ]

    doc.build(story)
    with open(tmp_pdf, "rb") as handle:
        data = handle.read()
    try:
        os.remove(tmp_pdf)
    except OSError:
        pass
    return data


def comparison_csv_bytes(result):
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["measure", "value"])
    for key in ("ssim", "mae", "lbp_chi_square", "lbp_cosine",
                "feature_cosine", "feature_similarity"):
        writer.writerow([key, result.get(key)])
    writer.writerow([])
    writer.writerow(["metric", "image_a", "image_b", "delta"])
    for metric in result["metrics_a"]:
        writer.writerow([metric, result["metrics_a"][metric],
                         result["metrics_b"].get(metric, ""),
                         result["deltas"].get(metric, "")])
    return buffer.getvalue().encode("utf-8")

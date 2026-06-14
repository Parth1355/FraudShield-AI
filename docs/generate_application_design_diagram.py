from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1700, 980
BG = "#f7f8fb"
NAVY = "#1f2a44"
TEXT = "#222222"
ARROW = "#333333"
OUTLINE = "#8aa0b8"


def font(name: str, size: int):
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


TITLE = font("arialbd.ttf", 34)
SECTION = font("arialbd.ttf", 22)
LABEL = font("arialbd.ttf", 18)
BODY = font("arial.ttf", 16)


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=OUTLINE, radius=14, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw: ImageDraw.ImageDraw, box, text, ft, fill=TEXT):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=ft, spacing=4)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) / 2
    y = top + (bottom - top - th) / 2
    draw.multiline_text((x, y), text, font=ft, fill=fill, spacing=4, align="center")


def draw_arrow(draw: ImageDraw.ImageDraw, x1, y, x2):
    draw.line((x1, y, x2, y), fill=ARROW, width=3)
    draw.polygon([(x2, y), (x2 - 14, y - 7), (x2 - 14, y + 7)], fill=ARROW)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, ft, max_width: int):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test = word if not line else f"{line} {word}"
        bbox = draw.textbbox((0, 0), test, font=ft)
        if bbox[2] - bbox[0] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return "\n".join(lines)


def draw_streamlit_ui(draw: ImageDraw.ImageDraw, box):
    rounded(draw, box, "#ffffff", "#555555", radius=8, width=2)
    x1, y1, x2, y2 = box
    draw.rectangle((x1 + 10, y1 + 8, x1 + 48, y2 - 8), fill="#edf3ff", outline="#6a88b4")
    for i in range(3):
        yy = y1 + 16 + i * 18
        draw.line((x1 + 16, yy, x1 + 40, yy), fill="#6a88b4", width=2)
    draw.rectangle((x1 + 58, y1 + 12, x2 - 12, y1 + 34), outline="#6a88b4", width=2)
    for i in range(3):
        xx = x1 + 60 + i * 50
        draw.rounded_rectangle((xx, y1 + 44, xx + 38, y1 + 68), radius=5, fill="#dce9ff", outline="#6a88b4")
    draw.text((x1 + 70, y2 - 28), "Dashboard / Prediction UI", font=BODY, fill=TEXT)


def draw_input_icons(draw: ImageDraw.ImageDraw, box):
    x1, y1, x2, y2 = box
    rounded(draw, box, "#ffffff", "#555555", radius=8, width=2)
    draw.rectangle((x1 + 16, y1 + 16, x1 + 84, y2 - 16), outline="#555555", width=2, fill="#fefefe")
    for i in range(3):
        yy = y1 + 28 + i * 14
        draw.line((x1 + 24, yy, x1 + 76, yy), fill="#666666", width=2)
    cx = x1 + 136
    cy = (y1 + y2) // 2 + 6
    draw.rectangle((cx - 26, cy - 22, cx + 26, cy + 18), outline="#555555", width=2, fill="#f7fbff")
    draw.line((cx, cy - 26, cx, cy + 6), fill="#666666", width=2)
    draw.polygon([(cx, cy - 34), (cx - 10, cy - 20), (cx - 4, cy - 20), (cx - 4, cy - 8),
                  (cx + 4, cy - 8), (cx + 4, cy - 20), (cx + 10, cy - 20)], fill="#666666")
    draw.text((x1 + 18, y2 - 30), "CSV or transaction form input", font=BODY, fill=TEXT)


def draw_preprocess(draw: ImageDraw.ImageDraw, box):
    rounded(draw, box, "#ffffff", "#555555", radius=8, width=2)
    x1, y1, x2, y2 = box
    draw.rectangle((x1 + 18, y1 + 16, x1 + 84, y2 - 16), outline="#555555", width=2)
    for i in range(3):
        x = x1 + 34 + i * 14
        draw.line((x, y1 + 24, x, y2 - 24), fill="#999999", width=1)
    for i in range(3):
        y = y1 + 30 + i * 14
        draw.line((x1 + 24, y, x1 + 78, y), fill="#999999", width=1)
    draw_arrow(draw, x1 + 96, (y1 + y2) // 2, x1 + 140)
    draw.rectangle((x1 + 154, y1 + 18, x1 + 228, y2 - 18), outline="#555555", width=2)
    draw.line((x1 + 164, y2 - 24, x1 + 216, y2 - 24), fill="#666666", width=2)
    for i, h in enumerate([18, 32, 22]):
        xx = x1 + 170 + i * 16
        draw.rectangle((xx, y2 - 24 - h, xx + 10, y2 - 24), fill="#6c86b1", outline="#466389")
    draw_arrow(draw, x1 + 240, (y1 + y2) // 2, x1 + 284)
    draw.rectangle((x1 + 298, y1 + 18, x2 - 18, y2 - 18), outline="#555555", width=2)
    draw.text((x1 + 318, y1 + 28), "Clean", font=BODY, fill=TEXT)
    draw.text((x1 + 318, y1 + 50), "Encode", font=BODY, fill=TEXT)
    draw.text((x1 + 392, y1 + 28), "Scale", font=BODY, fill=TEXT)
    draw.text((x1 + 392, y1 + 50), "Prepare", font=BODY, fill=TEXT)


def draw_features(draw: ImageDraw.ImageDraw, box):
    rounded(draw, box, "#ffffff", "#555555", radius=8, width=2)
    x1, y1, x2, y2 = box
    cols = [
        ("Balance\nDiff", "#f4f7ff"),
        ("Tx\nFreq", "#eef8ef"),
        ("Behavior\nFlags", "#fff8e8"),
        ("Ratios", "#f9eef8"),
    ]
    cx = x1 + 18
    for label, fill in cols:
        rounded(draw, (cx, y1 + 16, cx + 90, y2 - 16), fill, "#8a8a8a", radius=8, width=2)
        center_text(draw, (cx + 4, y1 + 20, cx + 86, y2 - 20), label, BODY)
        cx += 102


def draw_model(draw: ImageDraw.ImageDraw, box):
    rounded(draw, box, "#ffffff", "#555555", radius=8, width=2)
    x1, y1, x2, y2 = box
    xs = x1 + 20
    for w, h in [(30, 74), (24, 62), (18, 48), (12, 34), (8, 20)]:
        draw.polygon([(xs, y2 - 18), (xs, y2 - 18 - h), (xs + w, y2 - 26 - h), (xs + w, y2 - 26)],
                     fill="#f2f2f2", outline="#555555")
        xs += w + 12
    draw_arrow(draw, x1 + 150, (y1 + y2) // 2, x1 + 220)
    rounded(draw, (x1 + 238, y1 + 18, x2 - 18, y2 - 18), "#fffdf6", "#8a8a8a", radius=8, width=2)
    center_text(draw, (x1 + 246, y1 + 24, x2 - 26, y2 - 24), "Fraud\nor\nNon-Fraud", LABEL)


def draw_analytics(draw: ImageDraw.ImageDraw, box):
    rounded(draw, box, "#ffffff", "#555555", radius=8, width=2)
    x1, y1, x2, y2 = box
    draw.rectangle((x1 + 18, y1 + 18, x1 + 96, y2 - 18), outline="#555555", width=2)
    for i, h in enumerate([18, 34, 22]):
        xx = x1 + 28 + i * 18
        draw.rectangle((xx, y2 - 24 - h, xx + 10, y2 - 24), fill="#6c86b1", outline="#466389")
    draw_arrow(draw, x1 + 110, (y1 + y2) // 2, x1 + 160)
    draw.arc((x1 + 176, y1 + 18, x1 + 258, y2 - 18), start=180, end=360, fill="#555555", width=3)
    for start, end, color in [(180, 235, "#65b66f"), (235, 300, "#f2b84a"), (300, 360, "#de6a5f")]:
        draw.arc((x1 + 176, y1 + 18, x1 + 258, y2 - 18), start=start, end=end, fill=color, width=7)
    draw.line((x1 + 217, y1 + 58, x1 + 240, y1 + 38), fill="#555555", width=3)
    draw_arrow(draw, x1 + 270, (y1 + y2) // 2, x1 + 322)
    rounded(draw, (x1 + 336, y1 + 18, x2 - 18, y2 - 18), "#fefefe", "#8a8a8a", radius=8, width=2)
    center_text(draw, (x1 + 342, y1 + 24, x2 - 24, y2 - 24), "Dashboard +\nModel Insights", LABEL)


def draw_reporting(draw: ImageDraw.ImageDraw, box):
    rounded(draw, box, "#ffffff", "#555555", radius=8, width=2)
    x1, y1, x2, y2 = box
    draw.rectangle((x1 + 30, y1 + 16, x1 + 100, y2 - 16), outline="#555555", width=2, fill="#fefefe")
    draw.line((x1 + 44, y1 + 34, x1 + 84, y1 + 34), fill="#666666", width=2)
    draw.line((x1 + 44, y1 + 50, x1 + 88, y1 + 50), fill="#666666", width=2)
    draw.line((x1 + 44, y1 + 66, x1 + 80, y1 + 66), fill="#666666", width=2)
    draw_arrow(draw, x1 + 116, (y1 + y2) // 2, x1 + 180)
    draw.rectangle((x1 + 194, y1 + 20, x1 + 280, y2 - 20), outline="#555555", width=2)
    draw.text((x1 + 208, y1 + 30), "CSV", font=SECTION, fill=NAVY)
    draw.text((x1 + 205, y1 + 58), "Export", font=BODY, fill=TEXT)
    draw_arrow(draw, x1 + 296, (y1 + y2) // 2, x1 + 360)
    rounded(draw, (x1 + 374, y1 + 18, x2 - 18, y2 - 18), "#fefefe", "#8a8a8a", radius=8, width=2)
    center_text(draw, (x1 + 382, y1 + 24, x2 - 24, y2 - 24), "Fraud Results\nSummary Report", LABEL)


def main():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.text((WIDTH // 2 - 195, 24), "APPLICATION DESIGN", font=TITLE, fill=NAVY)
    draw.text((80, 86), "2.1 Application Design :-", font=SECTION, fill=NAVY)

    left_x = 80
    layer_w = 980
    right_x = 1110
    top = 130
    h = 92
    gap = 22

    layers = [
        ("Presentation Layer", "#dce9ff",
         "Handles user interaction through the Streamlit web interface.",
         draw_streamlit_ui),
        ("Data Input Layer", "#e6f5dc",
         "Accepts transaction datasets or user-provided transaction details.",
         draw_input_icons),
        ("Preprocessing Layer", "#fff1cc",
         "Performs data cleaning, encoding, scaling, and feature preparation.",
         draw_preprocess),
        ("Feature Engineering Layer", "#e8e1ff",
         "Creates meaningful features such as balance differences, transaction frequency, and behavioral indicators.",
         draw_features),
        ("Model Layer (LightGBM)", "#ffdfe5",
         "Predicts fraudulent and non-fraudulent transactions.",
         draw_model),
        ("Analytics and Interpretation Layer", "#dff4fb",
         "Displays dashboards, fraud probabilities, and model insights.",
         draw_analytics),
        ("Reporting Layer", "#e7f1ff",
         "Exports filtered fraud results and summaries for review.",
         draw_reporting),
    ]

    for idx, (label, fill, desc, renderer) in enumerate(layers):
        y1 = top + idx * (h + gap)
        y2 = y1 + h
        rounded(draw, (left_x, y1, left_x + layer_w, y2), fill, radius=10)
        center_text(draw, (left_x + 16, y1 + 8, left_x + 240, y2 - 8), label, LABEL)

        inner = (left_x + 258, y1 + 10, left_x + layer_w - 24, y2 - 10)
        renderer(draw, inner)

        draw.line((left_x + layer_w // 2, y2, left_x + layer_w // 2, y2 + gap - 8), fill=ARROW, width=2)
        if idx < len(layers) - 1:
            draw.polygon([
                (left_x + layer_w // 2, y2 + gap - 2),
                (left_x + layer_w // 2 - 6, y2 + gap - 14),
                (left_x + layer_w // 2 + 6, y2 + gap - 14),
            ], fill=ARROW)

        desc_text = wrap_text(draw, desc, LABEL, 520)
        draw.line((left_x + layer_w + 22, (y1 + y2) // 2, right_x, (y1 + y2) // 2), fill=ARROW, width=2)
        for x in range(left_x + layer_w + 22, right_x, 16):
            draw.line((x, (y1 + y2) // 2, min(x + 8, right_x), (y1 + y2) // 2), fill=ARROW, width=2)
        draw.multiline_text((right_x + 24, y1 + 18), desc_text, font=LABEL, fill=TEXT, spacing=6)

    out = Path("C:/Users/parth/Downloads/files/docs/fraudshield_application_design_layered.png")
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()

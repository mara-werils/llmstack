"""Generate social preview image for GitHub (1280x640)."""

from PIL import Image, ImageDraw, ImageFont
import os

WIDTH, HEIGHT = 1280, 640
BG = (15, 17, 23)  # GitHub dark bg
ACCENT = (88, 166, 255)  # Blue accent
WHITE = (230, 237, 243)
GRAY = (125, 133, 144)
GREEN = (63, 185, 80)

img = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(img)

# Try to load a nice font, fall back to default
def get_font(size: int):
    paths = [
        "/System/Library/Fonts/SFMono-Bold.otf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/SF-Mono-Bold.otf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

def get_font_regular(size: int):
    paths = [
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

font_title = get_font(72)
font_sub = get_font_regular(28)
font_tag = get_font_regular(22)
font_cmd = get_font_regular(26)

# Draw subtle grid pattern
for x in range(0, WIDTH, 40):
    draw.line([(x, 0), (x, HEIGHT)], fill=(25, 27, 33), width=1)
for y in range(0, HEIGHT, 40):
    draw.line([(0, y), (WIDTH, y)], fill=(25, 27, 33), width=1)

# Accent bar at top
draw.rectangle([(0, 0), (WIDTH, 4)], fill=ACCENT)

# Title: llmstack
title = "llmstack"
bbox = draw.textbbox((0, 0), title, font=font_title)
tw = bbox[2] - bbox[0]
draw.text(((WIDTH - tw) // 2, 120), title, fill=WHITE, font=font_title)

# Tagline
tagline = "One command. Full LLM stack. Zero config."
bbox = draw.textbbox((0, 0), tagline, font=font_sub)
tw = bbox[2] - bbox[0]
draw.text(((WIDTH - tw) // 2, 210), tagline, fill=GRAY, font=font_sub)

# Terminal-style command box
box_w, box_h = 520, 60
box_x = (WIDTH - box_w) // 2
box_y = 290
# Rounded rectangle background
draw.rounded_rectangle(
    [(box_x, box_y), (box_x + box_w, box_y + box_h)],
    radius=12,
    fill=(30, 33, 40),
    outline=(50, 54, 62),
    width=2,
)
# Terminal prompt
cmd = "$ pip install llmstack-cli"
bbox = draw.textbbox((0, 0), cmd, font=font_cmd)
cw = bbox[2] - bbox[0]
draw.text(((WIDTH - cw) // 2, box_y + 15), "$ ", fill=GREEN, font=font_cmd)
# Command text after prompt
draw.text(((WIDTH - cw) // 2 + 28, box_y + 15), "pip install llmstack-cli", fill=WHITE, font=font_cmd)

# Feature pills
features = ["Ollama / vLLM", "Qdrant", "Redis", "FastAPI Gateway", "Grafana"]
pill_y = 400
total_w = 0
pill_sizes = []
for f in features:
    bbox = draw.textbbox((0, 0), f, font=font_tag)
    pw = bbox[2] - bbox[0] + 24
    pill_sizes.append(pw)
    total_w += pw
gap = 16
total_w += gap * (len(features) - 1)
start_x = (WIDTH - total_w) // 2

for i, f in enumerate(features):
    pw = pill_sizes[i]
    draw.rounded_rectangle(
        [(start_x, pill_y), (start_x + pw, pill_y + 36)],
        radius=18,
        fill=(30, 33, 40),
        outline=(50, 54, 62),
        width=1,
    )
    bbox = draw.textbbox((0, 0), f, font=font_tag)
    fw = bbox[2] - bbox[0]
    draw.text((start_x + (pw - fw) // 2, pill_y + 6), f, fill=ACCENT, font=font_tag)
    start_x += pw + gap

# Bottom: "OpenAI-compatible API • Auto hardware detection • Plugin ecosystem"
bottom = "OpenAI-compatible API  •  Auto hardware detection  •  Plugin ecosystem"
bbox = draw.textbbox((0, 0), bottom, font=font_tag)
bw = bbox[2] - bbox[0]
draw.text(((WIDTH - bw) // 2, 500), bottom, fill=GRAY, font=font_tag)

# GitHub handle
handle = "github.com/mara-werils/llmstack"
bbox = draw.textbbox((0, 0), handle, font=font_tag)
hw = bbox[2] - bbox[0]
draw.text(((WIDTH - hw) // 2, 570), handle, fill=(75, 80, 90), font=font_tag)

out = "assets/social-preview.png"
os.makedirs("assets", exist_ok=True)
img.save(out, "PNG")
print(f"Saved {out} ({os.path.getsize(out) // 1024} KB)")

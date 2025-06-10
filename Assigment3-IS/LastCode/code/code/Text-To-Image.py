import os
import random

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import arabic_reshaper
from bidi.algorithm import get_display

# ===== Configuration =====
EXCEL_PATH    = r"G:\Group Project\PY CODE\FINALISE CODE\data\Dataset-Verse-by-Verse.xlsx"
FONT_PATH     = r"G:\Group Project\PY CODE\FINALISE CODE\data\Amiri-Regular.ttf"
BG_FOLDER     = r"G:\Group Project\PY CODE\FINALISE CODE\data\backgrounds"
OUTPUT_FOLDER = r"G:\Group Project\PY CODE\FINALISE CODE\data\quran_images"
LABEL_CSV     = r"G:\Group Project\PY CODE\FINALISE CODE\data\quran_labels.csv"

# How many synthetic variants per verse?
VARIANTS_PER_VERSE = 5

# Image size you want
IMG_W, IMG_H = 800, 200

# Create output dir
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load background paths
bg_paths = [
    os.path.join(BG_FOLDER, fn)
    for fn in os.listdir(BG_FOLDER)
    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
]
if not bg_paths:
    raise RuntimeError(f"No images found in background folder: {BG_FOLDER}")

# Load dataset
df = pd.read_excel(EXCEL_PATH)
records = []

def augment(img: Image.Image) -> Image.Image:
    """Apply random blur and brightness/contrast jitter."""
    # slight blur
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    # brightness/contrast
    arr = np.array(img).astype(np.float32)
    b = random.uniform(0.8, 1.2)  # brightness factor
    c = random.uniform(0.8, 1.2)  # contrast factor
    arr = (arr - 127.5 * c) * b + 127.5 * c
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# Loop over each verse
for idx, row in df.iterrows():
    surah_no       = int(row['SurahNo'])
    ayah_no        = int(row['AyahNo'])
    surah_name     = row['SurahNameEnglish']
    juz            = int(row['Juz'])
    classification = row['Classification']
    arabic_text    = str(row['OrignalArabicText'])

    # reshape + bidi for proper Arabic rendering
    reshaped     = arabic_reshaper.reshape(arabic_text)
    display_text = get_display(reshaped)

    for v in range(1, VARIANTS_PER_VERSE + 1):
        # Pick a random background and resize
        bg_path = random.choice(bg_paths)
        bg = Image.open(bg_path).convert("RGB")
        bg = bg.resize((IMG_W, IMG_H), Image.LANCZOS)

        draw = ImageDraw.Draw(bg)

        # Random font size & color
        font_size = random.randint(24, 48)
        font = ImageFont.truetype(FONT_PATH, font_size)
        base_color = 20
        color = (
            random.randint(0, base_color),
            random.randint(0, base_color),
            random.randint(0, base_color)
        )

        # Measure text size with textbbox
        bbox = draw.textbbox((0, 0), display_text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # If text is too wide, scale font down to fit
        if w > IMG_W:
            scale = IMG_W / w
            font_size = max(12, int(font_size * scale))
            font = ImageFont.truetype(FONT_PATH, font_size)
            bbox = draw.textbbox((0, 0), display_text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

        # Compute clamped random position
        max_dx = max(0, IMG_W - w)
        min_x = int(0.1 * max_dx)
        max_x = int(0.9 * max_dx)
        x = random.randint(min_x, max_x)

        max_dy = max(0, IMG_H - h)
        min_y = int(0.1 * max_dy)
        max_y = int(0.9 * max_dy)
        y = random.randint(min_y, max_y)

        # Draw the text
        draw.text((x, y), display_text, font=font, fill=color)

        # Optional slight rotation
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            bg = bg.rotate(angle, expand=True, fillcolor=(255, 255, 255))
            bg = bg.crop((0, 0, IMG_W, IMG_H))

        # Final augmentation pass
        img_final = augment(bg)

        # Save to disk
        filename = f"{surah_no:03d}_{ayah_no:03d}_v{v}.png"
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        img_final.save(out_path, quality=90)

        # Record label entry
        records.append({
            "filename":      filename,
            "surah_no":      surah_no,
            "ayah_no":       ayah_no,
            "surah_name":    surah_name,
            "juz":           juz,
            "classification": classification,
            "arabic_text":   arabic_text
        })

# Write out CSV
pd.DataFrame(records).to_csv(LABEL_CSV, index=False, encoding="utf-8-sig")

print(f"✅ Generated {len(records)} images (~{VARIANTS_PER_VERSE}× original).")
print(f"→ Images saved to: {OUTPUT_FOLDER}")
print(f"→ Label CSV saved to: {LABEL_CSV}")

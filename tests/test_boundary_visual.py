import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE = Path("output/documents/doc_000001")

PAGE = BASE / "pages/page_001.png"
PAGE_JSON = BASE / "page_json/page_001.json"
GEMINI_JSON = BASE / "gemini/page_001_response.json"

OUTPUT = BASE / "boundary_preview_page1.png"


# =========================================================
# LOAD PAGE
# =========================================================

if not PAGE.exists():
    raise FileNotFoundError(
        f"Page image not found:\n{PAGE}"
    )

img = Image.open(PAGE).convert("RGB")

draw = ImageDraw.Draw(img)


# =========================================================
# LOAD PAGE JSON
# =========================================================

with open(PAGE_JSON, encoding="utf-8") as f:
    page_data = json.load(f)

page_blocks = page_data.get("blocks", [])

block_lookup = {}

for block in page_blocks:

    block_id = block.get("id")

    if block_id is None:
        continue

    block_lookup[int(block_id)] = block


# =========================================================
# LOAD GEMINI RESPONSE
# =========================================================

with open(GEMINI_JSON, encoding="utf-8") as f:
    gemini_data = json.load(f)

gemini_articles = gemini_data.get(
    "articles",
    []
)


# =========================================================
# FONT
# =========================================================

try:

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf",
        32,
    )

except Exception:

    font = None


# =========================================================
# DRAW ARTICLES
# =========================================================

drawn = 0


for index, article in enumerate(
    gemini_articles,
    start=1,
):

    block_ids = article.get(
        "blocks",
        []
    )

    valid_blocks = []

    for block_id in block_ids:

        block = block_lookup.get(
            int(block_id)
        )

        if block is None:
            continue

        bbox = block.get(
            "bbox",
            {}
        )

        if not bbox:
            continue

        valid_blocks.append(
            bbox
        )

    if not valid_blocks:
        continue


    # -----------------------------------------------------
    # Outer boundary
    # -----------------------------------------------------

    x1 = min(
        int(b["x1"])
        for b in valid_blocks
    )

    y1 = min(
        int(b["y1"])
        for b in valid_blocks
    )

    x2 = max(
        int(b["x2"])
        for b in valid_blocks
    )

    y2 = max(
        int(b["y2"])
        for b in valid_blocks
    )


    # -----------------------------------------------------
    # Draw boundary
    # -----------------------------------------------------

    draw.rectangle(
        (x1, y1, x2, y2),
        outline="red",
        width=8,
    )


    # -----------------------------------------------------
    # Article label
    # -----------------------------------------------------

    label = str(index)

    label_x1 = x1
    label_y1 = max(
        0,
        y1 - 45
    )

    label_x2 = x1 + 65
    label_y2 = y1

    draw.rectangle(
        (
            label_x1,
            label_y1,
            label_x2,
            label_y2,
        ),
        fill="red",
    )

    draw.text(
        (
            label_x1 + 18,
            label_y1 + 5,
        ),
        label,
        fill="white",
        font=font,
    )

    drawn += 1


# =========================================================
# SAVE
# =========================================================

img.save(
    OUTPUT
)

print()
print("=" * 60)
print("BOUNDARY VISUAL TEST")
print("=" * 60)

print(
    f"Gemini articles : "
    f"{len(gemini_articles)}"
)

print(
    f"Boundaries drawn : "
    f"{drawn}"
)

print()
print(
    f"Saved:"
)
print(OUTPUT)

print("=" * 60)
print()
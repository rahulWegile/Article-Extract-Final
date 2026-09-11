import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from google import genai
from google.genai import types
from google.genai.errors import APIError

from pipeline.ocr.utrnet_engine import UTRNetRecognizer, MODEL_PATH, VOCAB_PATH
from pipeline.languages.urdu import URDU
from pipeline.finalization.logical_article_builder import LogicalArticleBuilder
from pipeline.database.import_final_articles import import_document_directory
from pipeline.openai.openai_service import OpenAIService


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETECTOR_PATH = PROJECT_ROOT / "models" / "urdu_text_detection" / "yolov8m_UrduDoc.pt"

DETECT_CONF = 0.15
DETECT_IMGSZ = 1920
DETECT_MAX_DET = 1000

CROP_PADDING = 20
OCR_BATCH_SIZE = 14
OPENAI_IMAGE_MAX_DIM = 1600

ORPHAN_MIN_GAP = -20
ORPHAN_MAX_GAP = 40
ORPHAN_MIN_COL_OVERLAP = 0.45

LEFTOVER_MIN_GAP = -25
LEFTOVER_MAX_GAP = 60
LEFTOVER_MIN_OVERLAP = 0.35

RULE_DARK_THRESHOLD = 140
RULE_DARK_ROW_RATIO = 0.35

ARTICLE_CROP_PADDING = 10


def _has_rule_separator(gray, y_top, y_bot, x_l, x_r):

    y_s = max(0, int(min(y_top, y_bot)))
    y_e = min(gray.shape[0], int(max(y_top, y_bot)))
    x_s = max(0, int(x_l))
    x_e = min(gray.shape[1], int(x_r))

    if y_e - y_s < 2 or x_e <= x_s:
        return False

    strip = gray[y_s:y_e, x_s:x_e]
    dark_counts = (strip < RULE_DARK_THRESHOLD).sum(axis=1)

    return bool((dark_counts > (x_e - x_s) * RULE_DARK_ROW_RATIO).any())


def _build_grouping_prompt(regions, page_number, page_width, page_height):

    ocr_text = json.dumps(
        [
            {
                "id": r["id"],
                "bbox": [round(v, 1) for v in r["bbox"]],
                "text": r["text"],
            }
            for r in regions
        ],
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are an expert Urdu newspaper layout understanding system.

Your ONLY task is to identify which OCR regions belong to the SAME
INDIVIDUAL NEWSPAPER ARTICLE. Do NOT return bounding box coordinates
-- Python computes the final pixel boundary from the OCR region ids
you assign to each article.

You are given the ORIGINAL page image (primary source of truth) and
UTRNet OCR regions with their id, bbox, and recognized text.

RULES:
- One article = headline + body + related images/captions across
  one or more columns. Do NOT split a multi-column article.
- Do not group regions merely because they are close together --
  use headline typography, column structure, separators, whitespace,
  and semantic continuity of the OCR text.
- Advertisements, the masthead, page furniture (date, page number,
  price, section labels) are NOT articles -- omit their OCR ids
  entirely rather than inventing an article for them.
- A photograph alone is not an article; keep its OCR/caption ids
  with the story it illustrates.
- Every OCR id you use must exist in the supplied data. Never invent
  ids. An OCR id normally belongs to at most one article.
- Do not omit the last line/region of an article -- Python uses the
  lowest region's coordinate as the bottom of the boundary.
- Do not try to maximize or minimize the article count: return
  exactly the number of real, independent editorial stories visible
  on the page.

Page: {page_number}, width={page_width}, height={page_height}

OCR REGIONS:
{ocr_text}
"""


_GROUPING_SCHEMA = {
    "type": "object",
    "properties": {
        "articles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string"},
                    "ocr_region_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["article_id", "ocr_region_ids"],
            },
        },
    },
    "required": ["articles"],
}

_OPENAI_GROUPING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "articles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "article_id": {"type": "string"},
                    "ocr_region_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["article_id", "ocr_region_ids"],
            },
        },
    },
    "required": ["articles"],
}


def _encode_jpeg_data_url(image_path, max_dim=OPENAI_IMAGE_MAX_DIM, quality=80):

    with Image.open(image_path) as image:
        img = image.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


class UrduPipelineService:
    """
    Dedicated Urdu (Nastaliq) pipeline: yolov8m_UrduDoc.pt detects raw
    text regions directly (no DocLayout-YOLO, no PageCleaner, no
    sidebox_absorption/article_splitter), UTRNet reads them, and a
    single boundary-grouping call (OpenAI by default, matching every
    other language's boundary provider; Gemini when LLM_PROVIDER=
    gemini or no OPENAI_API_KEY is configured) groups OCR region ids
    into articles -- the LLM never sees or returns pixel coordinates,
    Python computes the boundary as the union of each article's
    assigned OCR regions. Article-level extraction still uses Gemini
    (URDU.extractor_class) for its stronger Nastaliq comprehension.

    Avoids the DocLayout-YOLO full-page "figure" hallucination this
    project's shared boundary pipeline has repeatedly hit on Urdu
    broadsheets, and the block-absorption passes that cascade from it.
    """

    def __init__(self):

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Urdu page-level detector (device={self.device})...")
        self.detector = YOLO(str(DETECTOR_PATH))

        print(f"Loading UTRNet OCR (device={self.device})...")
        self.ocr_engine = UTRNetRecognizer(
            MODEL_PATH, VOCAB_PATH, device=self.device,
        )

        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.genai_model = os.getenv("GEMINI_BOUNDARY_MODEL", "gemini-3.6-flash")

        self.openai_service = None

        if (
            os.getenv("LLM_PROVIDER", "").strip().lower() != "gemini"
            and os.getenv("OPENAI_API_KEY")
        ):
            self.openai_service = OpenAIService(
                model=os.getenv("OPENAI_BOUNDARY_MODEL", "gpt-4o-mini"),
            )

        self.boundary_llm_provider = "openai" if self.openai_service else "gemini"

    # ========================================================
    # DETECTION + OCR
    # ========================================================

    def _detect_and_read(self, page_image, gray_image, page_number):

        width, height = page_image.size

        results = self.detector.predict(
            source=page_image,
            conf=DETECT_CONF,
            imgsz=DETECT_IMGSZ,
            max_det=DETECT_MAX_DET,
            device=self.device,
            verbose=False,
        )

        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        boxes.sort(key=lambda b: b[1])

        regions = [
            {
                "id": f"ocr_{page_number:03d}_{index:04d}",
                "bbox": [float(v) for v in box],
            }
            for index, box in enumerate(boxes)
        ]

        crops = []

        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            crop = gray_image.crop(
                (
                    max(0, x1 - CROP_PADDING),
                    max(0, y1 - CROP_PADDING),
                    min(width, x2 + CROP_PADDING),
                    min(height, y2 + CROP_PADDING),
                )
            )
            crops.append(cv2.cvtColor(np.array(crop), cv2.COLOR_GRAY2BGR))

        texts = []

        for start in range(0, len(crops), OCR_BATCH_SIZE):
            batch = crops[start:start + OCR_BATCH_SIZE]
            texts.extend(self.ocr_engine.recognize_batch(batch))

        for region, (text, confidence) in zip(regions, texts):
            region["text"] = text.strip()
            region["confidence"] = confidence

        return regions

    # ========================================================
    # BOUNDARY GROUPING (single call, OCR ids only)
    # ========================================================

    def _group_regions(self, page_image_path, regions, page_number, width, height):

        if not regions:
            return {"articles": []}

        prompt = _build_grouping_prompt(regions, page_number, width, height)

        if self.boundary_llm_provider == "openai":
            return self._group_regions_openai(page_image_path, prompt)

        return self._group_regions_gemini(page_image_path, prompt)

    def _group_regions_openai(self, page_image_path, prompt):

        contents = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": _encode_jpeg_data_url(page_image_path),
                    "detail": "high",
                },
            },
        ]

        return self.openai_service._call(
            contents,
            schema_name="urdu_article_grouping",
            schema=_OPENAI_GROUPING_SCHEMA,
        )

    def _group_regions_gemini(self, page_image_path, prompt):

        with open(page_image_path, "rb") as f:
            image_bytes = f.read()

        contents = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ]

        response = self.genai_client.models.generate_content(
            model=self.genai_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=_GROUPING_SCHEMA,
            ),
        )

        return json.loads((response.text or "{}").strip() or "{}")

    # ========================================================
    # ORPHAN ABSORPTION + LEFTOVER CLUSTERING
    # ========================================================

    def _absorb_and_cluster(self, gray, regions, grouped):

        articles = grouped.setdefault("articles", [])
        ocr_map = {r["id"]: r for r in regions}

        assigned_ids = set()
        for article in articles:
            assigned_ids.update(article.get("ocr_region_ids", []))

        absorbed_any = True

        while absorbed_any:

            absorbed_any = False
            unassigned = [r for r in regions if r["id"] not in assigned_ids]

            if not unassigned:
                break

            for article in articles:

                art_blocks = [
                    ocr_map[rid]
                    for rid in article.get("ocr_region_ids", [])
                    if rid in ocr_map
                ]

                if not art_blocks:
                    continue

                for u in list(unassigned):

                    u_x1, u_y1, u_x2, u_y2 = u["bbox"]
                    u_w = max(1.0, u_x2 - u_x1)

                    col_blocks = [
                        b for b in art_blocks
                        if (min(b["bbox"][2], u_x2) - max(b["bbox"][0], u_x1)) / u_w
                        > ORPHAN_MIN_COL_OVERLAP
                    ]

                    if not col_blocks:
                        continue

                    lowest_col_y2 = max(b["bbox"][3] for b in col_blocks)
                    gap = u_y1 - lowest_col_y2

                    if (
                        ORPHAN_MIN_GAP <= gap <= ORPHAN_MAX_GAP
                        and not _has_rule_separator(
                            gray, lowest_col_y2, u_y1, u_x1, u_x2,
                        )
                    ):
                        article.setdefault("ocr_region_ids", []).append(u["id"])
                        assigned_ids.add(u["id"])
                        art_blocks.append(u)
                        unassigned.remove(u)
                        absorbed_any = True

        leftover = [r for r in regions if r["id"] not in assigned_ids]

        if leftover:

            leftover.sort(key=lambda r: r["bbox"][1])
            clusters = []

            for u in leftover:

                placed = False

                for cluster in clusters:

                    cl_x1 = min(b["bbox"][0] for b in cluster)
                    cl_x2 = max(b["bbox"][2] for b in cluster)
                    cl_y2 = max(b["bbox"][3] for b in cluster)

                    u_x1, u_y1, u_x2, u_y2 = u["bbox"]
                    u_w = max(1.0, u_x2 - u_x1)

                    h_overlap = (min(cl_x2, u_x2) - max(cl_x1, u_x1)) / u_w
                    gap = u_y1 - cl_y2

                    if (
                        h_overlap > LEFTOVER_MIN_OVERLAP
                        and LEFTOVER_MIN_GAP <= gap <= LEFTOVER_MAX_GAP
                        and not _has_rule_separator(gray, cl_y2, u_y1, u_x1, u_x2)
                    ):
                        cluster.append(u)
                        placed = True
                        break

                if not placed:
                    clusters.append([u])

            for cluster in clusters:
                if len(cluster) >= 2:
                    articles.append(
                        {
                            "article_id": f"leftover_{len(articles) + 1}",
                            "ocr_region_ids": [b["id"] for b in cluster],
                        }
                    )

        return grouped

    # ========================================================
    # CROP + SAVE
    # ========================================================

    def _save_page(self, document_dir, page_number, page_image, regions, grouped):

        width, height = page_image.size
        ocr_map = {r["id"]: r for r in regions}

        crop_root = document_dir / "final_articles_crops" / f"page_{page_number:03d}"
        crop_root.mkdir(parents=True, exist_ok=True)

        boundary_image = page_image.copy()
        draw = ImageDraw.Draw(boundary_image)

        saved = []
        index = 0

        for article in grouped.get("articles", []):

            region_ids = [
                rid for rid in article.get("ocr_region_ids", [])
                if rid in ocr_map
            ]

            if not region_ids:
                continue

            boxes = [ocr_map[rid]["bbox"] for rid in region_ids]

            x1 = max(0, min(b[0] for b in boxes) - ARTICLE_CROP_PADDING)
            y1 = max(0, min(b[1] for b in boxes) - ARTICLE_CROP_PADDING)
            x2 = min(width, max(b[2] for b in boxes) + ARTICLE_CROP_PADDING)
            y2 = min(height, max(b[3] for b in boxes) + ARTICLE_CROP_PADDING)

            if x2 <= x1 or y2 <= y1:
                continue

            index += 1
            article_id = f"article_{index:03d}"

            article_dir = crop_root / article_id
            article_dir.mkdir(parents=True, exist_ok=True)

            page_image.crop((x1, y1, x2, y2)).save(
                article_dir / f"page_{page_number:03d}.png"
            )

            combined_text = "\n".join(
                ocr_map[rid]["text"] for rid in region_ids if ocr_map[rid]["text"]
            )

            with open(article_dir / "crop.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "article_id": article_id,
                        "page": page_number,
                        "bbox": {
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                        },
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "block_ids": region_ids,
                        "text": combined_text,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            draw.rectangle([x1, y1, x2, y2], outline="lime", width=6)

            saved.append({"page": page_number, "article_id": article_id})

        final_dir = document_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        boundary_image.save(final_dir / f"page_{page_number:03d}_final_boundaries.png")

        return saved

    # ========================================================
    # PROCESS ONE PAGE (split so the boundary-grouping network
    # call for page N can run in the background while the GPU
    # moves straight on to detection + OCR for page N+1)
    # ========================================================

    def _prepare_page(self, page_number, page_path):

        page_image = Image.open(page_path).convert("RGB")
        gray_image = page_image.convert("L")
        width, height = page_image.size

        regions = self._detect_and_read(page_image, gray_image, page_number)

        return page_image, gray_image, regions, width, height

    def _finish_page(self, document_dir, page_number, page_image, gray_image, regions, grouped):

        gray = np.array(gray_image)
        grouped = self._absorb_and_cluster(gray, regions, grouped)

        return self._save_page(document_dir, page_number, page_image, regions, grouped)

    # ========================================================
    # PROCESS DOCUMENT (pages + document dir already prepared by
    # PipelineService -- metadata/newspaper name/date already resolved)
    # ========================================================

    def process_document(self, document_dir, pages, metadata, progress_callback=None):

        def _report(stage, percent, **extra):
            if progress_callback:
                try:
                    progress_callback(stage=stage, progress=percent, **extra)
                except Exception:
                    pass

        document_dir = Path(document_dir)
        all_crops = []

        with ThreadPoolExecutor(max_workers=2) as grouping_executor:

            futures = []

            for index, page_path in enumerate(pages, start=1):

                page_image, gray_image, regions, width, height = self._prepare_page(
                    index, page_path,
                )

                _report(
                    f"Urdu page {index}/{len(pages)}: OCR complete, grouping layout",
                    10 + round(35 * index / len(pages)),
                )

                future = grouping_executor.submit(
                    self._group_regions, page_path, regions, index, width, height,
                )

                futures.append((index, page_path, regions, future))
                del page_image, gray_image

            for index, page_path, regions, future in futures:

                grouped = future.result()

                with Image.open(page_path) as page_img:
                    page_image = page_img.convert("RGB")
                    gray_image = page_image.convert("L")

                    page_crops = self._finish_page(
                        document_dir, index, page_image, gray_image, regions, grouped,
                    )

                all_crops.extend(page_crops)

                print(f"Page {index}: {len(page_crops)} article(s)")

                _report(
                    f"Urdu page {index}/{len(pages)}: crops finalized",
                    45 + round(25 * index / len(pages)),
                )

        if not all_crops:
            raise RuntimeError("No final article crops were created (Urdu pipeline).")

        # Mark boundaries_ready immediately so the document is openable in the frontend
        # and boundaries are visible even before or during Gemini extraction!
        doc_json_path = document_dir / "document.json"
        if doc_json_path.is_file():
            try:
                with open(doc_json_path, "r", encoding="utf-8") as f:
                    doc_meta = json.load(f)
                doc_meta["boundaries_ready"] = True
                doc_meta["article_count"] = len(all_crops)
                doc_meta["final_article_count"] = len(all_crops)
                with open(doc_json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_meta, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: could not update document.json with boundaries_ready: {e}")

        _report(
            "Article boundaries created",
            70,
            event="boundaries_created",
            article_count=len(all_crops),
        )

        _report("Article-level extraction (Gemini)", 72)

        gemini_success = False
        try:
            extractor = URDU.extractor_class(
                pages_per_batch=3,
                prompt_template=URDU.extraction_prompt_template,
            )
            extractor.process_document(document_dir)
            gemini_success = True
        except Exception as exc:
            print(f"\n⚠ Gemini extraction failed or quota exceeded ({exc}).")
            print("Falling back to UTRNet OCR texts from crops to assemble logical articles...\n")
            self._build_fallback_logical_articles(document_dir, all_crops)

        _report("Building logical articles", 90)

        LogicalArticleBuilder(document_dir).build_all()

        _report("Importing into database", 97)

        import_document_directory(document_dir)

        if not gemini_success and doc_json_path.is_file():
            try:
                with open(doc_json_path, "r", encoding="utf-8") as f:
                    doc_meta = json.load(f)
                doc_meta["status"] = "completed"
                doc_meta["extraction_note"] = "Extracted via UTRNet OCR fallback (Gemini batch quota exceeded)."
                with open(doc_json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_meta, f, indent=4, ensure_ascii=False)
            except Exception:
                pass

        _report("Completed", 100)

        return {"final_article_crops": all_crops}

    def _build_fallback_logical_articles(self, document_dir, all_crops):
        crop_root = document_dir / "final_articles_crops"
        batch_dir = document_dir / "gemini_article_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)

        articles = []
        logical_articles = []
        counter = 1

        for page_dir in sorted(crop_root.glob("page_*")):
            if not page_dir.is_dir():
                continue
            try:
                page_num = int(page_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            for art_dir in sorted(page_dir.glob("article_*")):
                if not art_dir.is_dir():
                    continue
                crop_json_path = art_dir / "crop.json"
                if not crop_json_path.exists():
                    continue
                try:
                    with open(crop_json_path, "r", encoding="utf-8") as f:
                        cdata = json.load(f)
                except Exception:
                    continue

                art_id = cdata.get("article_id") or art_dir.name
                text = (cdata.get("text") or "").strip()
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                headline = lines[0] if lines else f"Article {art_id}"
                crop_png = art_dir / f"page_{page_num:03d}.png"

                art_obj = {
                    "page": page_num,
                    "article_id": art_id,
                    "headline": headline,
                    "subheadline": None,
                    "author": None,
                    "location": None,
                    "date": None,
                    "article_text": text,
                    "summary": None,
                    "category": "News",
                    "entities": [],
                    "topics": [],
                    "keywords": [],
                    "sentiment": "neutral",
                    "content_type": "article",
                    "continuation": {
                        "is_continued": False,
                        "marker": None,
                        "next_page": None,
                    },
                    "images": {
                        "has_images": False,
                        "image_count": 0,
                        "items": [],
                    },
                    "crop_path": str(crop_png) if crop_png.exists() else None,
                    "bbox": cdata.get("bbox"),
                }
                articles.append(art_obj)

                logical_id = f"logical_{counter:04d}"
                counter += 1

                logical_articles.append({
                    "logical_article_id": logical_id,
                    "status": "completed",
                    "source_parts": [{"page": page_num, "article_id": art_id}],
                    "continuation_links": [],
                    "pending_continuations": [],
                    "articles": [art_obj],
                })

        final_logical = {
            "schema_version": "gemini_article_pipeline_v6_text_repair",
            "document_dir": str(document_dir),
            "model": "utrnet_ocr_fallback",
            "pages_per_batch": 3,
            "article_count": len(articles),
            "continuation_link_count": 0,
            "pending_count": 0,
            "logical_article_count": len(logical_articles),
            "articles": articles,
            "continuation_links": [],
            "pending_continuations": [],
            "logical_articles": logical_articles,
        }

        with open(batch_dir / "final_logical_articles.json", "w", encoding="utf-8") as f:
            json.dump(final_logical, f, indent=2, ensure_ascii=False)

# English Newspaper Extraction Pipeline Documentation

## 1. Overview & Architecture

The **English Pipeline** is the reference document extraction pipeline in Newspaper Archive Studio. It transforms scanned multi-page English broadsheet and tabloid newspapers into structured, searchable digital articles, complete with headlines, authors, publication dates, summaries, full column-accurate body text, semantic metadata, embedded figures, and cross-page continuations.

Unlike the dedicated Urdu pipeline (which uses YOLOv8-UrduDoc + UTRNet + Gemini), the English pipeline leverages **DocLayout-YOLO**, **RapidOCR**, and **OpenAI Vision (`gpt-5.6-luna` / `gpt-4o`)** for high-precision Latin script parsing.

```
                                ENGLISH PIPELINE WORKFLOW
                                
       [ Input PDF ]
             │
             ▼
    [ Stage 1: PDF Rendering ] ──► pages/page_XXX.png (200/300 DPI)
             │
             ▼
    [ Stage 2: Masthead Detection ] ──► Newspaper Name, Edition, Date (document.json)
             │
             ▼
    [ Stage 3: DocLayout-YOLO ] ──► models/doclayout_yolo.pt (title, text, figure, table)
             │
             ▼
    [ Stage 4: RapidOCR Engine ] ──► ONNX Runtime English OCR on Layout Blocks
             │
             ▼
    [ Stage 5: Page Cleaner ] ──► Filter Headers, Footers, Rule Lines, Artifacts
             │
             ▼
    [ Stage 6: Article Boundary Grouping ] ──► OpenAI Grouping (ENGLISH_GROUPING_PROMPT)
             │
             ▼
    [ Stage 7: Final Article Cropper ] ──► final_articles_crops/page_XXX/article_YYY/
             │                            (Sets boundaries_ready = true)
             ▼
    [ Stage 8: OpenAI Vision Extraction ] ──► OpenAIArticleExtractor (3-page batches)
             │
             ▼
    [ Stage 9: Continuation Resolution ] ──► Reconcile "Continued on Page X"
             │
             ▼
    [ Stage 10: Logical Article Builder ] ──► final_articles.json & Composited Images
             │
             ▼
    [ Stage 11: PostgreSQL Ingestion ] ──► articles table + Full-Text Search TSVector
```

---

## 2. Core Components & Models

| Component | Technology / Model | File / Configuration | Purpose |
| :--- | :--- | :--- | :--- |
| **PDF Renderer** | PyMuPDF / `pdf2image` | `pipeline/render_pdf.py` | Converts vector/scanned PDF pages to lossless RGB PNGs |
| **Masthead Extractor** | `LocalMastheadExtractor` / OpenAI | `pipeline/intelligence/local/masthead/` | Detects newspaper title, date, issue, language |
| **Layout Detector** | **DocLayout-YOLO** (`doclayout_yolo.pt`) | `pipeline/layout_detector.py` | Detects layout blocks: `title`, `text`, `figure`, `figure_caption`, `table` |
| **OCR Engine** | **RapidOCR** (ONNX Runtime) | `pipeline/ocr/rapidocr_engine.py` | Extracts text and bounding boxes for layout blocks |
| **Page Cleaner** | Geometric / Connected Components | `pipeline/preprocess/page_cleaner.py` | Removes repeated running headers, rules, scan artifacts |
| **Boundary Grouper** | `ArticleGrouper` + OpenAI | `pipeline/article/article_grouper.py` | Groups titles, text blocks, and pictures into cohesive article boundaries |
| **Article Cropper** | Shapely & Pillow | `pipeline/intelligence/final_article_cropper.py` | Generates verified image crops and composite polygons |
| **Vision Extractor** | **OpenAI (`gpt-5.6-luna` / `gpt-4o`)** | `pipeline/intelligence/openai_article_extractor.py` | Direct vision-to-JSON transcription in 3-page batches |
| **Logical Assembler** | `LogicalArticleBuilder` | `pipeline/finalization/logical_article_builder.py` | Merges multi-part articles and generates web-ready images |
| **Database** | PostgreSQL (`psycopg`) | `pipeline/database/import_final_articles.py` | Ingests articles with relational & full-text search schemas |

---

## 3. Detailed Stage-by-Stage Breakdown

### Stage 1: PDF Ingestion & High-Resolution Rendering
* **Input**: Uploaded `.pdf` file in `output/documents/{document_id}/`.
* **Action**:
  * Extracts page count and basic file information.
  * Renders each PDF page into a high-resolution PNG (`pages/page_001.png`, `pages/page_002.png`, etc.) at 200–300 DPI.
* **Output**: `pages/` directory containing raw page scans.

### Stage 2: Masthead & Metadata Detection
* **Action**:
  * Inspects the top 20–30% of the front page (Page 1) to identify newspaper identity.
  * Determines:
    * `newspaper_name` (e.g., *The Times of India*, *The Indian Express*, *The Hindu*)
    * `edition` (e.g., *Delhi*, *Mumbai*)
    * `publish_date` (normalized `YYYY-MM-DD`)
    * `language`: Detected as `English` / `en`
* **Output**: Initial `document.json` metadata record.

### Stage 3: DocLayout-YOLO Layout Detection
* **Model**: `models/doclayout_yolo.pt` (YOLOv10 document architecture).
* **Action**:
  * Scans each page at high resolution (`1024x1024` or native aspect ratio).
  * Classifies bounding regions into distinct layout classes:
    * `title` (Headlines, deck headers)
    * `text` (Body columns and paragraphs)
    * `figure` (Photos, editorial cartoons, infographics)
    * `figure_caption` (Photo captions and credits)
    * `table` (Stock tables, sports scoreboards, schedules)
* **Output**: Saved in `layout/page_XXX_layout.json`.

### Stage 4: RapidOCR Text Recognition
* **Engine**: RapidOCR (`onnxruntime` engine) with English recognition weights.
* **Action**:
  * Runs OCR targeted on the layout boxes detected by DocLayout-YOLO.
  * Produces text lines with line-level bounding coordinates and confidence scores.
* **English Pipeline Optimization**:
  * In `pipeline/languages/english/__init__.py`, `enable_misclassified_figure_recovery` and `enable_low_confidence_retry` are disabled to maximize speed without sacrificing accuracy on standard clean Latin fonts.

### Stage 5: Page Cleaning & Artifact Removal
* **Action**:
  * Removes horizontal and vertical printer rules (`pipeline/preprocess/page_cleaner.py`).
  * Filters out page running headers (e.g., "PAGE 4 | THE TIMES OF INDIA | TUESDAY, MAY 12, 2026").
  * Strips edge page borders and scanner shadows.

### Stage 6: Article Boundary Grouping
* **Prompt**: `pipeline/languages/english/grouping_prompt.py` (`ENGLISH_GROUPING_PROMPT`).
* **Action**:
  * Associates orphan body text blocks with their parent headline.
  * Connects relevant figures and captions to the corresponding story.
  * Uses geometric spatial cues (column borders, margins) and LLM semantic associations to form candidate article bounding boxes.
* **Configuration**:
  * Complex repair passes (`use_orphan_block_reassignment`, `use_article_splitter`, `use_dropped_article_recovery`) are turned off (`False`) for English to maintain deterministic, clean standard grouping behavior.

### Stage 7: Final Article Cropping & Verification
* **Action**:
  * Crops every verified article into its own directory:
    `output/documents/{document_id}/final_articles_crops/page_XXX/article_YYY/`
  * Generates:
    * `page_XXX.png` (the cropped visual article)
    * `crop.json` (precise pixel coordinates, source page, dimensions, padding)
  * Sets `boundaries_ready = True` in `document.json`. At this point, the document can be reviewed and edited in the Frontend Document Viewer.

### Stage 8: Visual Text Extraction via OpenAI Vision
* **Extractor**: `OpenAIArticleExtractor` (`pipeline/intelligence/openai_article_extractor.py`).
* **Model**: `gpt-5.6-luna` or `gpt-4o` (configured via `OPENAI_API_KEY`).
* **Batching Strategy**:
  * Processes articles in batches of **3 pages** per request (`DEFAULT_PAGES_PER_BATCH = 3`).
  * Directly reads the cropped article images visually (no reliance on intermediate OCR errors).
* **Extracted Schema**:
  ```json
  {
    "headline": "Prime Minister Announces New Renewable Energy Policy",
    "subheadline": "Target set for 500 GW non-fossil capacity by 2030",
    "author": "Special Correspondent",
    "location": "New Delhi",
    "date": "2026-05-12",
    "article_text": "Full transcribed body text maintaining exact multi-column reading flow...",
    "summary": "Concise 2-sentence synopsis of the article...",
    "category": "National",
    "sentiment": "positive",
    "entities": ["Ministry of Power", "New Delhi"],
    "topics": ["Renewable Energy", "Climate Policy"],
    "keywords": ["solar", "wind", "policy", "energy"],
    "continuation": {
      "is_continued": true,
      "marker": "Continued on Page 6, Col 2",
      "next_page": 6
    },
    "images": {
      "has_images": true,
      "image_count": 1,
      "items": [{"caption": "Solar park in Gujarat", "confidence": 0.95}]
    }
  }
  ```

### Stage 9: Cross-Page Continuation Resolution
* **Action**:
  * Evaluates articles marked with `continuation.is_continued = true`.
  * Matches continuation markers (e.g., *"Continued from Page 1"*) with target crops in subsequent batches.
  * Establishes directional edges (`source_parts`) between parent article segments.

### Stage 10: Logical Article Finalization & Image Compositing
* **Builder**: `LogicalArticleBuilder` (`pipeline/finalization/logical_article_builder.py`).
* **Action**:
  * Merges multi-page segments into unified `logical_article` records (`logical_0001`, `logical_0002`, etc.).
  * If an article spans multiple pages, composites the crops into a single contiguous display image.
  * Outputs the final manifest: `output/documents/{document_id}/final_articles/final_articles.json`.

### Stage 11: Database Ingestion & Search Indexing
* **Database**: PostgreSQL (`newspaper_archive` database).
* **Action**:
  * Ingests documents into `documents` table and articles into `articles` table.
  * Populates PostgreSQL `search_vector` (`tsvector`) using the `'english'` dictionary configuration for fast full-text search.
  * Updates `document.json` status to `"completed"`.

---

## 4. English Pipeline Configuration Reference

The configuration is declared in [`pipeline/languages/english/__init__.py`](pipeline/languages/english/__init__.py):

```python
ENGLISH = LanguagePipeline(
    code="english",
    matches=matches,
    ocr_engine_factory=lambda: RapidOCREngine(
        enable_misclassified_figure_recovery=False,
        enable_low_confidence_retry=False,
    ),
    ocr_engine_label="default",
    llm_provider="openai",
    grouping_prompt=ENGLISH_GROUPING_PROMPT,
    extraction_prompt_template=ENGLISH_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,

    # Clean standard pipeline settings: repair passes disabled
    use_orphan_block_reassignment=False,
    use_orphan_title_root_repair=False,
    use_unclaimed_kicker_recovery=False,
    use_unclaimed_image_recovery=False,
    use_unclaimed_footprint_recovery=False,
    use_article_splitter=False,
    use_dropped_article_recovery=False,
    use_boundary_decomposition=False,
)
```

---

## 5. Output Directory Structure

For an English document processed as `doc_000100`:

```
output/documents/doc_000100/
├── document.json                           # Overall status & metadata
├── pages/                                  # High-res rendered pages
│   ├── page_001.png
│   └── page_002.png
├── layout/                                 # DocLayout-YOLO detected blocks
│   ├── page_001_layout.json
│   └── page_002_layout.json
├── final_articles_crops/                   # Cropped visual boundaries
│   ├── page_001/
│   │   ├── article_001/
│   │   │   ├── crop.json
│   │   │   └── page_001.png
│   │   └── article_002/
│   │       ├── crop.json
│   │       └── page_001.png
├── openai_article_batches/                 # Vision batch responses & continuations
│   ├── batch_001.json
│   ├── manifest.json
│   └── final_logical_articles.json
└── final_articles/                         # Fully assembled logical articles
    ├── final_articles.json
    └── article_images/
```

---

## 6. Comparison: English vs. Urdu Pipeline

| Feature | English Pipeline | Urdu Pipeline |
| :--- | :--- | :--- |
| **Service Entrypoint** | `PipelineService` (`pipeline_service.py`) | `UrduPipelineService` (`urdu_pipeline_service.py`) |
| **Layout Detection** | **DocLayout-YOLO** (`doclayout_yolo.pt`) | **YOLOv8-UrduDoc** (`yolov8m_UrduDoc.pt`) |
| **OCR Engine** | **RapidOCR** (ONNX Runtime, English) | **UTRNet** (FP16 PyTorch model) |
| **Reading Flow** | Left-to-Right (LTR) | Right-to-Left (RTL) |
| **Vision Extractor** | **OpenAI** (`OpenAIArticleExtractor`) | **Gemini** (`GeminiArticleExtractor`) |
| **Article Density** | Moderate (5–15 articles / page) | Extremely Dense (20–55+ articles / page) |
| **Grouping Engine** | DocLayout blocks + OpenAI Grouping | Connected Component Slices + Geometric Heuristics |
| **Hardware Use** | DocLayout (GPU/CPU) + OpenAI API (Cloud) | YOLOv8 + UTRNet (RTX 2050 CUDA) + Gemini API |

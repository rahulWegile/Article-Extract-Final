-- ============================================================
-- Newspaper archive schema.
--
-- Mounted read-only into the `db` container at
-- /docker-entrypoint-initdb.d/init.sql (see docker-compose.yml) --
-- postgres runs every *.sql file found there once, only on the very
-- first boot of a fresh data volume.
--
-- There is no ORM/migration file anywhere else in this repo (see
-- pipeline/database/delete_document.py's own docstring), so this
-- schema is reverse-engineered from the code that actually reads and
-- writes it:
--   - pipeline/database/import_final_articles.py  (INSERT/ON CONFLICT)
--   - pipeline/database/delete_document.py         (DELETE)
--   - backend/services/article_search.py           (SELECT)
--
-- `documents` and `article_images` are not mentioned in the original
-- two-table spec, but both are required: articles.document_id and
-- article_segments/article_images join against `documents`, and
-- import_final_articles.py's `INSERT INTO article_images (...)` and
-- ArticleSearchService._get_images() both fail without it.
-- ============================================================


-- ============================================================
-- DOCUMENTS
--
-- One row per imported newspaper edition/PDF. Archive search
-- filters (newspaper name, publish date -- see
-- ArticleSearchService.get_newspapers/get_publish_dates) query this
-- table directly.
-- ============================================================

CREATE TABLE IF NOT EXISTS documents (
    id              VARCHAR PRIMARY KEY,
    filename        VARCHAR,
    source_path     VARCHAR,
    newspaper_name  VARCHAR,
    publish_date    DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_newspaper_idx
    ON documents (newspaper_name);

CREATE INDEX IF NOT EXISTS documents_publish_date_idx
    ON documents (publish_date);


-- ============================================================
-- ARTICLES
--
-- One row per LOGICAL article. A continued/jump story spans more
-- than one physical crop -- see article_segments below.
-- ============================================================

CREATE TABLE IF NOT EXISTS articles (
    id                    VARCHAR PRIMARY KEY,
    document_id           VARCHAR NOT NULL
                               REFERENCES documents (id)
                               ON DELETE CASCADE,
    logical_article_id    VARCHAR NOT NULL,
    title                 VARCHAR,
    author                VARCHAR,
    article_date          DATE,
    category              VARCHAR,
    sentiment             VARCHAR,
    summary               TEXT,
    article_text          TEXT,
    composed_image_path   VARCHAR,
    has_images            BOOLEAN NOT NULL DEFAULT false,
    image_count           INTEGER NOT NULL DEFAULT 0,
    entities              JSONB,
    topics                JSONB,
    keywords              JSONB,

    -- Auto-maintained full-text index. backend/services/
    -- article_search.py only ever READS this column (`search_vector
    -- @@ websearch_to_tsquery(...)`, `ts_rank_cd(search_vector,
    -- ...)`) -- it is never set explicitly by any INSERT/UPDATE, so
    -- it has to be a generated column, not a plain one a trigger
    -- keeps in sync.
    search_vector TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(article_text, '')), 'C')
    ) STORED,

    -- import_final_articles.py upserts with
    -- `ON CONFLICT (document_id, logical_article_id)`.
    CONSTRAINT articles_document_logical_unique
        UNIQUE (document_id, logical_article_id)
);

CREATE INDEX IF NOT EXISTS articles_search_idx
    ON articles USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS articles_doc_idx
    ON articles (document_id);


-- ============================================================
-- ARTICLE SEGMENTS
--
-- One row per PHYSICAL crop making up a logical article.
-- title/author/article_date are per-segment overrides that
-- ArticleSearchService._apply_display_metadata reads to prefer a
-- later segment's real headline/byline/date over a first-page
-- placeholder (e.g. a jump story that starts as "WINDOWS" on page 1
-- and only gets its real headline/author/date on the continuation on
-- page 3) -- left NULL by the current importer, which only ever
-- imports page_number/source_article_id/crop_path/bbox, but selected
-- by that method regardless.
-- ============================================================

CREATE TABLE IF NOT EXISTS article_segments (
    id                   VARCHAR PRIMARY KEY,
    article_id           VARCHAR NOT NULL
                              REFERENCES articles (id)
                              ON DELETE CASCADE,
    source_article_id    VARCHAR,
    page_number          INTEGER,
    segment_order        INTEGER NOT NULL,
    article_date         DATE,
    title                VARCHAR,
    author               VARCHAR,
    crop_path            VARCHAR,
    bbox                 JSONB,

    -- import_final_articles.py upserts with
    -- `ON CONFLICT (article_id, segment_order)`.
    CONSTRAINT article_segments_article_order_unique
        UNIQUE (article_id, segment_order)
);

CREATE INDEX IF NOT EXISTS segments_article_idx
    ON article_segments (article_id);


-- ============================================================
-- ARTICLE IMAGES
--
-- One row per Gemini-detected image attached to a logical article
-- (import_final_articles.py::import_images). Not part of the
-- original two-table spec, but required for that importer's
-- `INSERT INTO article_images (...)` to succeed and for
-- ArticleSearchService._get_images to return anything at all.
-- ============================================================

CREATE TABLE IF NOT EXISTS article_images (
    id                   VARCHAR PRIMARY KEY,
    article_id           VARCHAR NOT NULL
                              REFERENCES articles (id)
                              ON DELETE CASCADE,
    segment_id           VARCHAR
                              REFERENCES article_segments (id)
                              ON DELETE SET NULL,
    page_number          INTEGER,
    source_article_id    VARCHAR,
    image_path           VARCHAR,
    description          TEXT,
    caption              TEXT,
    confidence           REAL,
    bbox_x1              REAL,
    bbox_y1              REAL,
    bbox_x2              REAL,
    bbox_y2              REAL,
    width                INTEGER,
    height               INTEGER,

    -- ArticleSearchService._get_images orders by
    -- `page_number ASC, created_at ASC` -- required, not decorative.
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS images_article_idx
    ON article_images (article_id);

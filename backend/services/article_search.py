from __future__ import annotations

from typing import Any

from pipeline.database.db import get_connection


class ArticleSearchService:
    """
    Search logical newspaper articles stored in PostgreSQL.

    Supports:

    1. Full-text article search
    2. Exact article retrieval
    3. Article segments
    4. Article images
    5. Frontend image URLs
    """

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str | None = None,
        newspaper: str | None = None,
        publish_date: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:

        query = (query or "").strip()
        newspaper = (newspaper or "").strip() or None
        publish_date = (publish_date or "").strip() or None

        # No filters at all means "browse the entire archive" --
        # the WHERE clauses below already treat an empty query and
        # NULL newspaper/publish_date as "match everything".

        limit = max(
            1,
            min(limit, 100),
        )

        offset = max(
            0,
            offset,
        )

        # ----------------------------------------------------
        # COUNT
        # ----------------------------------------------------

        count_sql = """
            SELECT COUNT(*) AS count
            FROM articles a
            JOIN documents d
                ON d.id = a.document_id
            WHERE
                (
                    %s = ''
                    OR a.search_vector @@ websearch_to_tsquery(
                        'english',
                        %s
                    )
                )
                AND (
                    %s::text IS NULL
                    OR d.newspaper_name = %s
                )
                AND (
                    %s::date IS NULL
                    OR d.publish_date = %s::date
                )
        """

        # ----------------------------------------------------
        # SEARCH RESULTS
        # ----------------------------------------------------

        article_sql = """
            SELECT
                a.id,
                a.document_id,
                a.logical_article_id,
                a.title,
                a.author,
                a.article_date,
                a.article_text,
                a.summary,
                a.category,
                a.sentiment,
                a.entities,
                a.topics,
                a.keywords,
                a.has_images,
                a.image_count,
                a.composed_image_path,

                d.newspaper_name,
                d.publish_date,

                CASE
                    WHEN %s = '' THEN 0
                    ELSE ts_rank_cd(
                        a.search_vector,
                        websearch_to_tsquery(
                            'english',
                            %s
                        )
                    )
                END AS rank

            FROM articles a

            JOIN documents d
                ON d.id = a.document_id

            WHERE
                (
                    %s = ''
                    OR a.search_vector @@ websearch_to_tsquery(
                        'english',
                        %s
                    )
                )
                AND (
                    %s::text IS NULL
                    OR d.newspaper_name = %s
                )
                AND (
                    %s::date IS NULL
                    OR d.publish_date = %s::date
                )

            ORDER BY
                rank DESC,
                a.title ASC

            LIMIT %s
            OFFSET %s
        """

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    count_sql,
                    (
                        query,
                        query,
                        newspaper,
                        newspaper,
                        publish_date,
                        publish_date,
                    ),
                )

                count_row = cur.fetchone()

                total = int(
                    count_row["count"]
                )

                cur.execute(
                    article_sql,
                    (
                        query,
                        query,
                        query,
                        query,
                        newspaper,
                        newspaper,
                        publish_date,
                        publish_date,
                        limit,
                        offset,
                    ),
                )

                articles = cur.fetchall()

        results = []

        for article in articles:

            article_id = article["id"]

            # ------------------------------------------------
            # DISPLAY DATE
            # ------------------------------------------------

            article["display_date"] = article["article_date"]

            # ------------------------------------------------
            # EDITION METADATA
            # ------------------------------------------------

            if article.get("publish_date"):
                article["publish_date"] = (
                    article["publish_date"].isoformat()
                )

            # ------------------------------------------------
            # DATE → STRING
            # ------------------------------------------------

            if article["article_date"]:

                article["article_date"] = (
                    article[
                        "article_date"
                    ].isoformat()
                )

            if article["display_date"]:

                article["display_date"] = (
                    article[
                        "display_date"
                    ].isoformat()
                )

            # ------------------------------------------------
            # UUID → STRING
            # ------------------------------------------------

            article["id"] = str(
                article["id"]
            )

            article["document_id"] = str(
                article["document_id"]
            )

            # ------------------------------------------------
            # REMOVE INTERNAL RANK
            # ------------------------------------------------

            article.pop(
                "rank",
                None,
            )

            # ------------------------------------------------
            # SEGMENTS / DISPLAY METADATA
            # ------------------------------------------------

            article["segments"] = (
                self._get_segments(
                    article_id
                )
            )

            self._apply_display_metadata(
                article,
                article["segments"],
            )

            # ------------------------------------------------
            # IMAGES
            # ------------------------------------------------

            article["images"] = (
                self._get_images(
                    article_id
                )
            )

            results.append(
                article
            )

        return {
            "query": query,
            "newspaper": newspaper,
            "publish_date": publish_date,
            "total": total,
            "limit": limit,
            "offset": offset,
            "results": results,
        }

    # ========================================================
    # NEWSPAPERS
    # ========================================================

    def get_newspapers(self) -> list[str]:
        """
        Return all distinct newspaper names available in the
        archive.
        """

        sql = """
            SELECT DISTINCT
                newspaper_name
            FROM documents
            WHERE newspaper_name IS NOT NULL
              AND TRIM(newspaper_name) <> ''
            ORDER BY newspaper_name ASC
        """

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        return [
            row["newspaper_name"]
            for row in rows
            if row.get("newspaper_name")
        ]

    # ========================================================
    # PUBLISH DATES
    # ========================================================

    def get_publish_dates(
        self,
        newspaper: str | None = None,
    ) -> list[str]:
        """
        Return publication dates.

        If newspaper is supplied, return dates belonging only
        to that newspaper.
        """

        if newspaper:
            sql = """
                SELECT DISTINCT
                    publish_date
                FROM documents
                WHERE newspaper_name = %s
                  AND publish_date IS NOT NULL
                ORDER BY publish_date DESC
            """
            params = (newspaper,)
        else:
            sql = """
                SELECT DISTINCT
                    publish_date
                FROM documents
                WHERE publish_date IS NOT NULL
                ORDER BY publish_date DESC
            """
            params = ()

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            row["publish_date"].isoformat()
            for row in rows
            if row.get("publish_date")
        ]

    # ========================================================
    # EXACT ARTICLE
    # ========================================================

    def get_article(
        self,
        document_id: str,
        logical_article_id: str,
    ) -> dict[str, Any] | None:

        sql = """
            SELECT
                a.id,
                a.document_id,
                a.logical_article_id,
                a.title,
                a.author,
                a.article_date,
                a.article_text,
                a.summary,
                a.category,
                a.sentiment,
                a.entities,
                a.topics,
                a.keywords,
                a.has_images,
                a.image_count,
                a.composed_image_path,

                d.newspaper_name,
                d.publish_date

            FROM articles a

            JOIN documents d
                ON d.id = a.document_id

            WHERE
                a.document_id = %s
                AND
                a.logical_article_id = %s

            LIMIT 1
        """

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    sql,
                    (
                        document_id,
                        logical_article_id,
                    ),
                )

                article = cur.fetchone()

        if article is None:

            return None

        # ----------------------------------------------------
        # UUIDS
        # ----------------------------------------------------

        article["id"] = str(
            article["id"]
        )

        article["document_id"] = str(
            article["document_id"]
        )

        # ----------------------------------------------------
        # DATE
        #
        # Only the article's own date is returned.
        # ----------------------------------------------------

        article["display_date"] = article["article_date"]

        if article.get("publish_date"):
            article["publish_date"] = (
                article["publish_date"].isoformat()
            )

        if article["article_date"]:

            article["article_date"] = (
                article[
                    "article_date"
                ].isoformat()
            )

        if article["display_date"]:

            article["display_date"] = (
                article[
                    "display_date"
                ].isoformat()
            )

        # ----------------------------------------------------
        # SEGMENTS
        # ----------------------------------------------------

        article["segments"] = (
            self._get_segments(
                article["id"]
            )
        )

        # ----------------------------------------------------
        # DISPLAY METADATA
        # ----------------------------------------------------
        #
        # For a continued article, use the metadata from the
        # latest physical segment that actually contains it.
        #
        # Example:
        #   Page 1: WINDOWS / no author / no date
        #   Page 2: Atishi joins... / AGE CORRESPONDENT / 2026-07-13
        #
        # Frontend therefore receives the Page 2 metadata.
        # ----------------------------------------------------

        self._apply_display_metadata(
            article,
            article["segments"],
        )

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------

        article["images"] = (
            self._get_images(
                article["id"]
            )
        )

        return article

    # ========================================================
    # DISPLAY METADATA FOR CONTINUED ARTICLES
    # ========================================================

    @staticmethod
    def _apply_display_metadata(
        article: dict[str, Any],
        segments: list[dict[str, Any]],
    ) -> None:
        """
        Choose the frontend header metadata.

        For a continued logical article, the first physical segment
        may have a placeholder/short title such as "WINDOWS" and no
        author/date, while a later physical segment contains the
        actual headline, author and date.

        Therefore:
          - prefer the LAST segment with a non-empty title
          - prefer the LAST segment with a non-empty author
          - prefer the LAST segment with a non-empty article_date

        This does not modify segment order, crop paths, page numbers,
        source_article_id, or bounding boxes.
        """

        display_title = article.get("title")
        display_author = article.get("author")
        display_date = article.get("article_date")

        for segment in reversed(segments):
            value = segment.get("title")
            if value and str(value).strip():
                display_title = value
                break

        for segment in reversed(segments):
            value = segment.get("author")
            if value and str(value).strip():
                display_author = value
                break

        for segment in reversed(segments):
            value = segment.get("article_date")
            if value:
                display_date = value
                break

        article["title"] = display_title
        article["author"] = display_author
        article["article_date"] = display_date

        # The frontend uses display_date for the visible date.
        # Keep it synchronized with the selected physical segment.
        article["display_date"] = display_date

    # ========================================================
    # SEGMENTS
    # ========================================================

    def _get_segments(
        self,
        article_id: str,
    ) -> list[dict[str, Any]]:

        sql = """
            SELECT
                id,
                page_number,
                source_article_id,
                segment_order,
                crop_path,
                bbox,
                title,
                author,
                article_date
            FROM article_segments

            WHERE
                article_id = %s

            ORDER BY
                segment_order ASC
        """

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    sql,
                    (article_id,),
                )

                rows = cur.fetchall()

        segments = []

        for segment in rows:

            segment["id"] = str(
                segment["id"]
            )

            # Convert PostgreSQL date to JSON-safe ISO string.
            if segment.get("article_date"):
                segment["article_date"] = (
                    segment["article_date"].isoformat()
                )

            segment["crop_url"] = (
                self._build_image_url(
                    segment["crop_path"]
                )
            )

            segment.pop(
                "crop_path",
                None,
            )

            segments.append(
                segment
            )

        return segments

    # ========================================================
    # IMAGES
    # ========================================================

    def _get_images(
        self,
        article_id,
    ) -> list[dict[str, Any]]:

        sql = """
            SELECT
                id,
                segment_id,
                page_number,
                source_article_id,
                image_path,
                description,
                caption,
                confidence,
                bbox_x1,
                bbox_y1,
                bbox_x2,
                bbox_y2,
                width,
                height
            FROM article_images

            WHERE
                article_id = %s

            ORDER BY
                page_number ASC,
                created_at ASC
        """

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    sql,
                    (article_id,),
                )

                rows = cur.fetchall()

        images = []

        for image in rows:

            image["id"] = str(
                image["id"]
            )

            if image["segment_id"]:

                image["segment_id"] = str(
                    image["segment_id"]
                )

            image["url"] = (
                self._build_image_url(
                    image["image_path"]
                )
            )

            image.pop(
                "image_path",
                None,
            )

            images.append(
                image
            )

        return images

    # ========================================================
    # BUILD FRONTEND URL
    # ========================================================

    @staticmethod
    def _build_image_url(
        image_path: str | None,
    ) -> str | None:

        if not image_path:

            return None

        normalized = image_path.replace(
            "\\",
            "/",
        )

        marker = "/output/"

        if marker in normalized:

            relative_path = (
                normalized.split(
                    marker,
                    1,
                )[1]
            )

            return (
                f"/output/{relative_path}"
            )

        # ------------------------------------------------------
        # RELATIVE PATH (no leading slash / no drive letter)
        #
        # e.g. "output\documents\doc_.../page_001.png", possibly
        # with a leading "./" -- these never contain the "/output/"
        # marker above since they have no leading slash. Handle them
        # too rather than silently returning None: stored crop_path/
        # image_path values are CWD-relative wherever they came from
        # a source that didn't resolve() them.
        # ------------------------------------------------------

        stripped = (
            normalized[2:]
            if normalized.startswith("./")
            else normalized
        )

        if stripped == "output" or stripped.startswith("output/"):

            return f"/{stripped}"

        return None
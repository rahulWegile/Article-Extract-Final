"""
Remove a document and everything derived from it (articles, article
segments, article images) from the database.

Deletes are issued explicitly in child-to-parent order rather than
relying on `ON DELETE CASCADE` -- this repo's DB schema isn't version
controlled anywhere (no CREATE TABLE/migration file exists), so
whether the live FK constraints actually cascade can't be confirmed
from source. Explicit deletes are correct regardless of what the FKs
do (deleting an already-empty table is a no-op, not an error).
"""

from __future__ import annotations

from pipeline.database.db import get_connection
from pipeline.database.import_final_articles import deterministic_uuid


def delete_document_from_db(document_id: str) -> dict:
    """
    Args:
        document_id: the filesystem document id (e.g. "doc_000002"),
            matching `document.json["document_id"]" -- the same value
            import_final_articles.py hashes into the DB's document UUID.

    Returns:
        {"found": bool, "images_deleted": int, "segments_deleted": int,
         "articles_deleted": int, "document_deleted": int}
    """

    document_uuid = deterministic_uuid("document", document_id)

    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute(
            "SELECT 1 FROM documents WHERE id = %s",
            (document_uuid,),
        )
        found = cur.fetchone() is not None

        cur.execute(
            """
            DELETE FROM article_images
            WHERE article_id IN (
                SELECT id FROM articles WHERE document_id = %s
            )
            """,
            (document_uuid,),
        )
        images_deleted = cur.rowcount

        cur.execute(
            """
            DELETE FROM article_segments
            WHERE article_id IN (
                SELECT id FROM articles WHERE document_id = %s
            )
            """,
            (document_uuid,),
        )
        segments_deleted = cur.rowcount

        cur.execute(
            "DELETE FROM articles WHERE document_id = %s",
            (document_uuid,),
        )
        articles_deleted = cur.rowcount

        cur.execute(
            "DELETE FROM documents WHERE id = %s",
            (document_uuid,),
        )
        document_deleted = cur.rowcount

        conn.commit()

        return {
            "found": found,
            "images_deleted": images_deleted,
            "segments_deleted": segments_deleted,
            "articles_deleted": articles_deleted,
            "document_deleted": document_deleted,
        }

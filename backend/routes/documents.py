import json

from fastapi import APIRouter, HTTPException

from backend.core.settings import DOCUMENTS_DIR
from backend.services.document_manager import DocumentManager

from backend.models.document import (
    DocumentSummary,
    DocumentDetail,
    PageResponse,
)

from pipeline.database.delete_document import delete_document_from_db


router = APIRouter()




@router.get(
    "/documents",
    response_model=list[DocumentSummary],
)
def get_documents():

    documents = []

    if not DOCUMENTS_DIR.exists():
        return documents

    for document_dir in sorted(DOCUMENTS_DIR.iterdir()):

        if not document_dir.is_dir():
            continue

        metadata_file = document_dir / "document.json"

        if not metadata_file.exists():
            continue

        with open(
            metadata_file,
            "r",
            encoding="utf-8",
        ) as f:

            metadata = json.load(f)

        documents.append(metadata)

    return documents


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetail,
)
def get_document(document_id: str):

    document_dir = DOCUMENTS_DIR / document_id

    if not document_dir.exists():
        return {
            "error": "Document not found"
        }

    metadata_file = document_dir / "document.json"

    if not metadata_file.exists():
        return {
            "error": "Metadata not found"
        }

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    return metadata


@router.get(
    "/documents/{document_id}/page/{page_number}",
    response_model=PageResponse,
)
def get_page(
    document_id: str,
    page_number: int,
):

    document_dir = DOCUMENTS_DIR / document_id

    if not document_dir.exists():
        return {
            "error": "Document not found"
        }

    metadata_file = document_dir / "document.json"

    if not metadata_file.exists():
        return {
            "error": "Metadata not found"
        }

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    image_path = (
        document_dir
        / "final"
        / f"page_{page_number:03d}_final_boundaries.png"
    )

    if not image_path.exists():
        return {
            "error": "Page not found"
        }

    image_url = (
        f"/documents/{document_id}/final/"
        f"page_{page_number:03d}_final_boundaries.png"
    )

    return {

        "document_id": document_id,

        "pdf_name": metadata["pdf_name"],

        "page": page_number,

        "page_count": metadata["page_count"],

        "image": image_url,

    }


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """
    Permanently delete a document: its on-disk workspace
    (output/documents/doc_NNNNNN/) and its database rows
    (documents/articles/article_segments/article_images), if any.

    Deletes both stores independently and best-effort -- a document
    can legitimately exist on only one side (e.g. still processing
    and never imported to the DB yet, or DB rows left over from a
    manually-cleared output folder), so neither side existing is not
    itself an error; only "found on neither side" is a 404.
    """

    db_result = None
    db_error = None

    try:
        db_result = delete_document_from_db(document_id)
    except Exception as exc:
        db_error = str(exc)

    try:
        folder_deleted = DocumentManager().delete_document(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db_found = bool(db_result and (db_result["found"] or db_result["document_deleted"]))

    if not folder_deleted and not db_found:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}",
        )

    return {
        "success": True,
        "document_id": document_id,
        "folder_deleted": folder_deleted,
        "database": db_result,
        "database_error": db_error,
    }
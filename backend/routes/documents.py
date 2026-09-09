import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from backend.core.settings import DOCUMENTS_DIR
from backend.services.document_manager import DocumentManager
from backend.services import boundary_editor

from backend.models.document import (
    DocumentSummary,
    DocumentDetail,
    PageResponse,
    PageBoundariesResponse,
    BoundaryUpdateRequest,
    BoundaryUpdateResponse,
    BoundaryDeleteResponse,
    BoundaryMergeRequest,
    BoundaryMergeResponse,
)

from pipeline.database.delete_document import delete_document_from_db
from pipeline.database.import_final_articles import deterministic_uuid
from pipeline.export.pdf_export import export_document_pdf


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

        try:

            with open(
                metadata_file,
                "r",
                encoding="utf-8",
            ) as f:

                metadata = json.load(f)

        except (json.JSONDecodeError, OSError):
            continue

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

    plain_image_url = (
        f"/documents/{document_id}/pages/"
        f"page_{page_number:03d}.png"
    )

    return {

        "document_id": document_id,

        "pdf_name": metadata["pdf_name"],

        "page": page_number,

        "page_count": metadata["page_count"],

        "image": image_url,

        "plain_image": plain_image_url,

    }


THUMBNAIL_WIDTH = 260


@router.get("/documents/{document_id}/page/{page_number}/thumbnail")
def get_page_thumbnail(
    document_id: str,
    page_number: int,
):

    document_dir = DOCUMENTS_DIR / document_id

    source_path = document_dir / "pages" / f"page_{page_number:03d}.png"

    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Page not found")

    thumb_path = document_dir / "thumbnails" / f"page_{page_number:03d}.jpg"

    if (
        not thumb_path.is_file()
        or thumb_path.stat().st_mtime < source_path.stat().st_mtime
    ):

        thumb_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as image:

            image = image.convert("RGB")

            ratio = THUMBNAIL_WIDTH / image.width
            size = (THUMBNAIL_WIDTH, max(1, round(image.height * ratio)))

            image.resize(size, Image.LANCZOS).save(
                thumb_path,
                format="JPEG",
                quality=78,
            )

    return FileResponse(thumb_path, media_type="image/jpeg")


@router.get(
    "/documents/{document_id}/page/{page_number}/boundaries",
    response_model=PageBoundariesResponse,
)
def get_page_boundaries(
    document_id: str,
    page_number: int,
):

    document_dir = DOCUMENTS_DIR / document_id

    if not document_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    boundaries = boundary_editor.list_page_boundaries(
        document_dir,
        page_number,
    )

    return {
        "document_id": document_id,
        "document_uuid": deterministic_uuid("document", document_id),
        "page": page_number,
        "boundaries": boundaries,
    }


@router.post(
    "/documents/{document_id}/page/{page_number}/boundaries",
    response_model=BoundaryUpdateResponse,
)
def update_page_boundary(
    document_id: str,
    page_number: int,
    payload: BoundaryUpdateRequest,
):

    document_dir = DOCUMENTS_DIR / document_id

    if not document_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    try:

        result = boundary_editor.save_boundary_and_reextract(
            document_dir=document_dir,
            document_id=document_id,
            page_number=page_number,
            article_id=payload.article_id,
            bbox=payload.bbox.model_dump(),
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return result


@router.post(
    "/documents/{document_id}/page/{page_number}/boundaries/merge",
    response_model=BoundaryMergeResponse,
)
def merge_page_boundaries(
    document_id: str,
    page_number: int,
    payload: BoundaryMergeRequest,
):

    document_dir = DOCUMENTS_DIR / document_id

    if not document_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    try:

        result = boundary_editor.merge_boundaries(
            document_dir=document_dir,
            document_id=document_id,
            page_number=page_number,
            article_ids=payload.article_ids,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return result


@router.delete(
    "/documents/{document_id}/page/{page_number}/boundaries/{article_id}",
    response_model=BoundaryDeleteResponse,
)
def delete_page_boundary(
    document_id: str,
    page_number: int,
    article_id: str,
):

    document_dir = DOCUMENTS_DIR / document_id

    if not document_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    try:

        result = boundary_editor.delete_boundary(
            document_dir=document_dir,
            document_id=document_id,
            page_number=page_number,
            article_id=article_id,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@router.get("/documents/{document_id}/export/pdf")
def export_pdf(document_id: str):

    document_dir = DOCUMENTS_DIR / document_id

    if not document_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    metadata_file = document_dir / "document.json"

    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    try:
        pdf_path = export_document_pdf(document_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    pdf_name = Path(metadata["pdf_name"]).stem

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{pdf_name}_boundaries.pdf",
    )


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
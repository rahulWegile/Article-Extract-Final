import json
import re
import shutil

from pathlib import Path
from datetime import datetime

from backend.core.settings import DOCUMENTS_DIR


class DocumentManager:
    """
    Manage newspaper document workspaces.

    Document IDs are generated from the highest existing
    document number, not from the number of folders.

    Examples:

        doc_000001
        doc_000002
        doc_000003

    If only doc_000002 exists, the next document will be:

        doc_000003

    If no documents exist, the first document will be:

        doc_000001
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self):

        self.root = DOCUMENTS_DIR

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # FIND NEXT DOCUMENT ID
    # =========================================================

    def _get_next_document_id(self):
        """
        Find the highest existing document number and
        return the next sequential ID.

        This is safer than:

            len(existing) + 1

        because folders may have been deleted.

        Example:

            doc_000001
            doc_000004

        Next ID:

            doc_000005
        """

        highest_id = 0

        # -----------------------------------------------------
        # Find all document directories
        # -----------------------------------------------------

        for path in self.root.glob("doc_*"):

            if not path.is_dir():
                continue

            match = re.fullmatch(
                r"doc_(\d+)",
                path.name,
            )

            if not match:
                continue

            try:

                number = int(
                    match.group(1)
                )

            except ValueError:

                continue

            highest_id = max(
                highest_id,
                number,
            )

        next_id = highest_id + 1

        return f"doc_{next_id:06d}"

    # =========================================================
    # CREATE DOCUMENT
    # =========================================================

    def create_document(
        self,
        pdf_path,
    ):
        """
        Create a new document workspace.

        Steps:

            1. Generate safe document ID.
            2. Create document directory.
            3. Copy original PDF.
            4. Create initial document.json.
        """

        # =====================================================
        # VALIDATE PDF
        # =====================================================

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if not pdf_path.is_file():

            raise ValueError(
                f"PDF path is not a file: "
                f"{pdf_path}"
            )

        # =====================================================
        # GENERATE DOCUMENT ID
        # =====================================================

        document_id = (
            self._get_next_document_id()
        )

        # =====================================================
        # CREATE DOCUMENT DIRECTORY
        # =====================================================

        document_dir = (
            self.root
            / document_id
        )

        document_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        # =====================================================
        # COPY ORIGINAL PDF
        # =====================================================

        destination_pdf = (
            document_dir
            / "original.pdf"
        )

        shutil.copy2(
            pdf_path,
            destination_pdf,
        )

        # =====================================================
        # INITIAL METADATA
        # =====================================================

        metadata = {

            "document_id":
                document_id,

            "pdf_name":
                pdf_path.name,

            "uploaded_at":
                datetime.now().isoformat(),

            "status":
                "processing",

            "page_count":
                0,

        }

        # =====================================================
        # SAVE DOCUMENT JSON
        # =====================================================

        metadata_path = (
            document_dir
            / "document.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

        # =====================================================
        # LOG
        # =====================================================

        print()

        print("=" * 60)
        print("DOCUMENT CREATED")
        print("=" * 60)

        print(
            f"Document ID : "
            f"{document_id}"
        )

        print(
            f"Document Dir: "
            f"{document_dir}"
        )

        print(
            f"Original PDF: "
            f"{destination_pdf}"
        )

        print("=" * 60)

        # =====================================================
        # RETURN
        # =====================================================

        return {

            "document_id":
                document_id,

            "document_dir":
                document_dir,

        }

    # =========================================================
    # DELETE DOCUMENT
    # =========================================================

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """
        Permanently remove a document's on-disk workspace
        (output/documents/doc_NNNNNN/ and everything under it --
        pages, crops, final_articles, etc.).

        Returns:
            True if a directory was found and removed,
            False if no such document directory exists.
        """

        # =====================================================
        # VALIDATE ID FORMAT
        #
        # Rejects anything that isn't exactly "doc_<digits>" so a
        # crafted document_id (e.g. "../../etc") can never resolve
        # outside self.root.
        # =====================================================

        if not re.fullmatch(r"doc_\d+", document_id):

            raise ValueError(
                f"Invalid document_id: {document_id!r}"
            )

        document_dir = (
            self.root
            / document_id
        )

        if not document_dir.is_dir():

            return False

        shutil.rmtree(
            document_dir
        )

        print()
        print("=" * 60)
        print("DOCUMENT DELETED")
        print("=" * 60)
        print(f"Document ID : {document_id}")
        print(f"Document Dir: {document_dir}")
        print("=" * 60)

        return True
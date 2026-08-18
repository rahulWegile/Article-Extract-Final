import { useState } from "react";

import { Link } from "react-router-dom";

import api from "../api/api";

function DocumentCard({ document, onDeleted }) {

    const [deleting, setDeleting] = useState(false);

    const handleDelete = async () => {

        const confirmed = window.confirm(

            `Delete "${document.pdf_name}" (${document.document_id})? ` +
            "This permanently removes its pages, articles, and images. " +
            "This cannot be undone."

        );

        if (!confirmed) {
            return;
        }

        setDeleting(true);

        try {

            await api.delete(`/documents/${document.document_id}`);

            if (onDeleted) {

                onDeleted(document.document_id);

            }

        }

        catch (error) {

            console.error(error);

            window.alert(

                "Failed to delete document: " +
                (error.response?.data?.detail || error.message)

            );

            setDeleting(false);

        }

    };

    return (

        <div className="document-card">

            <h3>

                📄 {document.pdf_name}

            </h3>

            <p>

                <strong>ID:</strong> {document.document_id}

            </p>

            <p>

                <strong>Pages:</strong> {document.page_count}

            </p>

            <p>

                <strong>Status:</strong> {document.status}

            </p>

            <div className="document-card-actions">

                <Link

                    to={`/viewer/${document.document_id}`}

                    className="open-btn"

                >

                    Open

                </Link>

                <button

                    type="button"

                    className="delete-btn"

                    onClick={handleDelete}

                    disabled={deleting}

                >

                    {deleting ? "Deleting..." : "Delete"}

                </button>

            </div>

        </div>

    );

}

export default DocumentCard;

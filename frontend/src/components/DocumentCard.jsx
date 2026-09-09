import { Link } from "react-router-dom";

const STATUS_LABELS = {
    completed: "Completed",
    processing: "Processing",
    failed: "Failed",
    pending: "Pending",
};

function StatusPill({ status }) {

    const key = (status || "").toLowerCase();
    const label = STATUS_LABELS[key] || status || "Unknown";

    return (

        <span className={`status-pill status-${key || "unknown"}`}>

            <span className="status-dot" aria-hidden="true" />

            {label}

        </span>

    );

}

function DocumentCard({ document, deleting, onRequestDelete }) {

    return (

        <article className="document-card">

            <div className="document-card-top">

                <span className="document-card-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path
                            d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinejoin="round"
                        />
                        <path d="M14 3v5h5" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
                    </svg>
                </span>

                <StatusPill status={document.status} />

            </div>

            <h3 className="document-card-title" title={document.pdf_name}>

                {document.pdf_name}

            </h3>

            <dl className="document-card-meta">

                <div className="meta-row">
                    <dt>Document ID</dt>
                    <dd title={document.document_id}>{document.document_id}</dd>
                </div>

                <div className="meta-row">
                    <dt>Pages</dt>
                    <dd>{document.page_count}</dd>
                </div>

            </dl>

            <div className="document-card-actions">

                <Link

                    to={`/viewer/${document.document_id}`}

                    className="btn btn-primary btn-sm"

                >

                    Open

                </Link>

                <button

                    type="button"

                    className="btn btn-danger-ghost btn-sm"

                    onClick={() => onRequestDelete(document)}

                    disabled={deleting}

                >

                    {deleting ? "Deleting…" : "Delete"}

                </button>

            </div>

        </article>

    );

}

export default DocumentCard;

import { useCallback, useEffect, useRef, useState } from "react";

import api from "../api/api";

import Navbar from "../components/Navbar";
import UploadCard from "../components/UploadCard";
import DocumentCard from "../components/DocumentCard";
import ConfirmDialog from "../components/ConfirmDialog";
import ToastStack from "../components/ToastStack";

const TOAST_LIFETIME_MS = 5000;

function Home() {

    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    const [confirmTarget, setConfirmTarget] = useState(null);
    const [deletingId, setDeletingId] = useState(null);

    const [toasts, setToasts] = useState([]);
    const toastCounterRef = useRef(0);

    const dismissToast = useCallback((id) => {

        setToasts((current) => current.filter((toast) => toast.id !== id));

    }, []);

    const pushToast = useCallback((type, text) => {

        toastCounterRef.current += 1;

        const id = toastCounterRef.current;

        setToasts((current) => [...current, { id, type, text }]);

        setTimeout(() => dismissToast(id), TOAST_LIFETIME_MS);

    }, [dismissToast]);

    const loadDocuments = async () => {

        setLoading(true);
        setError("");

        try {

            const response = await api.get("/documents");
            setDocuments(response.data);

        }

        catch (error) {

            console.error(error);
            setError("Unable to load documents. Is the backend running?");

        }

        finally {

            setLoading(false);

        }

    };

    useEffect(() => {

        loadDocuments();

    }, []);

    const handleBoundariesCreated = useCallback((articleCount) => {

        pushToast(
            "info",
            articleCount
                ? `${articleCount} article${articleCount === 1 ? "" : "s"} detected -- boundaries are ready, you can open the document now.`
                : "Article boundaries are ready -- you can open the document now."
        );

        loadDocuments();

    }, [pushToast]);

    const handleRequestDelete = (document) => {

        setConfirmTarget(document);

    };

    const handleCancelDelete = () => {

        if (deletingId) {
            return;
        }

        setConfirmTarget(null);

    };

    const handleConfirmDelete = async () => {

        if (!confirmTarget) {
            return;
        }

        setDeletingId(confirmTarget.document_id);

        try {

            await api.delete(`/documents/${confirmTarget.document_id}`);

            setDocuments(

                (previous) => previous.filter(

                    (document) => document.document_id !== confirmTarget.document_id

                )

            );

            pushToast("success", `"${confirmTarget.pdf_name}" was deleted.`);

            setConfirmTarget(null);

        }

        catch (error) {

            console.error(error);

            pushToast(

                "error",

                "Failed to delete document: " +
                (error.response?.data?.detail || error.message)

            );

        }

        finally {

            setDeletingId(null);

        }

    };

    return (

        <div className="app-shell">

            <Navbar />

            <main className="home-main">

                <section className="home-hero">

                    <span className="home-eyebrow">Newspaper Studio</span>

                    <h1>Digitize &amp; structure your newspaper archive</h1>
                    <p>Upload a scanned edition to detect page layout, article boundaries, and text automatically.</p>

                </section>

                <section className="home-section">

                    <UploadCard
                        onUploadSuccess={loadDocuments}
                        onNotify={pushToast}
                        onBoundariesCreated={handleBoundariesCreated}
                    />

                </section>

                <section className="home-section documents-section">

                    <div className="documents-header">

                        <h2>Processed Documents</h2>

                        {!loading && documents.length > 0 && (
                            <span className="documents-count">
                                {documents.length} {documents.length === 1 ? "document" : "documents"}
                            </span>
                        )}

                    </div>

                    {error && (

                        <div className="state-banner state-banner-error">
                            {error}
                        </div>

                    )}

                    {loading ? (

                        <div className="documents-grid" aria-hidden="true">

                            {[0, 1, 2].map((key) => (
                                <div className="document-card-skeleton" key={key} />
                            ))}

                        </div>

                    ) : documents.length === 0 && !error ? (

                        <div className="empty-state">

                            <span className="empty-state-icon" aria-hidden="true">
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

                            <h3>No documents yet</h3>
                            <p>Upload a PDF above to get started with your first edition.</p>

                        </div>

                    ) : (

                        <div className="documents-grid">

                            {documents.map(

                                (document) => (

                                    <DocumentCard

                                        key={document.document_id}

                                        document={document}

                                        deleting={deletingId === document.document_id}

                                        onRequestDelete={handleRequestDelete}

                                    />

                                )

                            )}

                        </div>

                    )}

                </section>

            </main>

            <ConfirmDialog

                open={Boolean(confirmTarget)}
                title={confirmTarget ? `Delete "${confirmTarget.pdf_name}"?` : ""}
                message="This permanently removes its pages, articles, and images. This cannot be undone."
                confirmLabel="Delete document"
                danger
                loading={Boolean(deletingId)}
                onConfirm={handleConfirmDelete}
                onCancel={handleCancelDelete}

            />

            <ToastStack toasts={toasts} onDismiss={dismissToast} />

        </div>

    );

}

export default Home;

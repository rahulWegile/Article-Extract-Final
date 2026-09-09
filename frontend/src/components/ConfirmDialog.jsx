import { useEffect, useRef } from "react";

function ConfirmDialog({
    open,
    title,
    message,
    confirmLabel = "Confirm",
    cancelLabel = "Cancel",
    danger = false,
    loading = false,
    onConfirm,
    onCancel,
}) {

    const confirmButtonRef = useRef(null);

    useEffect(() => {

        if (!open) {
            return undefined;
        }

        confirmButtonRef.current?.focus();

        const handleKeyDown = (event) => {

            if (event.key === "Escape" && !loading) {
                onCancel();
            }

        };

        window.addEventListener("keydown", handleKeyDown);

        return () => window.removeEventListener("keydown", handleKeyDown);

    }, [open, loading, onCancel]);

    if (!open) {
        return null;
    }

    return (

        <div
            className="modal-overlay"
            onMouseDown={(event) => {

                if (event.target === event.currentTarget && !loading) {
                    onCancel();
                }

            }}
        >

            <div
                className="modal-card"
                role="alertdialog"
                aria-modal="true"
                aria-labelledby="confirm-dialog-title"
            >

                <h3 id="confirm-dialog-title" className="modal-title">
                    {title}
                </h3>

                {message && <p className="modal-message">{message}</p>}

                <div className="modal-actions">

                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={onCancel}
                        disabled={loading}
                    >
                        {cancelLabel}
                    </button>

                    <button
                        ref={confirmButtonRef}
                        type="button"
                        className={"btn " + (danger ? "btn-danger" : "btn-primary")}
                        onClick={onConfirm}
                        disabled={loading}
                    >
                        {loading ? "Please wait…" : confirmLabel}
                    </button>

                </div>

            </div>

        </div>

    );

}

export default ConfirmDialog;

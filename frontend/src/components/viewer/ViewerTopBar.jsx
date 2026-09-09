import { Link } from "react-router-dom";

function IconChevronLeft() {
    return (
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M15 18l-6-6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    );
}

function IconChevronRight() {
    return (
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    );
}

function IconPanelLeft() {
    return (
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="4" width="18" height="16" rx="2" stroke="currentColor" strokeWidth="1.6" />
            <path d="M9 4v16" stroke="currentColor" strokeWidth="1.6" />
        </svg>
    );
}

function IconPanelRight() {
    return (
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="4" width="18" height="16" rx="2" stroke="currentColor" strokeWidth="1.6" />
            <path d="M15 4v16" stroke="currentColor" strokeWidth="1.6" />
        </svg>
    );
}

function IconUndo() {
    return (
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 7 4 12l5 5M4 12h11a5 5 0 0 1 0 10h-1" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    );
}

function IconExport() {
    return (
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 15V4m0 11-4-4m4 4 4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    );
}

function ViewerTopBar({
    pdfName,
    page,
    pageCount,
    onPrevPage,
    onNextPage,
    navigationLocked,
    editMode,
    onToggleEditMode,
    addMode,
    onToggleAddMode,
    canUndo,
    onUndo,
    pendingCount,
    onDiscard,
    saving,
    onSaveAll,
    exportHref,
    sidebarOpen,
    onToggleSidebar,
    inspectorOpen,
    onToggleInspector,
}) {

    return (

        <header className="studio-topbar">

            <div className="studio-topbar-section studio-topbar-left">

                <button
                    type="button"
                    className={"studio-icon-btn" + (sidebarOpen ? " is-active" : "")}
                    onClick={onToggleSidebar}
                    title="Toggle page panel"
                    aria-label="Toggle page panel"
                    aria-pressed={sidebarOpen}
                >
                    <IconPanelLeft />
                </button>

                <Link to="/" className="studio-back-link" title="Back to documents">
                    <IconChevronLeft />
                </Link>

                <div className="studio-brand">
                    <span className="studio-brand-mark" aria-hidden="true">NA</span>
                    <div className="studio-brand-text">
                        <strong>Newspaper Archive Studio</strong>
                        <small>Boundary review &amp; extraction</small>
                    </div>
                </div>

            </div>

            <div className="studio-topbar-section studio-topbar-center">

                <span className="studio-doc-name" title={pdfName}>{pdfName}</span>

                <div className="studio-page-nav">

                    <button
                        type="button"
                        onClick={onPrevPage}
                        disabled={page <= 1 || navigationLocked}
                        aria-label="Previous page"
                        title="Previous page (Left arrow)"
                    >
                        <IconChevronLeft />
                    </button>

                    <span className="studio-page-nav-readout">{page} / {pageCount}</span>

                    <button
                        type="button"
                        onClick={onNextPage}
                        disabled={page >= pageCount || navigationLocked}
                        aria-label="Next page"
                        title="Next page (Right arrow)"
                    >
                        <IconChevronRight />
                    </button>

                </div>

            </div>

            <div className="studio-topbar-section studio-topbar-right">

                <div className="studio-toolbar-group">

                    <button
                        type="button"
                        className={"studio-btn studio-btn-ghost" + (editMode ? " is-active" : "")}
                        onClick={onToggleEditMode}
                        title="Select, drag, resize or delete existing boundaries"
                    >
                        {editMode ? "Done editing" : "Edit boundary"}
                    </button>

                    <button
                        type="button"
                        className={"studio-btn studio-btn-ghost" + (addMode ? " is-active is-danger" : "")}
                        onClick={onToggleAddMode}
                        title="Draw a new article boundary"
                    >
                        {addMode ? "Cancel drawing" : "+ Add boundary"}
                    </button>

                </div>

                <button
                    type="button"
                    className="studio-icon-btn"
                    onClick={onUndo}
                    disabled={!canUndo || saving}
                    title="Undo (Ctrl+Z)"
                    aria-label="Undo last change"
                >
                    <IconUndo />
                </button>

                <button
                    type="button"
                    className="studio-btn studio-btn-ghost"
                    onClick={onDiscard}
                    disabled={pendingCount === 0 || saving}
                    title="Discard all unsaved changes on this page"
                >
                    Discard
                </button>

                <button
                    type="button"
                    className="studio-btn studio-btn-primary"
                    onClick={onSaveAll}
                    disabled={pendingCount === 0 || saving}
                    title="Save all staged changes and re-extract affected articles"
                >
                    {saving ? "Saving…" : `Save & Extract${pendingCount ? ` (${pendingCount})` : ""}`}
                </button>

                <a
                    className="studio-icon-btn"
                    href={exportHref}
                    target="_blank"
                    rel="noreferrer"
                    title="Export page as PDF"
                    aria-label="Export page as PDF"
                >
                    <IconExport />
                </a>

                <button
                    type="button"
                    className={"studio-icon-btn" + (inspectorOpen ? " is-active" : "")}
                    onClick={onToggleInspector}
                    title="Toggle inspector panel"
                    aria-label="Toggle inspector panel"
                    aria-pressed={inspectorOpen}
                >
                    <IconPanelRight />
                </button>

            </div>

        </header>

    );

}

export default ViewerTopBar;

const SOURCE_LABELS = {
    final_verified_boundary: "Detected · verified",
    manual: "Manual",
};

function humanizeSource(source) {
    if (!source) {
        return "—";
    }
    return SOURCE_LABELS[source] || source.replace(/_/g, " ");
}

function InspectorRow({ label, value }) {
    return (
        <div className="studio-inspector-row">
            <dt>{label}</dt>
            <dd>{value}</dd>
        </div>
    );
}

function ArticleInspector({
    collapsed,
    page,
    pageCount,
    naturalSize,
    articleCount,
    pendingCount,
    documentLanguage,
    selectedItem,
    isEdited,
    editMode,
    saving,
    documentUuid,
    onEditBoundary,
    onRequestDelete,
    extractionResult,
}) {

    if (collapsed) {
        return null;
    }

    if (!selectedItem) {

        return (

            <aside className="studio-inspector">

                <div className="studio-inspector-header">
                    <span className="studio-inspector-eyebrow">Page overview</span>
                </div>

                <dl className="studio-inspector-body">
                    <InspectorRow label="Page" value={`${page} of ${pageCount}`} />
                    <InspectorRow
                        label="Dimensions"
                        value={naturalSize ? `${naturalSize.width} × ${naturalSize.height} px` : "—"}
                    />
                    <InspectorRow label="Articles on page" value={articleCount} />
                    <InspectorRow label="Document language" value={documentLanguage || "—"} />
                    <InspectorRow
                        label="Sync status"
                        value={pendingCount > 0 ? `${pendingCount} unsaved change${pendingCount === 1 ? "" : "s"}` : "All changes saved"}
                    />
                </dl>

                <p className="studio-inspector-empty-hint">
                    Select a boundary on the page to inspect article details.
                </p>

            </aside>

        );

    }

    const width = Math.round(selectedItem.bbox.x2 - selectedItem.bbox.x1);
    const height = Math.round(selectedItem.bbox.y2 - selectedItem.bbox.y1);

    const status = selectedItem.is_multi_page
        ? "Locked"
        : selectedItem.isNew
            ? "New"
            : isEdited
                ? "Edited"
                : "Saved";

    return (

        <aside className="studio-inspector">

            <div className="studio-inspector-header">
                <span className="studio-inspector-eyebrow">Article</span>
                <h3>{selectedItem.isNew ? "New boundary" : selectedItem.article_id}</h3>
            </div>

            <dl className="studio-inspector-body">
                <InspectorRow label="Status" value={<span className={`studio-status-chip status-${status.toLowerCase()}`}>{status}</span>} />
                <InspectorRow label="Page" value={page} />
                <InspectorRow label="Dimensions" value={`${width} × ${height} px`} />
                <InspectorRow label="Block count" value={selectedItem.block_count ?? 0} />
                <InspectorRow label="Boundary source" value={humanizeSource(selectedItem.boundary_source)} />
            </dl>

            {selectedItem.is_multi_page ? (

                <p className="studio-inspector-note">
                    This article continues across multiple pages, so its boundary can&rsquo;t be
                    edited or deleted here — merging continuation text safely requires the
                    automated pipeline, not a manual crop.
                </p>

            ) : (

                <div className="studio-inspector-actions">

                    <button
                        type="button"
                        className="studio-btn studio-btn-secondary"
                        onClick={onEditBoundary}
                        disabled={saving}
                    >
                        {editMode ? "Editing…" : "Edit boundary"}
                    </button>

                    <button
                        type="button"
                        className="studio-btn studio-btn-danger-ghost"
                        onClick={onRequestDelete}
                        disabled={saving}
                    >
                        Delete boundary
                    </button>

                </div>

            )}

            {selectedItem.logical_article_id && documentUuid && (

                <a
                    className="studio-inspector-link"
                    href={`/search/article/${documentUuid}/${selectedItem.logical_article_id}`}
                    target="_blank"
                    rel="noreferrer"
                >
                    View in archive →
                </a>

            )}

            {extractionResult && (

                <div className="studio-inspector-extraction">
                    <span className="studio-inspector-eyebrow">Last extraction</span>
                    {extractionResult.headline && (
                        <p className="studio-extraction-headline" dir="auto">{extractionResult.headline}</p>
                    )}
                    <p className="studio-extraction-text" dir="auto">
                        {extractionResult.article_text || "(no text extracted)"}
                    </p>
                </div>

            )}

        </aside>

    );

}

export default ArticleInspector;

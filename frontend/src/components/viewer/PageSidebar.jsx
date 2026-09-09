function PageThumbnail({ pageNumber, active, thumbSrc, boundaryCount, disabled, onSelect }) {

    return (

        <button
            type="button"
            className={"studio-page-thumb" + (active ? " is-active" : "")}
            onClick={() => onSelect(pageNumber)}
            disabled={disabled}
            aria-current={active}
            title={`Page ${pageNumber}`}
        >

            <span className="studio-page-thumb-frame">
                <img src={thumbSrc} alt="" loading="lazy" decoding="async" draggable={false} />
            </span>

            <span className="studio-page-thumb-label">
                <span className="studio-page-thumb-number">Page {pageNumber}</span>
                <span className="studio-page-thumb-meta">
                    {boundaryCount === undefined ? "—" : `${boundaryCount} article${boundaryCount === 1 ? "" : "s"}`}
                </span>
            </span>

        </button>

    );

}

function PageSidebar({
    collapsed,
    pdfName,
    documentId,
    page,
    pageCount,
    pageBoundaryCounts,
    navigationLocked,
    onSelectPage,
    thumbBaseUrl,
}) {

    if (collapsed) {
        return null;
    }

    const pages = Array.from({ length: pageCount || 0 }, (_, index) => index + 1);

    return (

        <aside className="studio-sidebar">

            <div className="studio-sidebar-section">
                <span className="studio-sidebar-label">Document</span>
                <p className="studio-sidebar-filename" title={pdfName}>{pdfName}</p>
            </div>

            <div className="studio-sidebar-section studio-sidebar-pages">

                <span className="studio-sidebar-label">Pages</span>

                <div className="studio-page-list">

                    {pages.map((pageNumber) => (

                        <PageThumbnail
                            key={pageNumber}
                            pageNumber={pageNumber}
                            active={pageNumber === page}
                            thumbSrc={`${thumbBaseUrl}/documents/${documentId}/page/${pageNumber}/thumbnail`}
                            boundaryCount={pageBoundaryCounts[pageNumber]}
                            disabled={navigationLocked && pageNumber !== page}
                            onSelect={onSelectPage}
                        />

                    ))}

                </div>

            </div>

        </aside>

    );

}

export default PageSidebar;

function StatusBar({ saveState, pendingCount, saveProgress, selectedItem, naturalSize, scalePercent }) {

    const stateText = {
        saving: saveProgress ? `Saving ${saveProgress.done}/${saveProgress.total}…` : "Saving…",
        unsaved: `${pendingCount} unsaved change${pendingCount === 1 ? "" : "s"}`,
        error: "Some changes failed to save",
        saved: "All changes saved",
    }[saveState];

    return (

        <footer className="studio-statusbar">

            <div className="studio-statusbar-section">
                <span className={`studio-status-dot status-${saveState}`} aria-hidden="true" />
                <span>{stateText}</span>
            </div>

            <div className="studio-statusbar-section studio-statusbar-center">
                {selectedItem
                    ? `Selected: ${selectedItem.isNew ? "New boundary" : selectedItem.article_id}`
                    : "No article selected"}
            </div>

            <div className="studio-statusbar-section studio-statusbar-right">
                {naturalSize && <span>{naturalSize.width} × {naturalSize.height}px</span>}
                <span>{scalePercent}%</span>
            </div>

        </footer>

    );

}

export default StatusBar;

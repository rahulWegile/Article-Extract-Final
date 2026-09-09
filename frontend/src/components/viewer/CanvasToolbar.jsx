function CanvasToolbar({ scalePercent, onZoomOut, onZoomIn, onFit, onFitWidth, onReset }) {

    return (

        <div className="studio-canvas-toolbar" role="toolbar" aria-label="Zoom controls">

            <button type="button" onClick={onZoomOut} title="Zoom out (-)" aria-label="Zoom out">−</button>

            <span className="studio-zoom-readout">{scalePercent}%</span>

            <button type="button" onClick={onZoomIn} title="Zoom in (+)" aria-label="Zoom in">+</button>

            <span className="studio-toolbar-divider" aria-hidden="true" />

            <button type="button" onClick={onFit} title="Fit page (F)" aria-label="Fit page to screen">Fit</button>

            <button type="button" onClick={onFitWidth} title="Fit width (W)" aria-label="Fit page width">Fit width</button>

            <button type="button" onClick={onReset} title="Reset zoom (0)" aria-label="Reset zoom">Reset</button>

        </div>

    );

}

export default CanvasToolbar;

import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import api from "../api/api";
import {
    TransformWrapper,
    TransformComponent,
} from "react-zoom-pan-pinch";

const MIN_BOX_SIZE = 10;

// The image is already sized to fit the viewer at scale 1 (via
// object-fit: contain on .boundary-image), so anything below 1 would
// just shrink it smaller than the viewer -- there's no reason to zoom
// out past the fit size for a single page.
const MIN_SCALE = 1;

const MAX_SCALE = 10;
const MAX_HISTORY = 100;

function clampBox(bbox, naturalSize) {
    return {
        x1: Math.max(0, Math.min(bbox.x1, naturalSize.width)),
        y1: Math.max(0, Math.min(bbox.y1, naturalSize.height)),
        x2: Math.max(0, Math.min(bbox.x2, naturalSize.width)),
        y2: Math.max(0, Math.min(bbox.y2, naturalSize.height)),
    };
}

function normalizeBox(start, current, naturalSize) {
    return clampBox(
        {
            x1: Math.min(start.x, current.x),
            y1: Math.min(start.y, current.y),
            x2: Math.max(start.x, current.x),
            y2: Math.max(start.y, current.y),
        },
        naturalSize
    );
}

function resizeBox(origBbox, handle, current, naturalSize) {
    const point = clampBox(
        { x1: current.x, y1: current.y, x2: current.x, y2: current.y },
        naturalSize
    );

    const box = { ...origBbox };

    if (handle === "nw") {
        box.x1 = Math.min(point.x1, origBbox.x2 - MIN_BOX_SIZE);
        box.y1 = Math.min(point.y1, origBbox.y2 - MIN_BOX_SIZE);
    } else if (handle === "ne") {
        box.x2 = Math.max(point.x1, origBbox.x1 + MIN_BOX_SIZE);
        box.y1 = Math.min(point.y1, origBbox.y2 - MIN_BOX_SIZE);
    } else if (handle === "sw") {
        box.x1 = Math.min(point.x1, origBbox.x2 - MIN_BOX_SIZE);
        box.y2 = Math.max(point.y1, origBbox.y1 + MIN_BOX_SIZE);
    } else if (handle === "se") {
        box.x2 = Math.max(point.x1, origBbox.x1 + MIN_BOX_SIZE);
        box.y2 = Math.max(point.y1, origBbox.y1 + MIN_BOX_SIZE);
    }

    return clampBox(box, naturalSize);
}

function roundBbox(bbox) {
    return {
        x1: Math.round(bbox.x1),
        y1: Math.round(bbox.y1),
        x2: Math.round(bbox.x2),
        y2: Math.round(bbox.y2),
    };
}

function isBoxValid(bbox) {
    return (
        bbox.x2 - bbox.x1 >= MIN_BOX_SIZE &&
        bbox.y2 - bbox.y1 >= MIN_BOX_SIZE
    );
}

function bboxEqual(a, b) {
    return (
        a.x1 === b.x1 &&
        a.y1 === b.y1 &&
        a.x2 === b.x2 &&
        a.y2 === b.y2
    );
}

function toDraftItem(boundary) {
    return {
        key: boundary.article_id,
        article_id: boundary.article_id,
        bbox: { ...boundary.bbox },
        is_multi_page: boundary.is_multi_page,
        isNew: false,
    };
}

// Diffs the working draft against the last-loaded-from-server state so a
// single "Save & extract" can batch every staged create/edit/delete
// instead of round-tripping to the backend (and re-running OpenAI
// extraction) after every single drag.
function diffBoundaries(original, draft) {

    const originalByArticleId = new Map(
        original.map((boundary) => [boundary.article_id, boundary])
    );

    const draftArticleIds = new Set(
        draft.filter((item) => !item.isNew).map((item) => item.article_id)
    );

    const creates = draft.filter((item) => item.isNew);

    const edits = draft.filter((item) => {

        if (item.isNew) {
            return false;
        }

        const original_ = originalByArticleId.get(item.article_id);

        return Boolean(original_) && !bboxEqual(original_.bbox, item.bbox);

    });

    const deletes = original.filter(
        (boundary) => !draftArticleIds.has(boundary.article_id)
    );

    return { creates, edits, deletes };

}

function Viewer() {

    const { documentId } = useParams();

    const [pageData, setPageData] = useState(null);
    const [page, setPage] = useState(1);

    const [originalBoundaries, setOriginalBoundaries] = useState([]);
    const [draftBoundaries, setDraftBoundaries] = useState([]);
    const [naturalSize, setNaturalSize] = useState(null);

    const [addMode, setAddMode] = useState(false);
    const [editMode, setEditMode] = useState(false);
    const [selectedKey, setSelectedKey] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [canUndo, setCanUndo] = useState(false);

    const [saving, setSaving] = useState(false);
    const [saveProgress, setSaveProgress] = useState(null);
    const [resultSummary, setResultSummary] = useState(null);

    // TransformWrapper drives pan/zoom imperatively for performance and
    // doesn't re-render its render-prop children on every change, so the
    // slider needs its own state kept in sync via onTransform below.
    const [transform, setTransformState] = useState({
        scale: 1,
        positionX: 0,
        positionY: 0,
    });

    const svgRef = useRef(null);
    const dragRef = useRef(null);
    const historyRef = useRef([]);
    const newKeyCounterRef = useRef(0);

    const { creates, edits, deletes } = diffBoundaries(
        originalBoundaries,
        draftBoundaries
    );

    const pendingCount = creates.length + edits.length + deletes.length;
    const navigationLocked = saving || pendingCount > 0;

    const editedArticleIds = new Set(edits.map((item) => item.article_id));

    const toSvgPoint = (clientX, clientY) => {

        const svg = svgRef.current;

        if (!svg) {
            return { x: 0, y: 0 };
        }

        const ctm = svg.getScreenCTM();

        if (!ctm) {
            return { x: 0, y: 0 };
        }

        const point = svg.createSVGPoint();
        point.x = clientX;
        point.y = clientY;

        const transformed = point.matrixTransform(ctm.inverse());

        return { x: transformed.x, y: transformed.y };

    };

    const pushHistory = (snapshot) => {

        historyRef.current = [
            ...historyRef.current,
            snapshot,
        ].slice(-MAX_HISTORY);

        setCanUndo(true);

    };

    const loadPage = async (pageNumber) => {

        try {

            const response = await api.get(
                `/documents/${documentId}/page/${pageNumber}`
            );

            setPageData(response.data);

        }

        catch (error) {

            console.error(error);

        }

    };

    const loadBoundaries = async (pageNumber) => {

        try {

            const response = await api.get(
                `/documents/${documentId}/page/${pageNumber}/boundaries`
            );

            const list = response.data.boundaries || [];

            setOriginalBoundaries(list);
            setDraftBoundaries(list.map(toDraftItem));

        }

        catch (error) {

            console.error(error);

            setOriginalBoundaries([]);
            setDraftBoundaries([]);

        }

    };

    useEffect(() => {

        loadPage(page);
        loadBoundaries(page);

        setNaturalSize(null);
        setAddMode(false);
        setEditMode(false);
        setSelectedKey(null);
        setResultSummary(null);

        historyRef.current = [];
        setCanUndo(false);

    }, [page]);

    const handleUndo = () => {

        if (historyRef.current.length === 0) {
            return;
        }

        const previous = historyRef.current[historyRef.current.length - 1];

        historyRef.current = historyRef.current.slice(0, -1);
        setCanUndo(historyRef.current.length > 0);

        setDraftBoundaries(previous);
        setSelectedKey(null);

    };

    useEffect(() => {

        const handleKeyDown = (event) => {

            if (["INPUT", "TEXTAREA"].includes(event.target.tagName)) {
                return;
            }

            const isUndoShortcut =
                (event.ctrlKey || event.metaKey) &&
                !event.shiftKey &&
                event.key.toLowerCase() === "z";

            if (isUndoShortcut) {
                event.preventDefault();
                handleUndo();
                return;
            }

            if (navigationLocked) {
                return;
            }

            if (event.key === "ArrowLeft" && page > 1) {
                setPage((current) => current - 1);
            }

            if (
                event.key === "ArrowRight" &&
                pageData &&
                page < pageData.page_count
            ) {
                setPage((current) => current + 1);
            }

        };

        window.addEventListener("keydown", handleKeyDown);

        return () => window.removeEventListener("keydown", handleKeyDown);

    }, [page, pageData, navigationLocked]);

    useEffect(() => {

        if (!isDragging) {
            return undefined;
        }

        const handlePointerMove = (event) => {

            const drag = dragRef.current;

            if (!drag || !naturalSize) {
                return;
            }

            const current = toSvgPoint(event.clientX, event.clientY);

            if (drag.type === "draw") {

                const bbox = normalizeBox(drag.start, current, naturalSize);

                setDraftBoundaries((prev) => {

                    if (!drag.historyPushed) {
                        pushHistory(prev);
                        drag.historyPushed = true;
                    }

                    const withoutDraft = prev.filter(
                        (item) => item.key !== drag.newKey
                    );

                    return [
                        ...withoutDraft,
                        {
                            key: drag.newKey,
                            article_id: null,
                            bbox,
                            is_multi_page: false,
                            isNew: true,
                        },
                    ];

                });

                setSelectedKey(drag.newKey);

            } else if (drag.type === "move") {

                const dx = current.x - drag.start.x;
                const dy = current.y - drag.start.y;

                const bbox = clampBox(
                    {
                        x1: drag.origBbox.x1 + dx,
                        y1: drag.origBbox.y1 + dy,
                        x2: drag.origBbox.x2 + dx,
                        y2: drag.origBbox.y2 + dy,
                    },
                    naturalSize
                );

                setDraftBoundaries((prev) => {

                    if (!drag.historyPushed) {
                        pushHistory(prev);
                        drag.historyPushed = true;
                    }

                    return prev.map((item) => (
                        item.key === drag.key ? { ...item, bbox } : item
                    ));

                });

            } else if (drag.type === "resize") {

                const bbox = resizeBox(drag.origBbox, drag.handle, current, naturalSize);

                setDraftBoundaries((prev) => {

                    if (!drag.historyPushed) {
                        pushHistory(prev);
                        drag.historyPushed = true;
                    }

                    return prev.map((item) => (
                        item.key === drag.key ? { ...item, bbox } : item
                    ));

                });

            }

        };

        const handlePointerUp = () => {

            const drag = dragRef.current;

            if (drag && drag.type === "draw") {

                setDraftBoundaries((prev) => {

                    const drawn = prev.find((item) => item.key === drag.newKey);

                    if (drawn && !isBoxValid(drawn.bbox)) {
                        return prev.filter((item) => item.key !== drag.newKey);
                    }

                    return prev;

                });

            }

            dragRef.current = null;
            setIsDragging(false);

        };

        window.addEventListener("pointermove", handlePointerMove);
        window.addEventListener("pointerup", handlePointerUp);

        return () => {
            window.removeEventListener("pointermove", handlePointerMove);
            window.removeEventListener("pointerup", handlePointerUp);
        };

    }, [isDragging, naturalSize]);

    const handleImageLoad = (event) => {

        setNaturalSize({
            width: event.target.naturalWidth,
            height: event.target.naturalHeight,
        });

    };

    const handleBackgroundPointerDown = (event) => {

        if (!addMode) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        newKeyCounterRef.current += 1;

        dragRef.current = {
            type: "draw",
            start: toSvgPoint(event.clientX, event.clientY),
            newKey: `new-${newKeyCounterRef.current}`,
            historyPushed: false,
        };

        setIsDragging(true);

    };

    const handleBoxPointerDown = (item) => (event) => {

        if (!editMode || addMode || item.is_multi_page) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        setSelectedKey(item.key);

        dragRef.current = {
            type: "move",
            start: toSvgPoint(event.clientX, event.clientY),
            origBbox: item.bbox,
            key: item.key,
            historyPushed: false,
        };

        setIsDragging(true);

    };

    const handleResizePointerDown = (item, handle) => (event) => {

        event.preventDefault();
        event.stopPropagation();

        dragRef.current = {
            type: "resize",
            handle,
            origBbox: item.bbox,
            key: item.key,
            historyPushed: false,
        };

        setIsDragging(true);

    };

    const handleToggleAddMode = () => {

        setAddMode((current) => !current);
        setEditMode(false);
        setSelectedKey(null);

    };

    const handleToggleEditMode = () => {

        setEditMode((current) => !current);
        setAddMode(false);
        setSelectedKey(null);

    };

    const handleDeleteSelected = () => {

        if (!selectedKey) {
            return;
        }

        pushHistory(draftBoundaries);

        setDraftBoundaries((prev) => (
            prev.filter((item) => item.key !== selectedKey)
        ));

        setSelectedKey(null);

    };

    const handleDiscardAll = () => {

        setDraftBoundaries(originalBoundaries.map(toDraftItem));
        historyRef.current = [];
        setCanUndo(false);
        setSelectedKey(null);

    };

    const handleSaveAll = async () => {

        if (pendingCount === 0) {
            return;
        }

        setSaving(true);
        setSaveProgress({ done: 0, total: pendingCount });

        const results = [];
        let completed = 0;

        const advance = () => {
            completed += 1;
            setSaveProgress({ done: completed, total: pendingCount });
        };

        for (const item of deletes) {

            try {

                const response = await api.delete(
                    `/documents/${documentId}/page/${page}/boundaries/${item.article_id}`
                );

                results.push({
                    article_id: item.article_id,
                    action: "deleted",
                    db_synced: response.data.db_synced,
                    db_error: response.data.db_error,
                });

            }

            catch (error) {

                results.push({
                    article_id: item.article_id,
                    action: "delete-failed",
                    error: error?.response?.data?.detail || "Failed to delete.",
                });

            }

            advance();

        }

        for (const item of edits) {

            try {

                const response = await api.post(
                    `/documents/${documentId}/page/${page}/boundaries`,
                    {
                        article_id: item.article_id,
                        bbox: roundBbox(item.bbox),
                    }
                );

                results.push({
                    article_id: item.article_id,
                    action: "edited",
                    headline: response.data.headline,
                    article_text: response.data.article_text,
                    db_synced: response.data.db_synced,
                    db_error: response.data.db_error,
                });

            }

            catch (error) {

                results.push({
                    article_id: item.article_id,
                    action: "edit-failed",
                    error: error?.response?.data?.detail || "Failed to save.",
                });

            }

            advance();

        }

        for (const item of creates) {

            try {

                const response = await api.post(
                    `/documents/${documentId}/page/${page}/boundaries`,
                    {
                        article_id: null,
                        bbox: roundBbox(item.bbox),
                    }
                );

                results.push({
                    article_id: response.data.article_id,
                    action: "created",
                    headline: response.data.headline,
                    article_text: response.data.article_text,
                    db_synced: response.data.db_synced,
                    db_error: response.data.db_error,
                });

            }

            catch (error) {

                results.push({
                    article_id: null,
                    action: "create-failed",
                    error: error?.response?.data?.detail || "Failed to create.",
                });

            }

            advance();

        }

        await loadBoundaries(page);

        historyRef.current = [];
        setCanUndo(false);
        setSelectedKey(null);
        setResultSummary(results);
        setSaving(false);
        setSaveProgress(null);

    };

    if (!pageData) {

        return (

            <div className="viewer-loading">

                <div className="viewer-spinner" />

                <p>Loading page...</p>

            </div>

        );

    }

    const handleSize = naturalSize
        ? Math.max(18, naturalSize.width / 120)
        : 20;

    const selectedItem = draftBoundaries.find((item) => item.key === selectedKey);

    return (

        <div className="viewer-container">

            <div className="viewer-header">

                <Link
                    to="/"
                    className="back-button"
                >
                    ← Back
                </Link>

                <h2>

                    {pageData.pdf_name}

                </h2>

                <div className="page-badge">

                    Page {pageData.page} of {pageData.page_count}

                </div>

                <button
                    type="button"
                    className={
                        "edit-boundary-toggle" + (editMode ? " active" : "")
                    }
                    onClick={handleToggleEditMode}
                    title="Select and drag/resize/delete existing boundaries"
                >
                    {editMode ? "Done editing" : "✎ Edit boundary"}
                </button>

                <button
                    type="button"
                    className={
                        "add-boundary-toggle" + (addMode ? " active" : "")
                    }
                    onClick={handleToggleAddMode}
                    title="Draw a new article boundary"
                >
                    {addMode ? "Cancel drawing" : "+ Add boundary"}
                </button>

                <a
                    className="save-pdf-button"
                    href={`${import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"}/documents/${documentId}/export/pdf`}
                    target="_blank"
                    rel="noreferrer"
                >
                    Save as PDF
                </a>

            </div>

            <div className="image-container">

                <TransformWrapper

                    initialScale={1}

                    minScale={MIN_SCALE}

                    maxScale={MAX_SCALE}

                    centerOnInit={true}

                    centerZoomedOut={true}

                    // Keeps the page from being panned completely off-screen
                    // (into blank space) at any zoom level.
                    limitToBounds={true}

                    wheel={{
                        step: 0.15,

                        // Two-finger trackpad scroll fires the same
                        // wheel event as a mouse wheel, so it would
                        // zoom by default. Disable that and let
                        // trackPadPanning below turn it into page
                        // scrolling instead. Pinch-to-zoom (which
                        // browsers report as ctrl+wheel) still zooms.
                        wheelDisabled: true,
                    }}

                    trackPadPanning={{
                        disabled: addMode,
                    }}

                    doubleClick={{
                        disabled: false,
                    }}

                    panning={{
                        disabled: addMode,
                        velocityDisabled: false,
                    }}

                    pinch={{
                        disabled: false,
                    }}

                    alignmentAnimation={{
                        disabled: true,
                    }}

                    onTransform={(_ref, state) =>
                        setTransformState(state)
                    }

                >

                    {({ zoomIn, zoomOut, resetTransform, centerView }) => (

                        <>

                            <div className="zoom-toolbar">

                                <button title="Zoom out" onClick={() => zoomOut()}>
                                    −
                                </button>

                                <input
                                    className="zoom-slider"
                                    type="range"
                                    title="Zoom"
                                    min={MIN_SCALE}
                                    max={MAX_SCALE}
                                    step={0.1}
                                    value={transform.scale}
                                    onChange={(event) =>
                                        // centerView (rather than setTransform) recomputes a
                                        // centered position for the new scale, so the page can
                                        // never end up scaled into a corner or off-screen after
                                        // panning around at a different zoom level.
                                        centerView(
                                            Number(event.target.value),
                                            0,
                                        )
                                    }
                                />

                                <button title="Zoom in" onClick={() => zoomIn()}>
                                    +
                                </button>

                                <button title="Reset zoom" onClick={() => resetTransform()}>
                                    Reset
                                </button>

                            </div>

                            <TransformComponent
                                wrapperStyle={{
                                    width: "100%",
                                    height: "100%",
                                }}
                                contentStyle={{
                                    width: "100%",
                                    height: "100%",
                                    display: "flex",
                                    justifyContent: "center",
                                    alignItems: "center",
                                }}
                            >

                                <div className="boundary-editor-frame">

                                    <img
                                        src={`${import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"}${pageData.plain_image}`}
                                        alt="Page"
                                        className="boundary-image"
                                        onLoad={handleImageLoad}
                                        draggable={false}
                                    />

                                    {naturalSize && (

                                        <svg
                                            ref={svgRef}
                                            className={
                                                "boundary-overlay-svg" +
                                                (addMode ? " add-mode" : "")
                                            }
                                            viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
                                            preserveAspectRatio="xMidYMid meet"
                                        >

                                            {addMode && (

                                                <rect
                                                    x={0}
                                                    y={0}
                                                    width={naturalSize.width}
                                                    height={naturalSize.height}
                                                    className="boundary-draw-catcher"
                                                    onPointerDown={handleBackgroundPointerDown}
                                                />

                                            )}

                                            {draftBoundaries.map((item) => (

                                                <rect
                                                    key={item.key}
                                                    x={item.bbox.x1}
                                                    y={item.bbox.y1}
                                                    width={item.bbox.x2 - item.bbox.x1}
                                                    height={item.bbox.y2 - item.bbox.y1}
                                                    className={
                                                        "boundary-box" +
                                                        // (item.is_multi_page ? " multi-page" : "") +
                                                        (!editMode || addMode ? " not-editable" : "") +
                                                        (item.key === selectedKey ? " selected" : "") +
                                                        (item.isNew ? " pending-new" : "") +
                                                        (editedArticleIds.has(item.article_id) ? " pending-edited" : "")
                                                    }
                                                    onPointerDown={
                                                        editMode && !addMode && !item.is_multi_page
                                                            ? handleBoxPointerDown(item)
                                                            : undefined
                                                    }
                                                >

                                                    {item.is_multi_page && (
                                                        <title>
                                                            Spans multiple pages — not editable here
                                                        </title>
                                                    )}

                                                </rect>

                                            ))}

                                            {editMode && selectedItem && !selectedItem.is_multi_page && (

                                                ["nw", "ne", "sw", "se"].map((handle) => (

                                                    <rect
                                                        key={handle}
                                                        x={
                                                            (handle === "nw" || handle === "sw"
                                                                ? selectedItem.bbox.x1
                                                                : selectedItem.bbox.x2) - handleSize / 2
                                                        }
                                                        y={
                                                            (handle === "nw" || handle === "ne"
                                                                ? selectedItem.bbox.y1
                                                                : selectedItem.bbox.y2) - handleSize / 2
                                                        }
                                                        width={handleSize}
                                                        height={handleSize}
                                                        className={`boundary-handle handle-${handle}`}
                                                        onPointerDown={handleResizePointerDown(selectedItem, handle)}
                                                    />

                                                ))

                                            )}

                                        </svg>

                                    )}

                                </div>

                            </TransformComponent>

                        </>

                    )}

                </TransformWrapper>

                {addMode && (

                    <div className="boundary-hint">
                        Drag on the page to draw a new article boundary.
                    </div>

                )}

                {editMode && !selectedItem && (

                    <div className="boundary-hint">
                        Click a boundary to select it, then drag to move
                        or use the corner handles to resize.
                    </div>

                )}

                {editMode && selectedItem && !selectedItem.is_multi_page && (

                    <div className="boundary-select-panel">

                        <p className="boundary-edit-title">
                            {selectedItem.isNew
                                ? "New boundary"
                                : selectedItem.article_id}
                        </p>

                        <button
                            type="button"
                            className="boundary-delete-button"
                            onClick={handleDeleteSelected}
                            disabled={saving}
                        >
                            Delete boundary
                        </button>

                    </div>

                )}

                {resultSummary && (

                    <div className="boundary-result-panel">

                        <div className="boundary-result-header">

                            <strong>
                                Save results
                            </strong>

                            <button
                                type="button"
                                className="boundary-result-close"
                                onClick={() => setResultSummary(null)}
                            >
                                ×
                            </button>

                        </div>

                        {resultSummary.map((result, index) => (

                            <div
                                key={`${result.article_id || "new"}-${index}`}
                                className={
                                    "boundary-summary-item" +
                                    (result.error ? " failed" : "")
                                }
                            >

                                <p className="boundary-summary-title">
                                    {result.article_id || "(new article)"} — {result.action}
                                </p>

                                {result.error && (
                                    <p className="boundary-summary-error">
                                        {result.error}
                                    </p>
                                )}

                                {result.db_synced === false && (
                                    <p className="boundary-edit-warning">
                                        Saved on disk, but the database sync failed
                                        {result.db_error ? `: ${result.db_error}` : "."}
                                    </p>
                                )}

                                {result.headline && (
                                    <p className="boundary-result-headline">
                                        {result.headline}
                                    </p>
                                )}

                                {"article_text" in result && (
                                    <p className="boundary-result-text">
                                        {result.article_text || "(no text extracted)"}
                                    </p>
                                )}

                            </div>

                        ))}

                    </div>

                )}

            </div>

            {/* Lives outside .image-container (in normal document flow,
                not overlaid on top of it) so it never covers part of the
                page -- a page's last articles are often near the bottom
                edge, right where this bar would otherwise sit. */}
            <div className="boundary-toolbar">

                <span className="boundary-pending-count">
                    {pendingCount > 0
                        ? `${pendingCount} unsaved change${pendingCount === 1 ? "" : "s"}`
                        : "No unsaved changes"}
                </span>

                <button
                    type="button"
                    className="boundary-undo-button"
                    onClick={handleUndo}
                    disabled={!canUndo || saving}
                    title="Undo (Ctrl+Z)"
                >
                    ↶ Undo
                </button>

                <button
                    type="button"
                    className="boundary-discard-button"
                    onClick={handleDiscardAll}
                    disabled={pendingCount === 0 || saving}
                >
                    Discard changes
                </button>

                <button
                    type="button"
                    className="boundary-save-all-button"
                    onClick={handleSaveAll}
                    disabled={pendingCount === 0 || saving}
                >
                    {saving
                        ? `Saving ${saveProgress ? saveProgress.done : 0}/${saveProgress ? saveProgress.total : pendingCount}…`
                        : `Save & extract (${pendingCount})`}
                </button>

            </div>

            <div className="viewer-navigation">

                <button

                    disabled={page <= 1 || navigationLocked}

                    onClick={() => setPage(page - 1)}

                >

                    ⬅ Previous

                </button>

                <span className="nav-page-indicator">

                    {page} / {pageData.page_count}

                </span>

                <button

                    disabled={page >= pageData.page_count || navigationLocked}

                    onClick={() => setPage(page + 1)}

                >

                    Next ➡

                </button>

            </div>

        </div>

    );

}

export default Viewer;

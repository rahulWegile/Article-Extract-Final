import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import api from "../api/api";
import {
    TransformWrapper,
    TransformComponent,
} from "react-zoom-pan-pinch";

import ViewerTopBar from "../components/viewer/ViewerTopBar";
import PageSidebar from "../components/viewer/PageSidebar";
import CanvasToolbar from "../components/viewer/CanvasToolbar";
import BoundaryLayer from "../components/viewer/BoundaryLayer";
import ArticleInspector from "../components/viewer/ArticleInspector";
import StatusBar from "../components/viewer/StatusBar";
import ConfirmDialog from "../components/ConfirmDialog";
import ToastStack from "../components/ToastStack";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const MIN_BOX_SIZE = 10;
const MIN_SCALE = 1;
const MAX_SCALE = 10;
const MAX_HISTORY = 100;
const TOAST_LIFETIME_MS = 5000;
const DOCUMENT_STATUS_POLL_MS = 4000;

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
    return a.x1 === b.x1 && a.y1 === b.y1 && a.x2 === b.x2 && a.y2 === b.y2;
}

function toDraftItem(boundary) {
    return {
        key: boundary.article_id,
        article_id: boundary.article_id,
        bbox: { ...boundary.bbox },
        is_multi_page: boundary.is_multi_page,
        isNew: false,
        sub_rects: (boundary.sub_rects || []).map((r) => ({ ...r })),
        block_count: boundary.block_count || 0,
        boundary_source: boundary.boundary_source || null,
        logical_article_id: boundary.logical_article_id || null,
    };
}

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

function getDefaultPanelState() {

    if (typeof window === "undefined") {
        return { sidebar: true, inspector: true };
    }

    return {
        sidebar: window.innerWidth > 1024,
        inspector: window.innerWidth > 1280,
    };

}

function Viewer() {

    const { documentId } = useParams();

    const [pageData, setPageData] = useState(null);
    const [documentDetail, setDocumentDetail] = useState(null);
    const [documentUuid, setDocumentUuid] = useState(null);
    const [page, setPage] = useState(1);

    const [originalBoundaries, setOriginalBoundaries] = useState([]);
    const [draftBoundaries, setDraftBoundaries] = useState([]);
    const [naturalSize, setNaturalSize] = useState(null);
    const [pageBoundaryCounts, setPageBoundaryCounts] = useState({});

    const [addMode, setAddMode] = useState(false);
    const [editMode, setEditMode] = useState(false);
    const [selectedKey, setSelectedKey] = useState(null);
    const [hoveredKey, setHoveredKey] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [canUndo, setCanUndo] = useState(false);

    const [saving, setSaving] = useState(false);
    const [saveProgress, setSaveProgress] = useState(null);
    const [lastSaveHadError, setLastSaveHadError] = useState(false);
    const [extractionResults, setExtractionResults] = useState({});

    const [confirmDialog, setConfirmDialog] = useState(null);
    const [toasts, setToasts] = useState([]);

    const [sidebarOpen, setSidebarOpen] = useState(() => getDefaultPanelState().sidebar);
    const [inspectorOpen, setInspectorOpen] = useState(() => getDefaultPanelState().inspector);

    const [transform, setTransformState] = useState({
        scale: 1,
        positionX: 0,
        positionY: 0,
    });

    const svgRef = useRef(null);
    const dragRef = useRef(null);
    const historyRef = useRef([]);
    const newKeyCounterRef = useRef(0);
    const transformRef = useRef(null);
    const canvasStageRef = useRef(null);
    const toastCounterRef = useRef(0);
    const failedToastShownRef = useRef(false);

    const { creates, edits, deletes } = diffBoundaries(
        originalBoundaries,
        draftBoundaries
    );

    const pendingCount = creates.length + edits.length + deletes.length;
    const navigationLocked = saving || pendingCount > 0;

    const editedArticleIds = new Set(edits.map((item) => item.article_id));

    const dismissToast = (id) => {
        setToasts((current) => current.filter((toast) => toast.id !== id));
    };

    const pushToast = (type, text) => {
        toastCounterRef.current += 1;
        const id = toastCounterRef.current;
        setToasts((current) => [...current, { id, type, text }]);
        setTimeout(() => dismissToast(id), TOAST_LIFETIME_MS);
    };

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
        historyRef.current = [...historyRef.current, snapshot].slice(-MAX_HISTORY);
        setCanUndo(true);
    };

    const loadPage = async (pageNumber) => {
        try {
            const response = await api.get(`/documents/${documentId}/page/${pageNumber}`);
            setPageData(response.data);
        } catch (error) {
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
            setPageBoundaryCounts((prev) => ({ ...prev, [pageNumber]: list.length }));
            setDocumentUuid(response.data.document_uuid || null);

        } catch (error) {

            console.error(error);
            setOriginalBoundaries([]);
            setDraftBoundaries([]);

        }
    };

    useEffect(() => {

        setPage(1);

    }, [documentId]);

    useEffect(() => {

        setPageBoundaryCounts({});
        setDocumentDetail(null);

        let cancelled = false;
        let pollTimer = null;

        const fetchDetail = () => {

            api.get(`/documents/${documentId}`)
                .then((response) => {

                    if (cancelled) {
                        return;
                    }

                    const detail = response.data;

                    setDocumentDetail(detail);

                    const status = detail?.status;

                    // Still being processed (no terminal status yet) --
                    // keep checking in the background so the viewer
                    // opens automatically the moment it's ready, instead
                    // of the user having to manually refresh.
                    if (status && status !== "completed" && status !== "failed") {
                        pollTimer = setTimeout(fetchDetail, DOCUMENT_STATUS_POLL_MS);
                    }

                })
                .catch(() => {
                    if (!cancelled) {
                        setDocumentDetail(null);
                    }
                });

        };

        fetchDetail();

        return () => {
            cancelled = true;
            if (pollTimer) {
                clearTimeout(pollTimer);
            }
        };

    }, [documentId]);

    const documentStatus = documentDetail?.status;

    // Boundary editing only needs page images + article crops, both
    // already on disk once `boundaries_ready` is set (before article
    // extraction/batching and finalization have run) -- no need to
    // wait for the full pipeline (`status === "completed"`) just to
    // open and edit boundaries.
    const documentReady =
        documentStatus === "completed" || documentDetail?.boundaries_ready === true;

    useEffect(() => {

        // The failed screen itself only ever shows a generic message
        // (see below) -- the actual reason (e.g. a provider quota/rate
        // limit error) is surfaced as a toast instead, once per
        // failure, rather than left permanently baked into the page.
        if (documentStatus === "failed" && !failedToastShownRef.current) {
            failedToastShownRef.current = true;
            pushToast(
                "error",
                documentDetail?.error?.message ||
                    "This document could not be processed."
            );
        }

    }, [documentStatus, documentDetail]);

    useEffect(() => {

        // Page images and boundaries don't exist on disk until
        // `documentReady` (see above), so fetching them any earlier
        // just surfaces broken images/404s -- wait for it instead
        // (see the wait/failed screens rendered below).
        if (!documentReady) {
            return;
        }

        loadPage(page);
        loadBoundaries(page);

        setNaturalSize(null);
        setAddMode(false);
        setEditMode(false);
        setSelectedKey(null);
        setLastSaveHadError(false);

        historyRef.current = [];
        setCanUndo(false);

    }, [documentId, page, documentReady]);

    const handleZoomIn = useCallback(() => transformRef.current?.zoomIn(0.25), []);
    const handleZoomOut = useCallback(() => transformRef.current?.zoomOut(0.25), []);
    const handleResetZoom = useCallback(() => transformRef.current?.resetTransform(), []);

    const handleFitWidth = useCallback(() => {

        const stage = canvasStageRef.current;
        const controls = transformRef.current;

        if (!stage || !naturalSize || !controls) {
            return;
        }

        const stageW = stage.clientWidth;
        const stageH = stage.clientHeight;

        const containScale = Math.min(stageW / naturalSize.width, stageH / naturalSize.height);
        const widthScale = (stageW / naturalSize.width) / containScale;

        controls.centerView(Math.max(MIN_SCALE, widthScale), 200);

    }, [naturalSize]);

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

            if (confirmDialog) {
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

            if (!event.ctrlKey && !event.metaKey && !event.altKey) {

                if (event.key === "+" || event.key === "=") {
                    event.preventDefault();
                    handleZoomIn();
                    return;
                }

                if (event.key === "-" || event.key === "_") {
                    event.preventDefault();
                    handleZoomOut();
                    return;
                }

                if (event.key === "0") {
                    event.preventDefault();
                    handleResetZoom();
                    return;
                }

                if (event.key.toLowerCase() === "f") {
                    event.preventDefault();
                    handleResetZoom();
                    return;
                }

                if (event.key.toLowerCase() === "w") {
                    event.preventDefault();
                    handleFitWidth();
                    return;
                }

                if (event.key === "Escape") {
                    setSelectedKey(null);
                    setAddMode(false);
                    return;
                }

            }

            if (navigationLocked) {
                return;
            }

            if (event.key === "ArrowLeft" && page > 1) {
                setPage((current) => current - 1);
            }

            if (event.key === "ArrowRight" && pageData && page < pageData.page_count) {
                setPage((current) => current + 1);
            }

        };

        window.addEventListener("keydown", handleKeyDown);

        return () => window.removeEventListener("keydown", handleKeyDown);

    }, [page, pageData, navigationLocked, confirmDialog, handleZoomIn, handleZoomOut, handleResetZoom, handleFitWidth]);

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

                    const withoutDraft = prev.filter((item) => item.key !== drag.newKey);

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

                    return prev.map((item) => (item.key === drag.key ? { ...item, bbox } : item));

                });

            } else if (drag.type === "resize") {

                const bbox = resizeBox(drag.origBbox, drag.handle, current, naturalSize);

                setDraftBoundaries((prev) => {

                    if (!drag.historyPushed) {
                        pushHistory(prev);
                        drag.historyPushed = true;
                    }

                    return prev.map((item) => (item.key === drag.key ? { ...item, bbox } : item));

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

        event.preventDefault();
        event.stopPropagation();

        setSelectedKey(item.key);

        if (!editMode || item.is_multi_page) {
            return;
        }

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
        setEditMode((current) => {
            const next = !current;
            if (!next) {
                setSelectedKey(null);
            }
            return next;
        });
        setAddMode(false);
    };

    const handleEditSelected = () => {
        setEditMode(true);
        setAddMode(false);
    };

    const handleBoxHover = (key) => setHoveredKey(key);
    const handleBoxHoverEnd = (key) => setHoveredKey((current) => (current === key ? null : current));

    const handleRequestDeleteSelected = () => {
        if (!selectedKey) {
            return;
        }
        setConfirmDialog("delete");
    };

    const handleConfirmDelete = () => {

        pushHistory(draftBoundaries);

        setDraftBoundaries((prev) => prev.filter((item) => item.key !== selectedKey));

        setSelectedKey(null);
        setConfirmDialog(null);

    };

    const handleRequestDiscard = () => {
        if (pendingCount === 0) {
            return;
        }
        setConfirmDialog("discard");
    };

    const handleConfirmDiscard = () => {

        setDraftBoundaries(originalBoundaries.map(toDraftItem));
        historyRef.current = [];
        setCanUndo(false);
        setSelectedKey(null);
        setConfirmDialog(null);

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

            } catch (error) {

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

            } catch (error) {

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

            } catch (error) {

                results.push({
                    article_id: null,
                    action: "create-failed",
                    error: error?.response?.data?.detail || "Failed to create.",
                });

            }

            advance();

        }

        await loadBoundaries(page);

        const failed = results.filter((r) => r.error);
        const succeeded = results.length - failed.length;
        const dbSyncFailures = results.filter((r) => r.db_synced === false);

        if (succeeded > 0) {
            pushToast("success", `${succeeded} boundary change${succeeded === 1 ? "" : "s"} saved.`);
        }

        if (failed.length > 0) {
            pushToast("error", `${failed.length} change${failed.length === 1 ? "" : "s"} failed: ${failed[0].error}`);
        }

        if (dbSyncFailures.length > 0) {
            pushToast("error", `${dbSyncFailures.length} change${dbSyncFailures.length === 1 ? "" : "s"} saved but failed to sync to the database.`);
        }

        setExtractionResults((prev) => {

            const next = { ...prev };

            results.forEach((r) => {
                if (r.article_id && (r.headline || r.article_text)) {
                    next[r.article_id] = { headline: r.headline, article_text: r.article_text };
                }
            });

            return next;

        });

        setLastSaveHadError(failed.length > 0);

        historyRef.current = [];
        setCanUndo(false);
        setSelectedKey(null);
        setSaving(false);
        setSaveProgress(null);

    };

    if (!documentDetail) {

        return (
            <div className="studio-loading">
                <div className="studio-spinner" />
                <p>Loading document…</p>
            </div>
        );

    }

    if (documentStatus === "failed" && !documentReady) {

        return (
            <div className="studio-loading">
                <h2 className="studio-loading-title">Processing failed</h2>
                <p className="studio-loading-detail">
                    This document could not be processed. Try uploading it again.
                </p>
                <div className="studio-loading-actions">
                    <Link to="/" className="btn btn-secondary">
                        Back to documents
                    </Link>
                </div>
                <ToastStack toasts={toasts} onDismiss={dismissToast} />
            </div>
        );

    }

    if (!documentReady) {

        return (
            <div className="studio-loading">
                <div className="studio-spinner" />
                <h2 className="studio-loading-title">Document under processing</h2>
                <p className="studio-loading-detail">
                    Please wait — this document is still being processed. It will
                    open automatically as soon as it's ready.
                </p>
                <div className="studio-loading-actions">
                    <Link to="/" className="btn btn-secondary">
                        Back to documents
                    </Link>
                </div>
            </div>
        );

    }

    if (!pageData) {

        return (
            <div className="studio-loading">
                <div className="studio-spinner" />
                <p>Loading document…</p>
            </div>
        );

    }

    const handleSize = naturalSize ? Math.max(18, naturalSize.width / 120) : 20;

    const selectedItem = draftBoundaries.find((item) => item.key === selectedKey);
    const isSelectedEdited = selectedItem ? editedArticleIds.has(selectedItem.article_id) : false;
    const extractionResult = selectedItem && !selectedItem.isNew
        ? extractionResults[selectedItem.article_id]
        : null;

    const saveState = saving
        ? "saving"
        : lastSaveHadError
            ? "error"
            : pendingCount > 0
                ? "unsaved"
                : "saved";

    const scalePercent = Math.round(transform.scale * 100);

    return (

        <div className="studio-shell">

            <ViewerTopBar
                pdfName={pageData.pdf_name}
                page={page}
                pageCount={pageData.page_count}
                onPrevPage={() => setPage((current) => current - 1)}
                onNextPage={() => setPage((current) => current + 1)}
                navigationLocked={navigationLocked}
                editMode={editMode}
                onToggleEditMode={handleToggleEditMode}
                addMode={addMode}
                onToggleAddMode={handleToggleAddMode}
                canUndo={canUndo}
                onUndo={handleUndo}
                pendingCount={pendingCount}
                onDiscard={handleRequestDiscard}
                saving={saving}
                onSaveAll={handleSaveAll}
                exportHref={`${API_BASE}/documents/${documentId}/export/pdf`}
                sidebarOpen={sidebarOpen}
                onToggleSidebar={() => setSidebarOpen((current) => !current)}
                inspectorOpen={inspectorOpen}
                onToggleInspector={() => setInspectorOpen((current) => !current)}
            />

            <div className="studio-body">

                <PageSidebar
                    collapsed={!sidebarOpen}
                    pdfName={pageData.pdf_name}
                    documentId={documentId}
                    page={page}
                    pageCount={pageData.page_count}
                    pageBoundaryCounts={pageBoundaryCounts}
                    navigationLocked={navigationLocked}
                    onSelectPage={setPage}
                    thumbBaseUrl={API_BASE}
                />

                <div className="studio-canvas" ref={canvasStageRef}>

                    <TransformWrapper
                        ref={transformRef}
                        initialScale={1}
                        minScale={MIN_SCALE}
                        maxScale={MAX_SCALE}
                        centerOnInit={true}
                        centerZoomedOut={true}
                        limitToBounds={true}
                        wheel={{ step: 0.15, wheelDisabled: true }}
                        trackPadPanning={{ disabled: addMode }}
                        doubleClick={{ disabled: false }}
                        panning={{ disabled: addMode, velocityDisabled: false }}
                        pinch={{ disabled: false }}
                        alignmentAnimation={{ disabled: true }}
                        onTransform={(_ref, state) => setTransformState(state)}
                    >

                        <TransformComponent
                            wrapperStyle={{ width: "100%", height: "100%" }}
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
                                    src={`${API_BASE}${pageData.plain_image}`}
                                    alt="Newspaper page"
                                    className="boundary-image"
                                    onLoad={handleImageLoad}
                                    draggable={false}
                                />

                                {naturalSize && (
                                    <BoundaryLayer
                                        ref={svgRef}
                                        naturalSize={naturalSize}
                                        items={draftBoundaries}
                                        selectedKey={selectedKey}
                                        hoveredKey={hoveredKey}
                                        addMode={addMode}
                                        editMode={editMode}
                                        editedArticleIds={editedArticleIds}
                                        handleSize={handleSize}
                                        onBackgroundPointerDown={handleBackgroundPointerDown}
                                        onBoxPointerDown={handleBoxPointerDown}
                                        onBoxHover={handleBoxHover}
                                        onBoxHoverEnd={handleBoxHoverEnd}
                                        onResizePointerDown={handleResizePointerDown}
                                    />
                                )}

                            </div>

                        </TransformComponent>

                    </TransformWrapper>

                    <CanvasToolbar
                        scalePercent={scalePercent}
                        onZoomOut={handleZoomOut}
                        onZoomIn={handleZoomIn}
                        onFit={handleResetZoom}
                        onFitWidth={handleFitWidth}
                        onReset={handleResetZoom}
                    />

                    {addMode && (
                        <div className="studio-hint">
                            Drag on the page to draw a new article boundary.
                        </div>
                    )}

                    {editMode && !selectedItem && (
                        <div className="studio-hint">
                            Click a boundary to select it, then drag to move or use the corner handles to resize.
                        </div>
                    )}

                </div>

                <ArticleInspector
                    collapsed={!inspectorOpen}
                    page={page}
                    pageCount={pageData.page_count}
                    naturalSize={naturalSize}
                    articleCount={draftBoundaries.length}
                    pendingCount={pendingCount}
                    documentLanguage={documentDetail?.language}
                    selectedItem={selectedItem}
                    isEdited={isSelectedEdited}
                    editMode={editMode}
                    saving={saving}
                    documentUuid={documentUuid}
                    onEditBoundary={handleEditSelected}
                    onRequestDelete={handleRequestDeleteSelected}
                    extractionResult={extractionResult}
                />

            </div>

            <StatusBar
                saveState={saveState}
                pendingCount={pendingCount}
                saveProgress={saveProgress}
                selectedItem={selectedItem}
                naturalSize={naturalSize}
                scalePercent={scalePercent}
            />

            <ConfirmDialog
                open={confirmDialog === "delete"}
                title="Delete boundary?"
                message={`This will remove ${selectedItem?.isNew ? "this new boundary" : selectedItem?.article_id || "this article"} from the page.`}
                confirmLabel="Delete"
                danger
                onConfirm={handleConfirmDelete}
                onCancel={() => setConfirmDialog(null)}
            />

            <ConfirmDialog
                open={confirmDialog === "discard"}
                title="Discard changes?"
                message={`This will revert every unsaved add, edit, and delete on this page (${pendingCount} change${pendingCount === 1 ? "" : "s"}).`}
                confirmLabel="Discard"
                danger
                onConfirm={handleConfirmDiscard}
                onCancel={() => setConfirmDialog(null)}
            />

            <ToastStack toasts={toasts} onDismiss={dismissToast} />

        </div>

    );

}

export default Viewer;

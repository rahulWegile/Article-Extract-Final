import { forwardRef } from "react";

function articleLabel(item) {

    if (item.isNew) {
        return "NEW";
    }

    const match = String(item.article_id || "").match(/(\d+)/);

    return match ? `ART ${match[1]}` : String(item.article_id || "?");

}

const BoundaryLayer = forwardRef(function BoundaryLayer(
    {
        naturalSize,
        items,
        selectedKey,
        hoveredKey,
        addMode,
        editMode,
        editedArticleIds,
        handleSize,
        onBackgroundPointerDown,
        onBoxPointerDown,
        onBoxHover,
        onBoxHoverEnd,
        onResizePointerDown,
    },
    svgRef,
) {

    const labelFontSize = Math.max(14, naturalSize.width / 170);
    const labelPadX = labelFontSize * 0.55;
    const labelHeight = labelFontSize * 1.7;

    const selectedItem = items.find((item) => item.key === selectedKey);

    // Render largest-area boxes first (so they paint at the bottom) and
    // smallest last (on top) -- when two boundaries overlap, the smaller
    // one is almost always the more specific target the user meant to
    // click, and SVG paints later siblings on top of earlier ones.
    const itemsByRenderOrder = [...items].sort((a, b) => {
        const areaA = (a.bbox.x2 - a.bbox.x1) * (a.bbox.y2 - a.bbox.y1);
        const areaB = (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1);
        return areaB - areaA;
    });

    return (

        <svg
            ref={svgRef}
            className={"boundary-overlay-svg" + (addMode ? " add-mode" : "")}
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
                    onPointerDown={onBackgroundPointerDown}
                />
            )}

            {itemsByRenderOrder.map((item) => {

                const hasSubRects = item.sub_rects && item.sub_rects.length > 0;
                const isSelected = item.key === selectedKey;
                const isHovered = item.key === hoveredKey && !isSelected;

                const boxClassName =
                    "boundary-box" +
                    (item.is_multi_page ? " locked" : "") +
                    (isSelected ? " selected" : "") +
                    (isHovered ? " hovered" : "") +
                    (item.isNew ? " pending-new" : "") +
                    (editedArticleIds.has(item.article_id) ? " pending-edited" : "");

                const label = articleLabel(item);
                const labelWidth = label.length * labelFontSize * 0.62 + labelPadX * 2;
                const labelAbove = item.bbox.y1 - labelHeight - 4 >= 0;
                const labelY = labelAbove ? item.bbox.y1 - labelHeight - 4 : item.bbox.y1 + 4;

                return (

                    <g key={item.key}>

                        <rect
                            x={item.bbox.x1}
                            y={item.bbox.y1}
                            width={item.bbox.x2 - item.bbox.x1}
                            height={item.bbox.y2 - item.bbox.y1}
                            className={boxClassName + (hasSubRects ? " boundary-box-hit-only" : "")}
                            onPointerDown={onBoxPointerDown(item)}
                            onPointerEnter={() => onBoxHover(item.key)}
                            onPointerLeave={() => onBoxHoverEnd(item.key)}
                        >
                            {item.is_multi_page && (
                                <title>Spans multiple pages — not editable here</title>
                            )}
                        </rect>

                        {hasSubRects && item.sub_rects.map((r, index) => (
                            <rect
                                key={`${item.key}-sub-${index}`}
                                x={r.x1}
                                y={r.y1}
                                width={r.x2 - r.x1}
                                height={r.y2 - r.y1}
                                className={boxClassName + " boundary-sub-rect"}
                                onPointerDown={onBoxPointerDown(item)}
                                onPointerEnter={() => onBoxHover(item.key)}
                                onPointerLeave={() => onBoxHoverEnd(item.key)}
                            >
                                {item.is_multi_page && (
                                    <title>Spans multiple pages — not editable here</title>
                                )}
                            </rect>
                        ))}

                        <g className={"boundary-label" + (isSelected ? " selected" : "")} pointerEvents="none">
                            <rect
                                x={item.bbox.x1}
                                y={labelY}
                                width={labelWidth}
                                height={labelHeight}
                                rx={labelHeight / 4}
                            />
                            <text
                                x={item.bbox.x1 + labelPadX}
                                y={labelY + labelHeight / 2}
                                fontSize={labelFontSize}
                                dominantBaseline="central"
                            >
                                {label}
                            </text>
                        </g>

                    </g>

                );

            })}

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
                        onPointerDown={onResizePointerDown(selectedItem, handle)}
                    />

                ))

            )}

        </svg>

    );

});

export default BoundaryLayer;

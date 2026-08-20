import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import api from "../api/api";
import {
    TransformWrapper,
    TransformComponent,
} from "react-zoom-pan-pinch";

function Viewer() {

    const { documentId } = useParams();

    const [pageData, setPageData] = useState(null);

    const [page, setPage] = useState(1);
  

    useEffect(() => {

        loadPage(page);

    }, [page]);

    useEffect(() => {

        const handleKeyDown = (event) => {

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

    }, [page, pageData]);

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

    if (!pageData) {

        return (

            <div className="viewer-loading">

                <div className="viewer-spinner" />

                <p>Loading page...</p>

            </div>

        );

    }

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

            </div>

            <div className="image-container">

                <TransformWrapper

                    initialScale={1}

                    minScale={0.4}

                    maxScale={10}

                    centerOnInit={true}

                    centerZoomedOut={true}

                    limitToBounds={false}

                    wheel={{
                        step: 0.15,
                    }}

                    doubleClick={{
                        disabled: false,
                    }}

                    panning={{
                        disabled: false,
                        velocityDisabled: false,
                    }}

                    pinch={{
                        disabled: false,
                    }}

                    alignmentAnimation={{
                        disabled: true,
                    }}

                >

                    {({ zoomIn, zoomOut, resetTransform }) => (

                        <>

                            <div className="zoom-toolbar">

                                <button title="Zoom in" onClick={() => zoomIn()}>
                                    +
                                </button>

                                <button title="Zoom out" onClick={() => zoomOut()}>
                                    −
                                </button>

                                <button title="Reset zoom" onClick={() => resetTransform()}>
                                    Reset
                                </button>

                            </div>

                            <TransformComponent
                                wrapperStyle={{
                                    width: "100%",
                                    height: "82vh",
                                }}
                                contentStyle={{
                                    width: "100%",
                                    height: "100%",
                                    display: "flex",
                                    justifyContent: "center",
                                    alignItems: "center",
                                }}
                            >

                                <img
                                    src={`${import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"}${pageData.image}`}
                                    alt="Boundary"
                                    className="boundary-image"
                                />

                            </TransformComponent>

                        </>

                    )}

                </TransformWrapper>

            </div>

            <div className="viewer-navigation">

                <button

                    disabled={page <= 1}

                    onClick={() => setPage(page - 1)}

                >

                    ⬅ Previous

                </button>

                <span className="nav-page-indicator">

                    {page} / {pageData.page_count}

                </span>

                <button

                    disabled={page >= pageData.page_count}

                    onClick={() => setPage(page + 1)}

                >

                    Next ➡

                </button>

            </div>

        </div>

    );

}

export default Viewer;
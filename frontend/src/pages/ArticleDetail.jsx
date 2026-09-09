import {
    useEffect,
    useState,
} from "react";

import {
    useNavigate,
    useParams,
} from "react-router-dom";

import api from "../api/api";

import "../styles/archive.css";


const API_BASE =
    import.meta.env.VITE_API_URL ??
    "http://127.0.0.1:8000";


function ArticleDetail() {

    const navigate =
        useNavigate();


    const {
        documentId,
        logicalArticleId,
    } = useParams();


    const [
        article,
        setArticle
    ] = useState(null);


    const [
        loading,
        setLoading
    ] = useState(true);


    const [
        error,
        setError
    ] = useState("");


    const [
        searchState,
        setSearchState
    ] = useState(null);


    // ========================================================
    // LOAD
    // ========================================================

    useEffect(() => {

        loadArticle();

        loadSearchState();

    }, [
        documentId,
        logicalArticleId,
    ]);


    // ========================================================
    // GET ARTICLE
    // ========================================================

    const loadArticle =
        async () => {

            setLoading(true);

            setError("");


            try {

                const response =
                    await api.get(
                        `/search/articles/${documentId}/${logicalArticleId}`
                    );


                setArticle(
                    response.data
                );

            }
            catch (err) {

                console.error(
                    err
                );

                setError(
                    "Unable to load this article."
                );

            }
            finally {

                setLoading(false);

            }

        };


    // ========================================================
    // SEARCH STATE
    // ========================================================

    const loadSearchState =
        () => {

            try {

                const raw =
                    sessionStorage.getItem(
                        "newspaperSearch"
                    );


                if (!raw) {
                    return;
                }


                setSearchState(
                    JSON.parse(raw)
                );

            }
            catch (err) {

                console.error(
                    err
                );

            }

        };


    // ========================================================
    // ARTICLE INDEX
    // ========================================================

    const articleIndex =
        searchState?.results?.findIndex(
            (item) =>
                item.document_id ===
                    documentId &&
                item.logical_article_id ===
                    logicalArticleId
        ) ?? -1;


    const totalArticles =
        searchState?.results?.length || 0;


    // ========================================================
    // PREVIOUS
    // ========================================================

    const goPrevious =
        () => {

            if (
                articleIndex <= 0 ||
                !searchState
            ) {

                return;

            }


            const previous =
                searchState.results[
                    articleIndex - 1
                ];


            navigate(
                `/search/article/${previous.document_id}/${previous.logical_article_id}`
            );

        };


    // ========================================================
    // NEXT
    // ========================================================

    const goNext =
        () => {

            if (
                articleIndex < 0 ||
                articleIndex >=
                    totalArticles - 1
            ) {

                return;

            }


            const next =
                searchState.results[
                    articleIndex + 1
                ];


            navigate(
                `/search/article/${next.document_id}/${next.logical_article_id}`
            );

        };


    // ========================================================
    // IMAGE URL
    // ========================================================

    const imageUrl =
        (url) => {

            if (!url) {
                return null;
            }


            if (
                url.startsWith(
                    "http://"
                ) ||
                url.startsWith(
                    "https://"
                )
            ) {

                return url;

            }


            return (
                `${API_BASE}${url}`
            );

        };


    // ========================================================
    // LOADING
    // ========================================================

    if (loading) {

        return (

            <div className="archive-page">

                <ArchiveHeader
                    navigate={
                        navigate
                    }
                />

                <div className="article-loading">

                    Loading article...

                </div>

            </div>

        );

    }


    // ========================================================
    // ERROR
    // ========================================================

    if (
        error ||
        !article
    ) {

        return (

            <div className="archive-page">

                <ArchiveHeader
                    navigate={
                        navigate
                    }
                />

                <div className="article-error">

                    <h2>
                        Article not found
                    </h2>

                    <p>
                        {error}
                    </p>


                    <button
                        onClick={() =>
                            navigate(
                                "/search"
                            )
                        }
                    >
                        Back to Search
                    </button>

                </div>

            </div>

        );

    }


    // ========================================================
    // ARTICLE
    // ========================================================

    return (

        <div className="archive-page">

            <ArchiveHeader
                navigate={
                    navigate
                }
            />


            <main className="article-page">


                {/* ==================================================
                    TOP NAVIGATION
                    ================================================== */}

                <div className="article-topbar">

                    <button
                        className="back-results"
                        onClick={() =>
                            navigate(
                                "/search"
                            )
                        }
                    >

                        ←

                        <span>
                            Back to results
                        </span>

                    </button>


                    {articleIndex >= 0 && (

                        <div className="article-counter">

                            <button
                                onClick={
                                    goPrevious
                                }
                                disabled={
                                    articleIndex <= 0
                                }
                            >
                                ‹
                            </button>


                            <span>

                                Article{" "}
                                {articleIndex + 1}
                                {" "}
                                of{" "}
                                {totalArticles}

                            </span>


                            <button
                                onClick={
                                    goNext
                                }
                                disabled={
                                    articleIndex >=
                                    totalArticles - 1
                                }
                            >
                                ›
                            </button>

                        </div>

                    )}


                    {articleIndex >= 0 && (

                        <button
                            className="next-article-button"
                            onClick={
                                goNext
                            }
                            disabled={
                                articleIndex >=
                                totalArticles - 1
                            }
                        >

                            Next Article

                            →

                        </button>

                    )}

                </div>


                {/* ==================================================
                    ARTICLE BODY
                    ================================================== */}

                <article className="article-content">


                    {/* CATEGORY */}

                    <div className="article-tags">

                        {article.category && (

                            <span className="article-category">

                                {
                                    article.category
                                }

                            </span>

                        )}


                        {article.sentiment && (

                            <span
                                className={
                                    `article-sentiment sentiment-${article.sentiment.toLowerCase()}`
                                }
                            >

                                {
                                    article.sentiment
                                }

                            </span>

                        )}

                    </div>


                    {/* TITLE */}

                    <h1 className="article-title" dir="auto">

                        {
                            article.title
                        }

                    </h1>


                    {/* AUTHOR + DATE */}

                    <div className="article-byline">

                        {article.author && (

                            <div className="byline-item">

                                <span>
                                    ✎
                                </span>

                                <span>
                                    {
                                        article.author
                                    }
                                </span>

                            </div>

                        )}


                        {(
                            article.display_date ||
                            article.article_date ||
                            article.publish_date
                        ) && (

                            <div className="byline-item">

                                <span>
                                    ▣
                                </span>

                                <span>
                                    {
                                        formatDate(
                                            article.display_date ||
                                            article.article_date ||
                                            article.publish_date
                                        )
                                    }
                                </span>

                            </div>

                        )}

                    </div>


                    <div className="article-divider" />


                    {/* SUMMARY */}

                    {article.summary && (

                        <section className="article-section">

                            <h2>
                                Summary
                            </h2>


                            <p className="article-summary" dir="auto">

                                {
                                    article.summary
                                }

                            </p>

                        </section>

                    )}


                    {/* ARTICLE TEXT */}

                    {article.article_text && (

                        <section className="article-section">

                            <h2>
                                Article
                            </h2>


                            <div className="article-body">

                                {splitParagraphs(
                                    article.article_text
                                ).map(
                                    (
                                        paragraph,
                                        index
                                    ) => (

                                        <p
                                            key={
                                                index
                                            }
                                            dir="auto"
                                        >
                                            {
                                                paragraph
                                            }
                                        </p>

                                    )
                                )}

                            </div>

                        </section>

                    )}


                    {/* ENTITIES */}

                    <MetadataSection
                        title="Entities"
                        items={
                            article.entities
                        }
                        className="blue-chip"
                    />


                    {/* TOPICS */}

                    <MetadataSection
                        title="Topics"
                        items={
                            article.topics
                        }
                        className="green-chip"
                    />


                    {/* KEYWORDS */}

                    <MetadataSection
                        title="Keywords"
                        items={
                            article.keywords
                        }
                        className="gray-chip"
                    />


                    {/* ==================================================
                        ARTICLE SEGMENTS
                        ================================================== */}

                    {article.segments &&
                        article.segments.length > 0 && (

                            <section className="article-section">

                                <h2>
                                    Article Pages
                                </h2>


                                <div className="segment-list">

                                    {article.segments.map(
                                        (
                                            segment,
                                            index
                                        ) => (

                                            <div
                                                className="segment-card"
                                                key={
                                                    segment.id
                                                }
                                            >

                                                <div className="segment-header">

                                                    <strong>

                                                        Page{" "}
                                                        {
                                                            segment.page_number
                                                        }

                                                    </strong>


                                                    <span>

                                                        Segment{" "}
                                                        {index + 1}

                                                    </span>

                                                </div>


                                                {segment.crop_url && (

                                                    <img
                                                        src={
                                                            imageUrl(
                                                                segment.crop_url
                                                            )
                                                        }
                                                        alt={
                                                            `Article page ${segment.page_number}`
                                                        }
                                                        className="segment-image"
                                                    />

                                                )}

                                            </div>

                                        )
                                    )}

                                </div>

                            </section>

                        )}


                    {/* ==================================================
                        IMAGES
                        ================================================== */}

                    {article.images &&
                        article.images.length > 0 && (

                            <section className="article-section">

                                <h2>
                                    Images
                                </h2>


                                <div className="article-images">

                                    {article.images.map(
                                        (
                                            image
                                        ) => (

                                            <figure
                                                className="article-image-card"
                                                key={
                                                    image.id
                                                }
                                            >

                                                {image.url && (

                                                    <img
                                                        src={
                                                            imageUrl(
                                                                image.url
                                                            )
                                                        }
                                                        alt={
                                                            image.description ||
                                                            image.caption ||
                                                            "Article image"
                                                        }
                                                    />

                                                )}


                                                <figcaption>

                                                    {image.caption && (

                                                        <strong>
                                                            {
                                                                image.caption
                                                            }
                                                        </strong>

                                                    )}


                                                    {image.description && (

                                                        <span>
                                                            {
                                                                image.description
                                                            }
                                                        </span>

                                                    )}

                                                </figcaption>

                                            </figure>

                                        )
                                    )}

                                </div>

                            </section>

                        )}

                </article>

            </main>

        </div>

    );

}


// ============================================================
// HEADER
// ============================================================

function ArchiveHeader({
    navigate,
}) {

    const [
        query,
        setQuery
    ] = useState("");


    const search =
        () => {

            const value =
                query.trim();


            if (!value) {
                return;
            }


            navigate(
                `/search?q=${encodeURIComponent(
                    value
                )}`
            );

        };


    return (

        <header className="archive-header">

            <div
                className="archive-logo"
                onClick={() =>
                    navigate("/")
                }
            >

                <span className="archive-logo-mark" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="3" y="3" width="18" height="18" rx="4" fill="currentColor" opacity="0.15" />
                        <path
                            d="M7 8h10M7 12h10M7 16h6"
                            stroke="currentColor"
                            strokeWidth="1.75"
                            strokeLinecap="round"
                        />
                    </svg>
                </span>

                <span>
                    Newspaper Archive
                </span>

            </div>


            <div className="archive-header-search">

                <span className="archive-header-search-icon" aria-hidden="true">
                    <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="9" cy="9" r="6.5" stroke="currentColor" strokeWidth="1.6" />
                        <path d="M17 17l-3.8-3.8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                    </svg>
                </span>

                <input
                    value={query}
                    placeholder="Search articles..."
                    aria-label="Search articles"
                    onChange={(event) =>
                        setQuery(
                            event.target.value
                        )
                    }
                    onKeyDown={(event) => {

                        if (
                            event.key ===
                            "Enter"
                        ) {

                            search();

                        }

                    }}
                />

            </div>


            <nav className="archive-nav" aria-label="Primary">

                <button
                    onClick={() =>
                        navigate("/")
                    }
                >
                    Home
                </button>


                <button
                    onClick={() =>
                        navigate(
                            "/search"
                        )
                    }
                >
                    Browse
                </button>


                <button>
                    Saved
                </button>

            </nav>

        </header>

    );

}


// ============================================================
// METADATA SECTION
// ============================================================

function MetadataSection({
    title,
    items,
    className,
}) {

    if (
        !items ||
        items.length === 0
    ) {

        return null;

    }


    return (

        <section className="metadata-section">

            <h2>
                {title}
            </h2>


            <div className="chips">

                {items.map(
                    (
                        item,
                        index
                    ) => (

                        <span
                            key={
                                index
                            }
                            className={
                                `chip ${className}`
                            }
                        >

                            {item}

                        </span>

                    )
                )}

            </div>

        </section>

    );

}


// ============================================================
// TEXT
// ============================================================

function splitParagraphs(
    text
) {

    return text
        .split(
            /\n\s*\n/
        )
        .map(
            item =>
                item.trim()
        )
        .filter(
            Boolean
        );

}


// ============================================================
// DATE
// ============================================================

function formatDate(
    value
) {

    if (!value) {
        return "";
    }


    const date =
        new Date(
            `${value}T00:00:00`
        );


    if (
        Number.isNaN(
            date.getTime()
        )
    ) {

        return value;

    }


    return date.toLocaleDateString(
        "en-GB",
        {
            day: "2-digit",
            month: "long",
            year: "numeric",
        }
    );

}


export default ArticleDetail;
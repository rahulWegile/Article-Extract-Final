import {
    useEffect,
    useRef,
    useState,
} from "react";

import {
    useNavigate,
    useSearchParams,
} from "react-router-dom";

import api from "../api/api";

import "../styles/archive.css";


const PAGE_SIZE = 20;


function Search() {

    const navigate =
        useNavigate();

    const [
        searchParams,
        setSearchParams
    ] = useSearchParams();


    // ========================================================
    // TEXT SEARCH
    // ========================================================

    const [
        query,
        setQuery
    ] = useState(
        searchParams.get("q") || ""
    );


    // ========================================================
    // NEWSPAPERS
    // ========================================================

    const [
        newspapers,
        setNewspapers
    ] = useState([]);


    const [
        selectedNewspaper,
        setSelectedNewspaper
    ] = useState(
        searchParams.get("newspaper") || ""
    );


    // ========================================================
    // PUBLICATION DATES
    // ========================================================

    const [
        publishDates,
        setPublishDates
    ] = useState([]);


    const [
        selectedDate,
        setSelectedDate
    ] = useState(
        searchParams.get("publish_date") || ""
    );


    // ========================================================
    // RESULTS
    // ========================================================

    const [
        results,
        setResults
    ] = useState([]);


    const [
        total,
        setTotal
    ] = useState(0);


    const [
        page,
        setPage
    ] = useState(1);


    // ========================================================
    // LOADING
    // ========================================================

    const [
        loading,
        setLoading
    ] = useState(false);


    const [
        loadingNewspapers,
        setLoadingNewspapers
    ] = useState(false);


    const [
        loadingDates,
        setLoadingDates
    ] = useState(false);


    // ========================================================
    // SEARCH STATE
    // ========================================================

    const [
        searched,
        setSearched
    ] = useState(false);


    const [
        error,
        setError
    ] = useState("");


    // ========================================================
    // LOAD NEWSPAPERS
    // ========================================================

    useEffect(() => {

        loadNewspapers();

    }, []);


    // ========================================================
    // LOAD INITIAL DATA FROM URL
    // ========================================================

    useEffect(() => {

        const urlQuery =
            searchParams.get("q") || "";

        const urlNewspaper =
            searchParams.get("newspaper") || "";

        const urlDate =
            searchParams.get("publish_date") || "";


        if (urlQuery) {

            setQuery(
                urlQuery
            );

        }


        if (urlNewspaper) {

            setSelectedNewspaper(
                urlNewspaper
            );

        }


        // ----------------------------------------------------
        // Load publication dates -- scoped to the newspaper if one
        // was supplied, otherwise every date across all newspapers.
        // ----------------------------------------------------

        loadDates(
            urlNewspaper
        );


        // ----------------------------------------------------
        // If URL contains newspaper/date, load edition.
        // ----------------------------------------------------

        if (
            urlNewspaper &&
            urlDate
        ) {

            performSearch({
                suppliedQuery: urlQuery,
                suppliedNewspaper: urlNewspaper,
                suppliedDate: urlDate,
            });

            return;

        }


        // ----------------------------------------------------
        // Otherwise perform a text search, or -- if nothing was
        // supplied in the URL at all -- browse the full archive
        // so landing on this page always shows every article.
        // ----------------------------------------------------

        performSearch({
            suppliedQuery: urlQuery,
        });

    }, []);


    // ========================================================
    // LOAD NEWSPAPERS
    // ========================================================

    const loadNewspapers = async () => {

        setLoadingNewspapers(
            true
        );

        try {

            const response =
                await api.get(
                    "/search/newspapers"
                );


            const data =
                response.data;


            setNewspapers(
                Array.isArray(data)
                    ? data
                    : []
            );

        }
        catch (err) {

            console.error(
                "Unable to load newspapers:",
                err
            );

            setError(
                "Unable to load newspapers."
            );

        }
        finally {

            setLoadingNewspapers(
                false
            );

        }

    };


    // ========================================================
    // LOAD DATES FOR NEWSPAPER
    // ========================================================

    const loadDates = async (
        newspaper
    ) => {

        setLoadingDates(
            true
        );


        try {

            const response =
                await api.get(
                    "/search/dates",
                    {
                        // Omitting `newspaper` returns every
                        // publish date across all newspapers.
                        params: newspaper
                            ? { newspaper }
                            : {},
                    }
                );


            const data =
                response.data;


            setPublishDates(
                Array.isArray(data)
                    ? data
                    : []
            );

        }
        catch (err) {

            console.error(
                "Unable to load publication dates:",
                err
            );

            setPublishDates([]);

            setError(
                "Unable to load publication dates."
            );

        }
        finally {

            setLoadingDates(
                false
            );

        }

    };


    // ========================================================
    // NEWSPAPER CHANGE
    // ========================================================

    const handleNewspaperChange = async (
        event
    ) => {

        const newspaper =
            event.target.value;


        setSelectedNewspaper(
            newspaper
        );


        // ----------------------------------------------------
        // Reset date whenever newspaper changes.
        // ----------------------------------------------------

        setSelectedDate(
            ""
        );


        setPublishDates([]);


        // ----------------------------------------------------
        // Clear previous results.
        // ----------------------------------------------------

        setResults([]);

        setTotal(0);

        setPage(1);

        setSearched(false);

        setError("");


        // ----------------------------------------------------
        // Load dates -- scoped to the newspaper if one was picked,
        // otherwise every date across all newspapers.
        // ----------------------------------------------------

        await loadDates(
            newspaper
        );


        // ----------------------------------------------------
        // Update URL.
        // ----------------------------------------------------

        const params = {};


        if (query.trim()) {

            params.q =
                query.trim();

        }


        if (newspaper) {

            params.newspaper =
                newspaper;

        }


        setSearchParams(
            params,
            {
                replace: true,
            }
        );


        // ----------------------------------------------------
        // Auto-fetch results for the newly selected newspaper
        // (date was just reset above, so search unfiltered by date).
        // ----------------------------------------------------

        await performSearch({
            suppliedNewspaper: newspaper,
            suppliedDate: "",
            suppliedPage: 1,
        });

    };


    // ========================================================
    // DATE CHANGE
    // ========================================================

    const handleDateChange = async (
        event
    ) => {

        const date =
            event.target.value;


        setSelectedDate(
            date
        );


        setResults([]);

        setTotal(0);

        setPage(1);

        setSearched(false);

        setError("");


        // ----------------------------------------------------
        // Update URL.
        // ----------------------------------------------------

        const params = {};


        if (query.trim()) {

            params.q =
                query.trim();

        }


        if (selectedNewspaper) {

            params.newspaper =
                selectedNewspaper;

        }


        if (date) {

            params.publish_date =
                date;

        }


        setSearchParams(
            params,
            {
                replace: true,
            }
        );


        // ----------------------------------------------------
        // Auto-fetch results for the newly selected date.
        // ----------------------------------------------------

        await performSearch({
            suppliedNewspaper: selectedNewspaper,
            suppliedDate: date,
            suppliedPage: 1,
        });

    };


    // ========================================================
    // SEARCH / BROWSE
    // ========================================================

    const performSearch = async ({
        suppliedQuery = null,
        suppliedNewspaper = null,
        suppliedDate = null,
        suppliedPage = null,
    } = {}) => {

        const value = (
            suppliedQuery !== null
                ? suppliedQuery
                : query
        ).trim();


        const newspaper =
            suppliedNewspaper !== null
                ? suppliedNewspaper
                : selectedNewspaper;


        const date =
            suppliedDate !== null
                ? suppliedDate
                : selectedDate;


        const currentPage =
            suppliedPage !== null
                ? suppliedPage
                : page;


        // ----------------------------------------------------
        // No filters at all means "browse the entire archive" --
        // the backend already treats an empty query and no
        // newspaper/date as "match everything", so fetch it
        // instead of bailing out.
        // ----------------------------------------------------

        const offset =
            (currentPage - 1) * PAGE_SIZE;


        setLoading(
            true
        );

        setError("");

        setSearched(
            true
        );


        try {

            const params = {

                // Empty string is allowed by backend.
                q: value,

                limit: PAGE_SIZE,

                offset,

            };


            // ------------------------------------------------
            // Add newspaper filter.
            // ------------------------------------------------

            if (newspaper) {

                params.newspaper =
                    newspaper;

            }


            // ------------------------------------------------
            // Add publication-date filter.
            // ------------------------------------------------

            if (date) {

                params.publish_date =
                    date;

            }


            const response =
                await api.get(
                    "/search/articles",
                    {
                        params,
                    }
                );


            const data =
                response.data;


            const articles =
                data.results || [];


            setResults(
                articles
            );


            setTotal(
                data.total || 0
            );


            setPage(
                currentPage
            );


            // ------------------------------------------------
            // Save results for article navigation.
            // ------------------------------------------------

            sessionStorage.setItem(
                "newspaperSearch",
                JSON.stringify({
                    query: value,

                    newspaper:
                        newspaper || "",

                    publish_date:
                        date || "",

                    total:
                        data.total || 0,

                    results:
                        articles,
                })
            );


            // ------------------------------------------------
            // Update state.
            // ------------------------------------------------

            setQuery(
                value
            );


            setSelectedNewspaper(
                newspaper || ""
            );


            setSelectedDate(
                date || ""
            );


            // ------------------------------------------------
            // Update URL.
            // ------------------------------------------------

            const urlParams = {};


            if (value) {

                urlParams.q =
                    value;

            }


            if (newspaper) {

                urlParams.newspaper =
                    newspaper;

            }


            if (date) {

                urlParams.publish_date =
                    date;

            }


            setSearchParams(
                urlParams,
                {
                    replace: true,
                }
            );

        }
        catch (err) {

            console.error(
                "Search error:",
                err
            );


            setError(
                "Unable to search the newspaper archive."
            );


            setResults([]);

            setTotal(0);

        }
        finally {

            setLoading(
                false
            );

        }

    };


    // ========================================================
    // ENTER KEY
    // ========================================================

    const handleKeyDown = (
        event
    ) => {

        if (
            event.key === "Enter"
        ) {

            performSearch({
                suppliedPage: 1,
            });

        }

    };


    // ========================================================
    // PAGINATION
    // ========================================================

    const totalPages =
        Math.max(
            1,
            Math.ceil(
                total / PAGE_SIZE
            )
        );


    const goToPage = (
        targetPage
    ) => {

        if (
            targetPage === page ||
            targetPage < 1 ||
            targetPage > totalPages
        ) {

            return;

        }


        performSearch({
            suppliedPage: targetPage,
        });

    };


    // ========================================================
    // OPEN ARTICLE
    // ========================================================

    const openArticle = (
        article
    ) => {

        navigate(
            `/search/article/${article.document_id}/${article.logical_article_id}`
        );

    };


    // ========================================================
    // RESET FILTERS
    // ========================================================

    const clearFilters = () => {

        setQuery("");

        setSelectedNewspaper("");

        setSelectedDate("");

        setPublishDates([]);

        setResults([]);

        setTotal(0);

        setPage(1);

        setSearched(false);

        setError("");


        setSearchParams(
            {},
            {
                replace: true,
            }
        );

    };


    // ========================================================
    // RENDER
    // ========================================================

    return (

        <div className="archive-page">

            {/* =================================================
                HEADER
                ================================================= */}

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
                        onKeyDown={
                            handleKeyDown
                        }
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
                        className="active"
                        aria-current="page"
                    >
                        Browse
                    </button>


                    <button>
                        Saved
                    </button>

                </nav>

            </header>


            {/* =================================================
                MAIN
                ================================================= */}

            <main className="search-page-container">

                <div className="search-heading">

                    <h1>
                        Search Newspaper Archive
                    </h1>

                    <p>
                        Search articles, people,
                        topics and keywords.
                    </p>

                </div>


                {/* =================================================
                    TEXT SEARCH
                    ================================================= */}

                <div className="large-search">

                    <span className="large-search-icon" aria-hidden="true">
                        <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="9" cy="9" r="6.5" stroke="currentColor" strokeWidth="1.6" />
                            <path d="M17 17l-3.8-3.8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                        </svg>
                    </span>

                    <input
                        value={query}
                        placeholder="Search articles, people, topics and keywords..."
                        aria-label="Search articles, people, topics and keywords"
                        onChange={(event) =>
                            setQuery(
                                event.target.value
                            )
                        }
                        onKeyDown={
                            handleKeyDown
                        }
                    />


                    <button
                        onClick={() =>
                            performSearch({
                                suppliedPage: 1,
                            })
                        }
                        disabled={loading}
                    >

                        {loading
                            ? "Searching..."
                            : "Search"
                        }

                    </button>

                </div>


                {/* =================================================
                    NEWSPAPER / DATE FILTERS
                    ================================================= */}

                <section className="archive-filters" aria-label="Filters">

                    {/* =============================================
                        NEWSPAPER
                        ============================================= */}

                    <div className="archive-filter">

                        <label
                            htmlFor="newspaper-select"
                            className="sr-only"
                        >
                            Newspaper
                        </label>


                        <FilterDropdown
                            id="newspaper-select"
                            placeholder="Newspaper"
                            loadingLabel="Loading newspapers..."
                            value={selectedNewspaper}
                            disabled={loadingNewspapers}
                            options={newspapers.map(
                                (newspaper) => ({
                                    value: newspaper,
                                    label: newspaper,
                                })
                            )}
                            onChange={(value) =>
                                handleNewspaperChange({
                                    target: { value },
                                })
                            }
                        />

                    </div>


                    {/* =============================================
                        PUBLICATION DATE
                        ============================================= */}

                    <div className="archive-filter">

                        <label
                            htmlFor="publish-date-select"
                            className="sr-only"
                        >
                            Publish date
                        </label>


                        <FilterDropdown
                            id="publish-date-select"
                            placeholder="Publish date"
                            loadingLabel="Loading dates..."
                            value={selectedDate}
                            disabled={loadingDates}
                            options={publishDates.map(
                                (date) => ({
                                    value: date,
                                    label: formatDate(date),
                                })
                            )}
                            onChange={(value) =>
                                handleDateChange({
                                    target: { value },
                                })
                            }
                        />

                    </div>


                    {/* =============================================
                        FILTER BUTTON
                        ============================================= */}

                    <button
                        className="archive-filter-button"
                        onClick={() =>
                            performSearch({
                                suppliedPage: 1,
                            })
                        }
                        disabled={
                            loading
                        }
                    >

                        {loading
                            ? "Loading..."
                            : "Apply"
                        }

                    </button>


                    {/* =============================================
                        CLEAR BUTTON
                        ============================================= */}

                    {(query ||
                        selectedNewspaper ||
                        selectedDate) && (

                        <button
                            className="archive-clear-button"
                            onClick={
                                clearFilters
                            }
                        >
                            Clear
                        </button>

                    )}

                </section>


                {/* =================================================
                    ACTIVE FILTER SUMMARY
                    ================================================= */}

                {(selectedNewspaper ||
                    selectedDate) && (

                    <div className="active-filters">

                        {selectedNewspaper && (

                            <span className="active-filter-chip">
                                Newspaper:
                                {" "}
                                <strong>
                                    {
                                        selectedNewspaper
                                    }
                                </strong>
                            </span>

                        )}


                        {selectedDate && (

                            <span className="active-filter-chip">
                                Date:
                                {" "}
                                <strong>
                                    {
                                        formatDate(
                                            selectedDate
                                        )
                                    }
                                </strong>
                            </span>

                        )}

                    </div>

                )}


                {/* =================================================
                    ERROR
                    ================================================= */}

                {error && (

                    <div className="state-panel search-error-panel" role="alert">

                        <span className="state-panel-icon" aria-hidden="true">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
                                <path d="M12 8v5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                                <circle cx="12" cy="16" r="1" fill="currentColor" />
                            </svg>
                        </span>

                        <h2>
                            Search unavailable
                        </h2>

                        <p>
                            We couldn't load the archive results.
                            Please try again.
                        </p>

                        <button
                            className="state-panel-retry"
                            onClick={() =>
                                performSearch()
                            }
                        >
                            Try again
                        </button>

                    </div>

                )}


                {/* =================================================
                    LOADING SKELETON
                    ================================================= */}

                {loading && (

                    <div className="search-results-skeleton" aria-hidden="true">

                        {[0, 1, 2, 3, 4].map((key) => (

                            <div className="result-skeleton-card" key={key}>

                                <div className="result-skeleton-body">
                                    <div className="skeleton-bar h-title" />
                                    <div className="skeleton-bar h-meta" />
                                    <div className="skeleton-bar w-full" />
                                    <div className="skeleton-bar w-80" />
                                </div>

                                <div className="result-skeleton-thumb" />

                            </div>

                        ))}

                    </div>

                )}

                {/* =================================================
                    RESULTS
                    ================================================= */}

                {searched && !loading && !error && (

                    <section className="search-results">

                        <div className="results-summary">

                            <h2>

                                <span className="results-summary-eyebrow">
                                    Search results
                                </span>

                                {total}
                                {" "}
                                {total === 1
                                    ? "article"
                                    : "articles"
                                }
                                {" "}
                                found

                            </h2>


                            <span className="results-summary-context">

                                {selectedNewspaper
                                    ? (
                                        <>
                                            from
                                            {" "}
                                            "{selectedNewspaper}"

                                            {selectedDate && (
                                                <>
                                                    {" "}
                                                    on
                                                    {" "}
                                                    "{formatDate(
                                                        selectedDate
                                                    )}"
                                                </>
                                            )}
                                        </>
                                    )
                                    : query
                                        ? (
                                            <>
                                                matching
                                                {" "}
                                                "{query}"
                                            </>
                                        )
                                        : null
                                }

                            </span>

                        </div>


                        {results.length === 0 ? (

                            <div className="state-panel no-results">

                                <span className="state-panel-icon" aria-hidden="true">
                                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <circle cx="10.5" cy="10.5" r="6.5" stroke="currentColor" strokeWidth="1.6" />
                                        <path d="M19 19l-4.3-4.3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                                    </svg>
                                </span>

                                <h2>
                                    No articles found
                                </h2>

                                <p>
                                    Try a different keyword or adjust your filters.
                                </p>

                            </div>

                        ) : (

                            <>

                                {results.map(
                                    (
                                        article,
                                        index
                                    ) => (

                                        <article
                                            key={
                                                `${article.document_id}-${article.logical_article_id}`
                                            }
                                            className="search-result-card"
                                            tabIndex={0}
                                            role="link"
                                            onClick={() =>
                                                openArticle(
                                                    article
                                                )
                                            }
                                            onKeyDown={(event) => {

                                                if (
                                                    event.key === "Enter" ||
                                                    event.key === " "
                                                ) {

                                                    event.preventDefault();

                                                    openArticle(
                                                        article
                                                    );

                                                }

                                            }}
                                        >

                                            <div className="result-content">

                                                <div className="result-number">

                                                    {String(
                                                        (page - 1) * PAGE_SIZE +
                                                        index + 1
                                                    ).padStart(
                                                        2,
                                                        "0"
                                                    )}

                                                </div>


                                                <div className="result-main">

                                                    <div className="result-topline">

                                                        {article.category && (

                                                            <span className="category-tag">

                                                                {
                                                                    article.category
                                                                }

                                                            </span>

                                                        )}

                                                    </div>


                                                    <h3>
                                                        {
                                                            highlightMatch(article.title, query)
                                                        }
                                                    </h3>


                                                    {(article.author ||
                                                        article.display_date) && (

                                                        <p className="result-byline">

                                                            {
                                                                [
                                                                    article.author,
                                                                    article.display_date &&
                                                                        formatDate(article.display_date),
                                                                ]
                                                                    .filter(Boolean)
                                                                    .join(" · ")
                                                            }

                                                        </p>

                                                    )}


                                                    {article.summary && (

                                                        <p className="result-summary">

                                                            {
                                                                highlightMatch(article.summary, query)
                                                            }

                                                        </p>

                                                    )}


                                                    {getResultTags(article).length > 0 && (

                                                        <div className="result-tags">

                                                            {getResultTags(article).map(
                                                                (tag, tagIndex) => (

                                                                    <span
                                                                        className="result-tag"
                                                                        key={tagIndex}
                                                                    >
                                                                        {tag}
                                                                    </span>

                                                                )
                                                            )}

                                                        </div>

                                                    )}

                                                </div>

                                            </div>


                                            <div className="result-thumb">

                                                {article.images &&
                                                    article.images.length >
                                                        0 &&
                                                    article.images[0].url
                                                    ? (

                                                        <img
                                                            className="result-image"
                                                            src={
                                                                buildUrl(
                                                                    article.images[0].url
                                                                )
                                                            }
                                                            alt={
                                                                article.title
                                                            }
                                                        />

                                                    )
                                                    : (

                                                        <span className="result-thumb-placeholder" aria-hidden="true">
                                                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                                <path
                                                                    d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z"
                                                                    stroke="currentColor"
                                                                    strokeWidth="1.4"
                                                                    strokeLinejoin="round"
                                                                />
                                                                <path d="M14 3v5h5" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
                                                            </svg>
                                                        </span>

                                                    )
                                                }

                                            </div>

                                        </article>

                                    )
                                )}


                                {/* =========================================
                                    PAGINATION
                                    ========================================= */}

                                {totalPages > 1 && (

                                    <nav className="pagination" aria-label="Search results pages">

                                        <button
                                            className="pagination-nav-btn"
                                            onClick={() =>
                                                goToPage(page - 1)
                                            }
                                            disabled={
                                                page <= 1 ||
                                                loading
                                            }
                                        >
                                            ← Previous
                                        </button>


                                        <div className="pagination-pages">

                                            {getPageNumbers(page, totalPages).map(
                                                (item, itemIndex) => (

                                                    item === "…"
                                                        ? (
                                                            <span
                                                                className="pagination-ellipsis"
                                                                key={`ellipsis-${itemIndex}`}
                                                            >
                                                                …
                                                            </span>
                                                        )
                                                        : (
                                                            <button
                                                                key={item}
                                                                className={
                                                                    "pagination-page" +
                                                                    (item === page
                                                                        ? " is-active"
                                                                        : "")
                                                                }
                                                                aria-current={
                                                                    item === page
                                                                        ? "page"
                                                                        : undefined
                                                                }
                                                                onClick={() =>
                                                                    goToPage(item)
                                                                }
                                                                disabled={
                                                                    loading
                                                                }
                                                            >
                                                                {item}
                                                            </button>
                                                        )

                                                )
                                            )}

                                        </div>


                                        <button
                                            className="pagination-nav-btn"
                                            onClick={() =>
                                                goToPage(page + 1)
                                            }
                                            disabled={
                                                page >= totalPages ||
                                                loading
                                            }
                                        >
                                            Next →
                                        </button>

                                    </nav>

                                )}

                            </>

                        )}

                    </section>

                )}


                {/* =================================================
                    INITIAL EMPTY STATE
                    ================================================= */}

                {!searched && (

                    <div className="search-empty">

                        <div className="search-empty-icon">
                            🔎
                        </div>

                        <h2>
                            Search the archive
                        </h2>

                        <p>
                            Find articles by headline,
                            person, topic or keyword.
                        </p>

                    </div>

                )}

            </main>

        </div>

    );

}


// ============================================================
// FILTER DROPDOWN (custom-styled combobox)
// ============================================================

function FilterDropdown({
    id,
    placeholder,
    loadingLabel,
    value,
    disabled,
    options,
    onChange,
}) {

    const [
        open,
        setOpen
    ] = useState(false);


    const [
        activeIndex,
        setActiveIndex
    ] = useState(-1);


    const rootRef = useRef(null);

    const listRef = useRef(null);

    const triggerRef = useRef(null);


    const allOptions = [
        {
            value: "",
            label: placeholder,
        },
        ...options,
    ];


    const selected =
        allOptions.find(
            (option) =>
                option.value === value
        );


    useEffect(() => {

        if (!open) {
            return;
        }


        listRef.current?.focus();


        const handleOutside = (event) => {

            if (
                rootRef.current &&
                !rootRef.current.contains(event.target)
            ) {

                setOpen(false);

            }

        };

        document.addEventListener(
            "mousedown",
            handleOutside
        );

        return () => {

            document.removeEventListener(
                "mousedown",
                handleOutside
            );

        };

    }, [open]);


    const openDropdown = () => {

        const index =
            allOptions.findIndex(
                (option) =>
                    option.value === value
            );

        setActiveIndex(
            index >= 0
                ? index
                : 0
        );

        setOpen(true);

    };


    const closeAndFocusTrigger = () => {

        setOpen(false);

        triggerRef.current?.focus();

    };


    const selectOption = (
        option
    ) => {

        onChange(option.value);

        closeAndFocusTrigger();

    };


    const handleListKeyDown = (
        event
    ) => {

        if (event.key === "Escape") {

            event.preventDefault();

            closeAndFocusTrigger();

            return;

        }


        if (event.key === "ArrowDown") {

            event.preventDefault();

            setActiveIndex(
                (index) =>
                    Math.min(
                        index + 1,
                        allOptions.length - 1
                    )
            );

            return;

        }


        if (event.key === "ArrowUp") {

            event.preventDefault();

            setActiveIndex(
                (index) =>
                    Math.max(
                        index - 1,
                        0
                    )
            );

            return;

        }


        if (
            event.key === "Enter" ||
            event.key === " "
        ) {

            event.preventDefault();

            if (allOptions[activeIndex]) {

                selectOption(
                    allOptions[activeIndex]
                );

            }

        }

    };


    return (

        <div className="filter-dropdown" ref={rootRef}>

            <button
                type="button"
                id={id}
                ref={triggerRef}
                className={
                    "filter-dropdown-trigger" +
                    (open ? " is-open" : "")
                }
                aria-haspopup="listbox"
                aria-expanded={open}
                disabled={disabled}
                onClick={() => {

                    if (open) {
                        setOpen(false);
                    }
                    else {
                        openDropdown();
                    }

                }}
                onKeyDown={(event) => {

                    if (event.key === "ArrowDown") {

                        event.preventDefault();

                        openDropdown();

                    }

                }}
            >

                <span className="filter-dropdown-trigger-label">

                    {disabled && loadingLabel
                        ? loadingLabel
                        : (selected?.label || placeholder)
                    }

                </span>

                <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path
                        d="M5.5 7.5l4.5 4.5 4.5-4.5"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                </svg>

            </button>


            {open && (

                <ul
                    className="filter-dropdown-list"
                    role="listbox"
                    tabIndex={-1}
                    ref={listRef}
                    aria-activedescendant={
                        activeIndex >= 0
                            ? `${id}-option-${activeIndex}`
                            : undefined
                    }
                    onKeyDown={handleListKeyDown}
                >

                    {allOptions.map(
                        (option, index) => (

                            <li
                                key={
                                    option.value ||
                                    "placeholder"
                                }
                                id={`${id}-option-${index}`}
                                role="option"
                                aria-selected={option.value === value}
                                className={
                                    "filter-dropdown-option" +
                                    (option.value === value
                                        ? " is-selected"
                                        : "") +
                                    (index === activeIndex
                                        ? " is-active"
                                        : "")
                                }
                                onMouseEnter={() =>
                                    setActiveIndex(index)
                                }
                                onClick={() =>
                                    selectOption(option)
                                }
                            >

                                {option.label}

                            </li>

                        )
                    )}

                </ul>

            )}

        </div>

    );

}


// ============================================================
// RESULT TAGS (topics + keywords)
// ============================================================

function getResultTags(article) {

    const topics =
        Array.isArray(article.topics)
            ? article.topics
            : [];

    const keywords =
        Array.isArray(article.keywords)
            ? article.keywords
            : [];

    return [
        ...topics,
        ...keywords,
    ]
        .filter(Boolean)
        .slice(0, 3);

}


// ============================================================
// PAGE NUMBERS (with ellipsis windowing)
// ============================================================

function getPageNumbers(current, total) {

    const pages = [];

    const delta = 1;

    const start = Math.max(2, current - delta);
    const end = Math.min(total - 1, current + delta);

    pages.push(1);

    if (start > 2) {
        pages.push("…");
    }

    for (let i = start; i <= end; i++) {
        pages.push(i);
    }

    if (end < total - 1) {
        pages.push("…");
    }

    if (total > 1) {
        pages.push(total);
    }

    return pages;

}


// ============================================================
// HIGHLIGHT SEARCH TERM
// ============================================================

function escapeRegExp(value) {

    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

}

function highlightMatch(text, query) {

    if (!text) {
        return text;
    }

    const trimmed = (query || "").trim();

    if (!trimmed) {
        return text;
    }

    const parts = text.split(
        new RegExp(`(${escapeRegExp(trimmed)})`, "ig")
    );

    if (parts.length === 1) {
        return text;
    }

    return parts.map(
        (part, index) => (
            part.toLowerCase() === trimmed.toLowerCase()
                ? <mark className="search-highlight" key={index}>{part}</mark>
                : part
        )
    );

}


// ============================================================
// BUILD IMAGE URL
// ============================================================

function buildUrl(
    url
) {

    if (!url) {

        return "";

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
        `${import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"}${url}`
    );

}


// ============================================================
// FORMAT DATE
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


export default Search;
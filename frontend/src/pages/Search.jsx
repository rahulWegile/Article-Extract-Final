import {
    useEffect,
    useState,
} from "react";

import {
    useNavigate,
    useSearchParams,
} from "react-router-dom";

import api from "../api/api";

import "../styles/archive.css";


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
        });

    };


    // ========================================================
    // SEARCH / BROWSE
    // ========================================================

    const performSearch = async ({
        suppliedQuery = null,
        suppliedNewspaper = null,
        suppliedDate = null,
        append = false,
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


        // ----------------------------------------------------
        // No filters at all means "browse the entire archive" --
        // the backend already treats an empty query and no
        // newspaper/date as "match everything", so fetch it
        // instead of bailing out.
        // ----------------------------------------------------

        const offset =
            append
                ? results.length
                : 0;


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

                limit: 100,

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


            const combinedResults =
                append
                    ? [
                        ...results,
                        ...articles,
                    ]
                    : articles;


            setResults(
                combinedResults
            );


            setTotal(
                data.total || 0
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
                        combinedResults,
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

            performSearch();

        }

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

                    <span>
                        Newspaper Archive
                    </span>

                </div>


                <div className="archive-header-search">

                    <input
                        value={query}
                        placeholder="Search articles..."
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
                            performSearch()
                        }
                    >
                        🔍
                    </button>

                </div>


                <nav className="archive-nav">

                    <button
                        onClick={() =>
                            navigate("/")
                        }
                    >
                        Home
                    </button>


                    <button
                        className="active"
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

                    <input
                        value={query}
                        placeholder="Search articles, people, topics and keywords..."
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
                            performSearch()
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

                <section className="archive-filters">

                    {/* =============================================
                        NEWSPAPER
                        ============================================= */}

                    <div className="archive-filter">

                        <label
                            htmlFor="newspaper-select"
                        >
                            Newspaper
                        </label>


                        <select
                            id="newspaper-select"
                            value={
                                selectedNewspaper
                            }
                            onChange={
                                handleNewspaperChange
                            }
                            disabled={
                                loadingNewspapers
                            }
                        >

                            <option value="">
                                All newspapers
                            </option>


                            {newspapers.map(
                                (newspaper) => (

                                    <option
                                        key={
                                            newspaper
                                        }
                                        value={
                                            newspaper
                                        }
                                    >
                                        {
                                            newspaper
                                        }
                                    </option>

                                )
                            )}

                        </select>

                    </div>


                    {/* =============================================
                        PUBLICATION DATE
                        ============================================= */}

                    <div className="archive-filter">

                        <label
                            htmlFor="publish-date-select"
                        >
                            Publish Date
                        </label>


                        <select
                            id="publish-date-select"
                            value={
                                selectedDate
                            }
                            onChange={
                                handleDateChange
                            }
                            disabled={
                                loadingDates
                            }
                        >

                            <option value="">

                                {loadingDates
                                    ? "Loading dates..."
                                    : "All dates"
                                }

                            </option>


                            {publishDates.map(
                                (date) => (

                                    <option
                                        key={date}
                                        value={date}
                                    >
                                        {
                                            formatDate(
                                                date
                                            )
                                        }
                                    </option>

                                )
                            )}

                        </select>

                    </div>


                    {/* =============================================
                        FILTER BUTTON
                        ============================================= */}

                    <button
                        className="archive-filter-button"
                        onClick={() =>
                            performSearch()
                        }
                        disabled={
                            loading
                        }
                    >

                        {loading
                            ? "Loading..."
                            : "Show Articles"
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

                            <span>
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

                            <span>
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

                    <div className="search-error">

                        {error}

                    </div>

                )}


                {/* =================================================
                    RESULTS
                    ================================================= */}

                {searched && !loading && (

                    <section className="search-results">

                        <div className="results-header">

                            <h2>

                                {total}

                                {" "}

                                {total === 1
                                    ? "article"
                                    : "articles"
                                }

                            </h2>


                            <span>

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
                                        : "found"
                                }

                            </span>

                        </div>


                        {results.length === 0 ? (

                            <div className="no-results">

                                <h2>
                                    No articles found
                                </h2>

                                <p>

                                    {selectedNewspaper
                                        ? "No articles are available for the selected newspaper and date."
                                        : "Try another name, topic or keyword."
                                    }

                                </p>

                            </div>

                        ) : (

                            results.map(
                                (
                                    article,
                                    index
                                ) => (

                                    <article
                                        key={
                                            `${article.document_id}-${article.logical_article_id}`
                                        }
                                        className="search-result-card"
                                        onClick={() =>
                                            openArticle(
                                                article
                                            )
                                        }
                                    >

                                        <div className="result-content">

                                            <div className="result-number">

                                                {String(
                                                    index + 1
                                                ).padStart(
                                                    2,
                                                    "0"
                                                )}

                                            </div>


                                            <div className="result-main">

                                                <h3>
                                                    {
                                                        article.title
                                                    }
                                                </h3>


                                                {article.author && (

                                                    <p className="result-author">

                                                        {
                                                            article.author
                                                        }

                                                    </p>

                                                )}


                                                <div className="result-meta">

                                                    {article.display_date && (

                                                        <span>
                                                            {formatDate(
                                                                article.display_date
                                                            )}
                                                        </span>

                                                    )}


                                                    {article.category && (

                                                        <span className="category-tag">

                                                            {
                                                                article.category
                                                            }

                                                        </span>

                                                    )}

                                                </div>


                                                {article.summary && (

                                                    <p className="result-summary">

                                                        {
                                                            article.summary
                                                        }

                                                    </p>

                                                )}

                                            </div>

                                        </div>


                                        {article.images &&
                                            article.images.length >
                                                0 &&
                                            article.images[0].url && (

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

                                            )}

                                    </article>

                                )
                            )

                        )}


                        {/* =========================================
                            LOAD MORE
                            ========================================= */}

                        {results.length > 0 &&
                            results.length < total && (

                                <button
                                    className="archive-filter-button load-more-button"
                                    onClick={() =>
                                        performSearch({
                                            append: true,
                                        })
                                    }
                                    disabled={
                                        loading
                                    }
                                >

                                    {loading
                                        ? "Loading..."
                                        : `Load More (${
                                            results.length
                                        } of ${
                                            total
                                        })`
                                    }

                                </button>

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
        `${import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"}${url}`
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
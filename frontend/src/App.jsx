import {
    Routes,
    Route,
} from "react-router-dom";


import Home from "./pages/Home";

import Viewer from "./pages/Viewer";

import Search from "./pages/Search";

import ArticleDetail from "./pages/ArticleDetail";


function App() {

    return (

        <Routes>

            {/* =================================================
                EXISTING HOME
                ================================================= */}

            <Route
                path="/"
                element={
                    <Home />
                }
            />


            {/* =================================================
                EXISTING ARTICLE BOUNDARY VIEWER

                DO NOT REMOVE
                ================================================= */}

            <Route
                path="/viewer/:documentId"
                element={
                    <Viewer />
                }
            />


            {/* =================================================
                SEARCH
                ================================================= */}

            <Route
                path="/search"
                element={
                    <Search />
                }
            />


            {/* =================================================
                SINGLE ARTICLE
                ================================================= */}

            <Route
                path="/search/article/:documentId/:logicalArticleId"
                element={
                    <ArticleDetail />
                }
            />

        </Routes>

    );

}


export default App;
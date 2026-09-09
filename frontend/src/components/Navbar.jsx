import { Link, useLocation } from "react-router-dom";

function Navbar() {

    const location = useLocation();
    const onSearch = location.pathname.startsWith("/search");

    return (

        <header className="app-header">

            <div className="app-header-inner">

                <Link to="/" className="app-logo">

                    <span className="app-logo-mark" aria-hidden="true">
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

                    <span className="app-logo-text">

                        <strong>Newspaper Studio</strong>
                        <small>Digitize &amp; structure archives</small>

                    </span>

                </Link>

                <nav className="app-header-nav">

                    <Link
                        to="/"
                        className={"nav-link" + (!onSearch ? " active" : "")}
                    >
                        Documents
                    </Link>

                    <Link
                        to="/search"
                        className={"nav-link nav-link-cta" + (onSearch ? " active" : "")}
                    >
                        View Articles
                    </Link>

                </nav>

            </div>

        </header>

    );

}

export default Navbar;

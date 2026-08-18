from datetime import date

from fastapi import APIRouter, HTTPException, Query

from backend.services.article_search import (
    ArticleSearchService,
)


router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


search_service = ArticleSearchService()


# ============================================================
# GET ALL NEWSPAPERS
# ============================================================

@router.get("/newspapers")
def get_newspapers():
    """
    Return all newspapers available in the archive.

    Example:
        GET /search/newspapers
    """

    return search_service.get_newspapers()


# ============================================================
# GET PUBLICATION DATES
# ============================================================

@router.get("/dates")
def get_publish_dates(
    newspaper: str | None = Query(
        None,
        description="Newspaper name",
    ),
):
    """
    Return publication dates.

    If newspaper is provided, only dates belonging
    to that newspaper are returned.

    Examples:

        GET /search/dates

        GET /search/dates?newspaper=The%20Times%20of%20India
    """

    return search_service.get_publish_dates(
        newspaper=newspaper,
    )


# ============================================================
# SEARCH / BROWSE ARTICLES
# ============================================================

@router.get("/articles")
def search_articles(
    q: str | None = Query(
        None,
        description="Search newspaper articles",
    ),

    newspaper: str | None = Query(
        None,
        description="Newspaper name",
    ),

    publish_date: date | None = Query(
        None,
        description="Newspaper publication date",
    ),

    limit: int = Query(
        20,
        ge=1,
        le=100,
    ),

    offset: int = Query(
        0,
        ge=0,
    ),
):
    """
    Search the newspaper archive.

    Examples:

        Search text:

        GET /search/articles?q=Mamata


        Search by newspaper:

        GET /search/articles?
            newspaper=The%20Times%20of%20India


        Search by newspaper and date:

        GET /search/articles?
            newspaper=The%20Times%20of%20India&
            publish_date=2026-08-04


        Browse an entire newspaper edition:

        GET /search/articles?
            newspaper=THE%20ASIAN%20AGE&
            publish_date=2026-07-14
    """

    return search_service.search(
        query=q,
        newspaper=newspaper,
        publish_date=(
            publish_date.isoformat()
            if publish_date
            else None
        ),
        limit=limit,
        offset=offset,
    )


# ============================================================
# GET ONE LOGICAL ARTICLE
# ============================================================

@router.get(
    "/articles/{document_id}/{logical_article_id}"
)
def get_article(
    document_id: str,
    logical_article_id: str,
):
    """
    Return exactly one logical article.

    Example:

        GET /search/articles/
            f3fe711b-caba-5ae2-a600-dbe6ccdb15ff/
            logical_0002
    """

    article = search_service.get_article(
        document_id=document_id,
        logical_article_id=logical_article_id,
    )

    if article is None:

        raise HTTPException(
            status_code=404,
            detail="Article not found",
        )

    return article
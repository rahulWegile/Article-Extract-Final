from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class DocumentSummary(BaseModel):
    document_id: str
    pdf_name: str
    page_count: int
    model_config = {"extra": "allow"}

class DocumentDetail(BaseModel):
    document_id: str
    pdf_name: str
    page_count: int
    model_config = {"extra": "allow"}

class PageResponse(BaseModel):
    document_id: str
    pdf_name: str
    page: int
    page_count: int
    image: str
    plain_image: str
    model_config = {"extra": "allow"}


class BoundaryBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ArticleBoundary(BaseModel):
    article_id: str
    bbox: BoundaryBox
    is_multi_page: bool
    # Precise sub-rectangles when boundary_decomposer.py reshaped this
    # article to avoid enclosing a neighbouring article's content
    # instead of leaving one rectangle that overlaps it -- empty for
    # every other article, which renders as `bbox` alone exactly as
    # before. See Article.sub_rects (pipeline/article/article_grouper.py).
    sub_rects: List[BoundaryBox] = []
    block_count: int = 0
    boundary_source: Optional[str] = None
    logical_article_id: Optional[str] = None
    model_config = {"extra": "allow"}


class PageBoundariesResponse(BaseModel):
    document_id: str
    document_uuid: str
    page: int
    boundaries: List[ArticleBoundary]


class BoundaryUpdateRequest(BaseModel):
    article_id: Optional[str] = None
    bbox: BoundaryBox


class BoundaryUpdateResponse(BaseModel):
    article_id: str
    bbox: BoundaryBox
    headline: Optional[str] = None
    article_text: Optional[str] = None
    is_new: bool
    db_synced: bool
    db_error: Optional[str] = None
    model_config = {"extra": "allow"}


class BoundaryDeleteResponse(BaseModel):
    article_id: str
    deleted: bool
    db_synced: bool
    db_error: Optional[str] = None


class BoundaryMergeRequest(BaseModel):
    article_ids: List[str]


class BoundaryMergeResponse(BaseModel):
    article_id: str
    bbox: BoundaryBox
    headline: Optional[str] = None
    article_text: Optional[str] = None
    merged_from: List[str]
    removed_article_ids: List[str]
    db_synced: bool
    db_error: Optional[str] = None
    model_config = {"extra": "allow"}

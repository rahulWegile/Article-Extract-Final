from pipeline.article_growth.current_article import (
    CurrentArticle,
)


class Block:

    def __init__(

        self,

        block_id,

        bottom,

    ):

        self.id = block_id

        self.y2 = bottom


title = Block(

    5,

    180,

)

article = CurrentArticle(

    article_id=1,

    nodes=[5],

    visited={5},

    last_block=title,

    bottom=180,

    column=1,

)

body = Block(

    6,

    260,

)

article.add_block(

    body,

)

print()

print("=" * 60)

print(article)

print()

print("=" * 60)
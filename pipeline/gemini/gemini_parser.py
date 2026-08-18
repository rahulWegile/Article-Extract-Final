class GeminiParser:
    """
    Parse Gemini response.

    Responsibilities
    ----------------
    ✓ Assign semantic roles to every LayoutBlock
    ✓ Return parsed article groups
    """

    def parse(
        self,
        response,
        blocks,
    ):

        print()
        print("=" * 60)
        print("GEMINI PARSER")
        print("=" * 60)

        # -------------------------------------------------
        # Build lookup
        # -------------------------------------------------

        block_lookup = {
            block.id: block
            for block in blocks
        }

        # -------------------------------------------------
        # Assign semantic role to every block
        # -------------------------------------------------

        roles_assigned = 0

        for item in response.get("blocks", []):

            block_id = item["id"]

            role = item["role"]

            if block_id not in block_lookup:
                continue

            block = block_lookup[block_id]

            block.role = role

            roles_assigned += 1

        print(f"Roles Assigned : {roles_assigned}")

        # -------------------------------------------------
        # Build parsed response
        # -------------------------------------------------

        parsed_articles = []

        for article in response.get("articles", []):

            parsed_articles.append(

                {
                    "article_id": article["article_id"],
                    "blocks": article["blocks"],
                }

            )

        print(f"Articles Parsed : {len(parsed_articles)}")

        print("=" * 60)
        print()

        return {

            "articles": parsed_articles,

            "blocks": blocks,

        }
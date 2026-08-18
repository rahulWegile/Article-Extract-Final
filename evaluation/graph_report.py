from statistics import mean, median


class GraphReport:

    """
    Reports graph quality metrics.

    Pure evaluation module.
    """

    def report(
        self,
        articles,
        blocks,
    ):

        sizes = [
            len(article["nodes"])
            for article in articles
        ]

        total_articles = len(articles)

        single = sum(
            size == 1
            for size in sizes
        )

        double = sum(
            size == 2
            for size in sizes
        )

        triple = sum(
            size == 3
            for size in sizes
        )

        large = sum(
            size >= 4
            for size in sizes
        )

        largest = max(sizes) if sizes else 0

        average = mean(sizes) if sizes else 0

        med = median(sizes) if sizes else 0

        # --------------------------------------
        # Heading statistics
        # --------------------------------------

        heading_blocks = 0

        connected_headings = 0

        for article in articles:

            article_blocks = [
                blocks[idx]
                for idx in article["nodes"]
            ]

            has_heading = False

            for block in article_blocks:

                if (
                    block.knowledge.heading_probability
                    > 0.80
                ):
                    has_heading = True
                    heading_blocks += 1

            if has_heading and len(article_blocks) > 1:
                connected_headings += 1

        # --------------------------------------
        # Caption statistics
        # --------------------------------------

        caption_blocks = 0

        connected_captions = 0

        for article in articles:

            article_blocks = [
                blocks[idx]
                for idx in article["nodes"]
            ]

            has_caption = False

            for block in article_blocks:

                if (
                    block.knowledge.caption_probability
                    > 0.80
                ):
                    has_caption = True
                    caption_blocks += 1

            if has_caption and len(article_blocks) > 1:
                connected_captions += 1

        # --------------------------------------

        print()

        print("=" * 60)
        print("GRAPH REPORT")
        print("=" * 60)

        print(f"Total Blocks           : {len(blocks)}")
        print(f"Detected Articles      : {total_articles}")

        print()

        print(f"Single Block Articles  : {single}")
        print(f"Two Block Articles     : {double}")
        print(f"Three Block Articles   : {triple}")
        print(f"Four+ Block Articles   : {large}")

        print()

        print(f"Largest Article        : {largest}")
        print(f"Average Article Size   : {average:.2f}")
        print(f"Median Article Size    : {med:.2f}")

        print()

        print(
            f"Headings Connected     : "
            f"{connected_headings}/{heading_blocks}"
        )

        print(
            f"Captions Connected     : "
            f"{connected_captions}/{caption_blocks}"
        )

        print("=" * 60)
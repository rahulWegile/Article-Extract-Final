class PipelineComparison:
    """
    Compare Graph Pipeline and Seed Pipeline.
    """

    def report(
        self,
        graph_articles,
        seed_articles,
    ):

        print()
        print("=" * 60)
        print("PIPELINE COMPARISON")
        print("=" * 60)

        #
        # Article Count
        #

        print(f"Graph Articles : {len(graph_articles)}")
        print(f"Seed Articles  : {len(seed_articles)}")

        #
        # Single Block Articles
        #

        graph_single = sum(
            len(article["nodes"]) == 1
            for article in graph_articles
        )

        seed_single = sum(
            len(article.nodes) == 1
            for article in seed_articles
        )

        print()
        print(f"Graph Single Block : {graph_single}")
        print(f"Seed Single Block  : {seed_single}")

        #
        # Average Article Size
        #

        graph_average = (
            sum(len(article["nodes"]) for article in graph_articles)
            / len(graph_articles)
            if graph_articles else 0
        )

        seed_average = (
            sum(len(article.nodes) for article in seed_articles)
            / len(seed_articles)
            if seed_articles else 0
        )

        print()
        print(f"Graph Average Size : {graph_average:.2f}")
        print(f"Seed Average Size  : {seed_average:.2f}")

        #
        # Largest Article
        #

        graph_largest = max(
            (len(article["nodes"]) for article in graph_articles),
            default=0,
        )

        seed_largest = max(
            (len(article.nodes) for article in seed_articles),
            default=0,
        )

        print()
        print(f"Graph Largest : {graph_largest}")
        print(f"Seed Largest  : {seed_largest}")

        #
        # Fragmentation Reduction
        #

        reduction = graph_single - seed_single

        print()
        print(f"Fragmentation Reduction : {reduction}")

        print("=" * 60)
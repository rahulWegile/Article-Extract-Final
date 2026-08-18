from statistics import mean


class EdgeThresholdEvaluator:
    """
    Evaluate different edge-selection thresholds
    without changing the graph algorithm.
    """

    def evaluate(
        self,
        scored_edges,
        selector,
        graph_builder,
        article_builder,
        blocks,
        thresholds,
    ):

        print()
        print("=" * 80)
        print("EDGE THRESHOLD EVALUATION")
        print("=" * 80)

        print(
            f"{'Threshold':<12}"
            f"{'Edges':<10}"
            f"{'Articles':<12}"
            f"{'Singles':<10}"
            f"{'Avg Size':<10}"
        )

        print("-" * 80)

        for threshold in thresholds:

            selector.fixed_threshold = threshold

            selected = selector.select(
                scored_edges
            )

            graph = graph_builder.build(
                blocks,
                selected,
            )

            articles = article_builder.build(
                graph
            )

            sizes = [

                len(article["nodes"])

                for article in articles

            ]

            singles = sum(
                size == 1
                for size in sizes
            )

            avg = (
                mean(sizes)
                if sizes else 0
            )

            print(

                f"{threshold:<12.2f}"

                f"{len(selected):<10}"

                f"{len(articles):<12}"

                f"{singles:<10}"

                f"{avg:<10.2f}"

            )

        print("=" * 80)
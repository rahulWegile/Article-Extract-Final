from evaluation.metrics import EvaluationMetrics


class Benchmark:

    def evaluate(
        self,
        page_number,
        candidate_edges,
        selected_edges,
        graph,
        articles,
    ):

        metrics = EvaluationMetrics()

        metrics.candidate_edges = len(candidate_edges)

        metrics.selected_edges = len(selected_edges)

        metrics.graph_edges = graph.number_of_edges()

        metrics.detected_articles = len(articles)

        #
        # Ground-truth metrics will be added later
        #

        metrics.precision = 0.0

        metrics.recall = 0.0

        metrics.fragmentation = 0.0

        metrics.incorrect_merges = 0

        return metrics
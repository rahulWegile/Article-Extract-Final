from dataclasses import dataclass


@dataclass
class EvaluationMetrics:

    candidate_edges: int = 0

    selected_edges: int = 0

    graph_edges: int = 0

    detected_articles: int = 0

    precision: float = 0.0

    recall: float = 0.0

    fragmentation: float = 0.0

    incorrect_merges: int = 0
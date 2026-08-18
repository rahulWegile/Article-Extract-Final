from evaluation.edge_threshold_evaluator import (
    EdgeThresholdEvaluator,
)


class MockSelector:

    def __init__(self):
        self.threshold = 0.0

    def select(self, edges):

        return [

            e

            for e in edges

            if e >= self.threshold

        ]


class MockGraphBuilder:

    def build(
        self,
        blocks,
        selected,
    ):

        return selected


class MockArticleBuilder:

    def build(
        self,
        graph,
    ):

        return [

            {

                "nodes": [0]

            }

            for _ in graph

        ]


edges = [

    0.5,
    0.7,
    0.9,
    1.1,

]

EdgeThresholdEvaluator().evaluate(

    scored_edges=edges,

    selector=MockSelector(),

    graph_builder=MockGraphBuilder(),

    article_builder=MockArticleBuilder(),

    blocks=[],

    thresholds=[

        0.5,
        0.7,
        0.9,
        1.1,

    ],

)
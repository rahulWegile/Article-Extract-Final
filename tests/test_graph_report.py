from evaluation.graph_report import GraphReport


class Knowledge:

    def __init__(self, heading=0, caption=0):

        self.heading_probability = heading
        self.caption_probability = caption


class Block:

    def __init__(self, heading=0, caption=0):

        self.knowledge = Knowledge(
            heading,
            caption,
        )


blocks = [

    Block(heading=0.95),

    Block(),

    Block(),

    Block(caption=0.95),

]

articles = [

    {

        "id": 1,

        "nodes": [0, 1],

    },

    {

        "id": 2,

        "nodes": [2],

    },

    {

        "id": 3,

        "nodes": [3],

    },

]

GraphReport().report(
    articles,
    blocks,
)
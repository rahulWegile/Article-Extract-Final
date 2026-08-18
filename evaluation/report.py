import csv
from pathlib import Path


class EvaluationReport:

    HEADER = [

        "page",

        "candidate_edges",

        "selected_edges",

        "graph_edges",

        "detected_articles",

        "precision",

        "recall",

        "fragmentation",

        "incorrect_merges",

    ]

    def save(
        self,
        page_number,
        metrics,
    ):

        output_dir = Path("evaluation")

        output_dir.mkdir(exist_ok=True)

        csv_file = output_dir / "metrics.csv"

        write_header = not csv_file.exists()

        with open(
            csv_file,
            "a",
            newline="",
        ) as f:

            writer = csv.writer(f)

            if write_header:
                writer.writerow(self.HEADER)

            writer.writerow([

                page_number,

                metrics.candidate_edges,

                metrics.selected_edges,

                metrics.graph_edges,

                metrics.detected_articles,

                metrics.precision,

                metrics.recall,

                metrics.fragmentation,

                metrics.incorrect_merges,

            ])

        print(f"Saved evaluation -> {csv_file}")
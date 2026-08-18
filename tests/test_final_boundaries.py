from pathlib import Path

from pipeline.gemini.gemini_boundary_pipeline import (
    GeminiBoundaryPipeline,
)

PAGE_DIR = Path("output/pages")
JSON_DIR = Path("output/page_json")
RESPONSE_DIR = Path("output/gemini")
OUTPUT_DIR = Path("output/final")


def main():

    print()
    print("=" * 60)
    print("BUILD FINAL BOUNDARIES")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    response_files = sorted(
        RESPONSE_DIR.glob("page_*_response.json")
    )

    print(f"Found {len(response_files)} Gemini responses")

    for response_path in response_files:

        page_name = response_path.stem.replace(
            "_response",
            "",
        )

        print()
        print("=" * 60)
        print(page_name.upper())
        print("=" * 60)

        json_path = JSON_DIR / f"{page_name}.json"

        image_path = PAGE_DIR / f"{page_name}.png"

        output_path = OUTPUT_DIR / (
            f"{page_name}_final_boundaries.png"
        )

        GeminiBoundaryPipeline().run(

            image_path=str(image_path),

            page_json_path=str(json_path),

            gemini_response_path=str(response_path),

            output_path=str(output_path),

        )

        print(f"✓ Saved -> {output_path}")

    print()
    print("=" * 60)
    print("ALL PAGES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
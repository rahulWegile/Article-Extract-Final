import json
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline.gemini.gemini_service import GeminiService
from pipeline.languages.hindi.grouping_prompt import (
    HINDI_GROUPING_PROMPT,
)


PAGE_DIR = Path("output/pages")
JSON_DIR = Path("output/page_json")
OUTPUT_DIR = Path("output/gemini")


def main():

    print()
    print("=" * 60)
    print("TEST GEMINI SERVICE")
    print("=" * 60)

    # -----------------------------------------------------
    # Load Environment
    # -----------------------------------------------------

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        print("✗ GEMINI_API_KEY NOT FOUND")
        return

    print("✓ GEMINI_API_KEY found")

    # -----------------------------------------------------
    # Initialize Gemini
    # -----------------------------------------------------

    print()
    print("Initializing Gemini Service...")

    service = GeminiService()

    print("✓ Gemini initialized")

    # -----------------------------------------------------
    # Find all exported pages
    # -----------------------------------------------------

    json_files = sorted(JSON_DIR.glob("page_*.json"))

    if len(json_files) == 0:

        print()
        print("No page JSON files found.")
        return

    print()
    print(f"Found {len(json_files)} pages")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Process every page
    # -----------------------------------------------------

    success = 0
    failed = 0

    for json_path in json_files:

        page_name = json_path.stem

        image_path = PAGE_DIR / f"{page_name}.png"

        output_path = OUTPUT_DIR / f"{page_name}_response.json"

        print()
        print("=" * 60)
        print(page_name.upper())
        print("=" * 60)

        if not image_path.exists():

            print(f"Image not found : {image_path}")

            failed += 1

            continue

        try:

            response = service.analyze_page(

                image_path=str(image_path),

                json_path=str(json_path),

                prompt=HINDI_GROUPING_PROMPT,

            )

            with open(

                output_path,

                "w",

                encoding="utf-8",

            ) as f:

                json.dump(

                    response,

                    f,

                    indent=4,

                    ensure_ascii=False,

                )

            print(
                f"✓ Saved -> {output_path}"
            )

            success += 1

        except Exception as e:

            print()
            print("FAILED")

            print(type(e).__name__)

            print(e)

            failed += 1

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Successful : {success}")
    print(f"Failed     : {failed}")

    print("=" * 60)


if __name__ == "__main__":

    main()
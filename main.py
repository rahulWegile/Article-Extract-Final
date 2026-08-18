from pipeline.render_pdf import render_pdf
from pipeline.layout_detector import LayoutDetector
from pipeline.ocr.easyocr_engine import EasyOCREngine
#from pipeline.page_processor import process_page
from pipeline.page_processor_gemini import process_page

def main():

    print("=" * 60)
    print("NEWSPAPER ARTICLE BOUNDARY DETECTOR")
    print("=" * 60)

    print("\n[1] Rendering PDF...")

    pages = render_pdf(
        pdf_path="input/newspaper.pdf",
        output_dir="output/pages",
        dpi=300,
    )

    print(f"Rendered {len(pages)} pages")

    print("\n[2] Loading Models...")

    detector = LayoutDetector()

    ocr_engine = EasyOCREngine()

    print("✓ All models loaded")

    for page_number, page_path in enumerate(pages, start=1):

        process_page(
            page_number=page_number,
            page_path=page_path,
            detector=detector,
            ocr_engine=ocr_engine,
        )

    print("\n" + "=" * 60)
    print("Pipeline Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
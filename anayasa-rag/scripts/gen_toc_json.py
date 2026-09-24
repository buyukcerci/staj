import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.text_extractor import extract_text_from_pdf
from src.toc_extractor import extract_toc


def main():
    if len(sys.argv) < 2:
        pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "anayasa.pdf")
    else:
        pdf_path = sys.argv[1]

    text = extract_text_from_pdf(pdf_path)
    toc = extract_toc(text)
    print(json.dumps(toc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

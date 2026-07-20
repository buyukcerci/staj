import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.text_extracter import extract_text_from_pdf

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <pdf_path>")
    sys.exit(1)

pdf_path = sys.argv[1]
text = extract_text_from_pdf(pdf_path)
print(text)

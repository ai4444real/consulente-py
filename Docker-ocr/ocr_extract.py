import sys
import pytesseract
from pdf2image import convert_from_path

def run_ocr(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image, lang="ita")
    return text

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ocr_extract.py <pdf_file>")
        sys.exit(1)
    
    result = run_ocr(sys.argv[1])
    print(result)

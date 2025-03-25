import pytesseract
from pdf2image import convert_from_path
import os

PDF_PATH = os.path.join("sample_pdfs", "test_ocr.pdf")

def extract_text_from_scanned_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    full_text = ""

    for i, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img, lang="ita")
        full_text += f"\n📄 Pagina {i}:\n{text.strip()}\n"

    return full_text

if __name__ == "__main__":
    print("🧠 Estrazione testo OCR dal PDF...\n")
    result = extract_text_from_scanned_pdf(PDF_PATH)
    print(result)

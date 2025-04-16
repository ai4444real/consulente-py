from pathlib import Path
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from tempfile import TemporaryDirectory

def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyMuPDF (for digitally created PDFs)."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)

def extract_ocr(pdf_path: str, dpi: int = 300) -> str:
    """Extract text from a scanned PDF using OCR."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    with TemporaryDirectory() as temp_dir:
        images = convert_from_path(str(path), dpi=dpi, output_folder=temp_dir)
        return "\n".join(pytesseract.image_to_string(img) for img in images)

def read_pdf(pdf_path: str, dpi: int = 300, threshold: int = 20) -> str:
    try:
        text = extract_text(pdf_path)
        if len(text.strip()) >= threshold:
            return text
    except Exception:
        pass

    try:
        return extract_ocr(pdf_path, dpi=dpi)
    except Exception as e:
        raise RuntimeError("Failed to extract text from PDF by any method.") from e

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract.py <pdf_path>")
    else:
        pdf_path = sys.argv[1]
        result = read_pdf(pdf_path)
        print(result if result else "❌ Nessun contenuto estratto.")

import pytest
import sys
import os
import fitz

# Aggiunge il percorso del progetto ai sys.path per permettere l'import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pdf_extractor.extract import extract_text_from_pdf

TEST_PDF_PATH = "pdf_extractor/tests/sample_test.pdf"

def create_test_pdf():
    """Crea un piccolo PDF di test con testo noto."""
    doc = fitz.open()
    page = doc.new_page()
    test_text = "Questo è un test di estrazione PDF.\nLinea 2 del testo."
    page.insert_text((50, 100), test_text)  # Scrive il testo nel PDF
    doc.save(TEST_PDF_PATH)
    doc.close()
    return test_text  # Restituisce il testo atteso

@pytest.fixture(scope="function", autouse=True)
def setup_and_cleanup():
    """Genera il PDF prima del test e lo elimina dopo."""
    expected_text = create_test_pdf()
    yield expected_text  # Il test riceverà questo valore
    if os.path.exists(TEST_PDF_PATH):
        os.remove(TEST_PDF_PATH)  # Pulizia finale

def test_extract_text(setup_and_cleanup):
    """Verifica che il testo estratto dal PDF corrisponda a quello atteso."""
    expected_text = expected_text = "📄 Pagina 1:\n" + setup_and_cleanup
    extracted_text = extract_text_from_pdf(TEST_PDF_PATH).strip()  # Rimuove spazi extra

    # Mostra SEMPRE il testo estratto, anche se il test passa
    print("\n📄 Testo atteso:\n", expected_text)
    print("\n📜 Testo estratto:\n", extracted_text)

    assert extracted_text == expected_text, "❌ Il testo estratto non corrisponde a quello atteso!"
    print("✅ Test superato: il testo estratto è corretto!")

def test_extract_text_ocr_like():
    """Verifica che il PDF OCR-like venga processato ma restituisca poco o nessun testo."""
    ocr_pdf_path = "pdf_extractor/sample_pdfs/test_ocr.pdf"
    extracted_text = extract_text_from_pdf(ocr_pdf_path).strip()

    print("\n📄 Test OCR-like:")
    print(extracted_text if extracted_text else "⚠️ Nessun testo trovato (possibile PDF immagine)")

    if len(extracted_text) < 30:
        print("⚠️ PDF probabilmente immagine: OCR richiesto per leggerlo.")
    else:
        print("✅ Testo leggibile: OCR non necessario.")

    assert len(extracted_text) < 30, "❌ Il PDF OCR-like ha restituito troppo testo, forse non è un'immagine"

import sys
from extract import read_pdf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_reader.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    try:
        result = read_pdf(pdf_path)
        print("\n=== Estratto dal PDF ===\n")
        print(result)
    except Exception as e:
        print(f"\n❌ Errore durante la lettura del PDF: {e}")

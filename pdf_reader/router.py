from fastapi import APIRouter, UploadFile, File
from pdf_reader.extract import read_pdf

router = APIRouter(prefix="/pdf")

@router.post("/read")
async def read_pdf_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = "temp_uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(contents)
    try:
        result = read_pdf(temp_path)
        return {"content": result}
    except Exception as e:
        return {"error": str(e)}

from fastapi.responses import JSONResponse
import pdfplumber
import tempfile
import re

@router.post("/extract/creditcard-transactions")
async def extract_creditcard_transactions_from_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    transaction_start_re = re.compile(r"^(\d{2}\.\d{2}\.\d{2})\s+\d{2}\.\d{2}\.\d{2}\s+(.+?)\s+([\d'\.]+)$")
    single_date_line = re.compile(r"^\d{2}\.\d{2}\.\d{2}\s+.+[\d'\.]+$")

    parsed = []

    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if single_date_line.match(line) and not transaction_start_re.match(line):
                    continue  # ignora riporti con una sola data

                match = transaction_start_re.match(line)
                if match:
                    date = match.group(1)
                    description = match.group(2).strip()
                    chf_raw = match.group(3).replace("'", "")
                    try:
                        chf = float(chf_raw)
                    except ValueError:
                        continue
                    parsed.append({
                        "date": date,
                        "text": description,
                        "chf": chf
                    })

    return JSONResponse(content=parsed)

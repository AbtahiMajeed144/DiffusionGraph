import PyPDF2
import os
import glob

pdf_dir = "literatures"
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

for pdf_path in pdf_files:
    txt_path = pdf_path + ".txt"
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            # Extract first 6 pages for abstract/intro/theory
            for i in range(min(10, len(reader.pages))):
                page = reader.pages[i]
                t = page.extract_text()
                if t:
                    text += t + "\n"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extracted {pdf_path}")
    except Exception as e:
        print(f"Failed {pdf_path}: {e}")

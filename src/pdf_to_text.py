import pdfplumber
import os

pdf_folder = "data/pdf"
output_folder = "data/raw"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)

        text = ""

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            txt_name = filename.replace(".pdf", ".txt")

            with open(
                os.path.join(output_folder, txt_name),
                "w",
                encoding="utf-8"
            ) as f:
                f.write(text)

            print(f"Berhasil: {filename}")

        except Exception as e:
            print(f"Gagal {filename}: {e}")
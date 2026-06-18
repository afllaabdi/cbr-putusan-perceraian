import os
import re

INPUT_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/cleaned"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(INPUT_FOLDER, filename)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # ======================
    # LOWERCASE
    # ======================
    text = text.lower()

    # ======================
    # HAPUS DISCLAIMER MA
    # ======================
    text = re.sub(
        r'disclaimer kepaniteraan mahkamah agung.*?mahkamah agung ri melalui',
        ' ',
        text,
        flags=re.DOTALL
    )

    # ======================
    # HAPUS HEADER DIREKTORI
    # ======================
    text = re.sub(
        r'direktori putusan mahkamah agung republik indonesia',
        ' ',
        text
    )

    text = re.sub(
        r'putusan mahkamahagung go id',
        ' ',
        text
    )

    # ======================
    # HAPUS NOMOR HALAMAN
    # ======================
    text = re.sub(
        r'dari\s+\d+\s+halaman\s+putusan\s+nomor.*?pa kab mlg',
        ' ',
        text
    )

    # ======================
    # HAPUS KARAKTER ACAK
    # CONTOH:
    # a i s e n o d n i ...
    # ======================
    text = re.sub(
        r'(?:\b[a-z]\b\s*){10,}',
        ' ',
        text
    )

    # ======================
    # HAPUS URL
    # ======================
    text = re.sub(
        r'https?://\S+',
        ' ',
        text
    )

    # ======================
    # HAPUS KARAKTER ANEH
    # ======================
    text = re.sub(
        r'[^a-z0-9\s]',
        ' ',
        text
    )

    # ======================
    # NORMALISASI SPASI
    # ======================
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    text = text.strip()

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✓ Cleaned: {filename}")

print("\nSemua file selesai dibersihkan.")
import os
import re
import pandas as pd

INPUT_FOLDER = "data/cleaned"

data = []

for filename in os.listdir(INPUT_FOLDER):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(INPUT_FOLDER, filename)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # ======================
    # NOMOR PERKARA
    # ======================
    no_perkara = ""

    match = re.search(
        r'nomor\s+([\d]+\/pdt\.g\/\d{4}\/pa\.[a-z\.]+)',
        text,
        re.IGNORECASE
    )

    if match:
        no_perkara = match.group(1)

    # ======================
    # JENIS PERKARA
    # ======================
    jenis = ""

    if "cerai talak" in text:
        jenis = "Cerai Talak"

    elif "cerai gugat" in text:
        jenis = "Cerai Gugat"

    # ======================
    # PUTUSAN
    # ======================
    putusan = ""

    if "mengabulkan gugatan" in text:
        putusan = "Dikabulkan"

    elif "mengabulkan permohonan" in text:
        putusan = "Dikabulkan"

    elif "menolak gugatan" in text:
        putusan = "Ditolak"

    # ======================
    # AMAR PUTUSAN
    # ======================
    amar = ""

    amar_match = re.search(
        r'menjatuhkan.*?talak.*?(?:\.|,)',
        text,
        re.IGNORECASE
    )

    if amar_match:
        amar = amar_match.group(0)

    # ======================
    # ALASAN PERCERAIAN
    # ======================
    alasan = []

    keywords = [
        "nafkah",
        "perselisihan",
        "pertengkaran",
        "kekerasan",
        "hutang",
        "selingkuh",
        "perempuan lain",
        "meninggalkan rumah",
        "pisah tempat tinggal",
        "tidak memperdulikan"
    ]

    for k in keywords:
        if k in text:
            alasan.append(k)

    alasan = "; ".join(alasan)

    # ======================
    # RINGKASAN FAKTA
    # ambil 1000 karakter awal
    # ======================
    ringkasan = text[:1000]

    # ======================
    # CASE ID
    # ======================
    case_id = filename.replace(".txt", "")

    data.append({
        "case_id": case_id,
        "no_perkara": no_perkara,
        "jenis_perkara": jenis,
        "alasan_perceraian": alasan,
        "putusan": putusan,
        "amar_putusan": amar,
        "ringkasan_fakta": ringkasan,
        "text_full": text
    })

df = pd.DataFrame(data)

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/cases.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"Berhasil ekstrak {len(df)} kasus")
print(df.head())
# Sistem Case-Based Reasoning: Analisis Putusan Perceraian
## Pengadilan Agama Kabupaten Malang

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Proyek Akhir SubCPMK-3 — Penalaran Komputer**  
> Universitas Muhammadiyah Malang (UMM) · 2026

---

## Daftar Isi

1. [Deskripsi Proyek](#1-deskripsi-proyek)
2. [Arsitektur Sistem CBR](#2-arsitektur-sistem-cbr)
3. [Hirarki Direktori](#3-hirarki-direktori)
4. [Prasyarat & Instalasi](#4-prasyarat--instalasi)
5. [Panduan Eksekusi Pipeline](#5-panduan-eksekusi-pipeline)
6. [Ringkasan Performa](#6-ringkasan-performa)
7. [Karakteristik Dataset](#7-karakteristik-dataset)
8. [Keterbatasan & Pengembangan Lanjutan](#8-keterbatasan--pengembangan-lanjutan)
9. [Kontribusi](#9-kontribusi)

---

## 1. Deskripsi Proyek

Sistem ini mengimplementasikan siklus penuh **Case-Based Reasoning (CBR)** empat-fase Aamodt & Plaza (1994) — *Retrieve → Reuse → Revise → Retain* — untuk mendukung analisis prediktif putusan perkara perceraian di Pengadilan Agama Kabupaten Malang.

**Tujuan utama sistem:**
- Meretrieve kasus putusan historis yang paling relevan secara hukum terhadap suatu query deskripsi perkara baru.
- Memprediksi amar putusan (*outcome*) berbasis analogi dengan kasus-kasus terdahulu yang serupa.
- Menyediakan infrastruktur evaluasi kuantitatif berbasis metrik *Information Retrieval* dan *Classification*.

**Cakupan domain hukum:**
- **Cerai Gugat**: Permohonan cerai yang diajukan oleh pihak istri (Penggugat).
- **Cerai Talak**: Permohonan ikrar talak yang diajukan oleh pihak suami (Pemohon).

**Teknologi inti:**
- Representasi dokumen: **TF-IDF** (N-Gram: 1–2, `max_features=5000`)
- Metrik kemiripan: **Cosine Similarity**
- Prediksi solusi: **Majority Vote** (fasa Reuse)
- Sumber data: 56 dokumen Putusan Pengadilan Agama Kabupaten Malang (diekstrak dari PDF via OCR)

---

## 2. Arsitektur Sistem CBR

```
Query Baru (Deskripsi Perkara)
         │
         ▼
┌─────────────────────┐
│  [1] RETRIEVE       │  TF-IDF Vectorization + Cosine Similarity
│  Temukan Top-5 Kasus│  → Kembalikan kasus dengan skor tertinggi
│  Paling Mirip       │
└────────┬────────────┘
         │ Top-5 Kasus + Skor Kemiripan
         ▼
┌─────────────────────┐
│  [2] REUSE          │  Majority Vote atas label solusi
│  Adaptasi Solusi    │  → Prediksi: Dikabulkan / Ditolak / N.O.
│  dari Kasus Lama    │
└────────┬────────────┘
         │ Solusi Kandidat
         ▼
┌─────────────────────┐
│  [3] REVISE         │  Validasi manual / domain expert
│  Penyesuaian Solusi │  (Opsional dalam versi otomatis)
└────────┬────────────┘
         │ Solusi Tervalidasi
         ▼
┌─────────────────────┐
│  [4] RETAIN         │  Simpan kasus baru ke case base
│  Pembaruan Case Base│  untuk memperkaya pengetahuan sistem
└─────────────────────┘
```

---

## 3. Hirarki Direktori

```
cbr-perceraian-pa-malang/
│
├── notebooks/                          # Berkas eksplorasi & eksperimen interaktif
│   ├── 000.ipynb                       # Notebook eksplorasi awal & EDA dataset
│   └── src.ipynb                       # Notebook prototipe pipeline CBR terintegrasi
│
├── src/                                # Skrip operasional pipeline produksi
│   ├── 01_extract_text.py              # Ekstraksi teks dari PDF via OCR (pdfplumber/pytesseract)
│   ├── 02_preprocessing.py             # Prapemrosesan: cleaning, tokenisasi, stopword removal
│   ├── 03_feature_extraction.py        # Konstruksi matriks TF-IDF & serialisasi vectorizer
│   ├── 04_retrieval.py                 # Komputasi Cosine Similarity & retrieval Top-K
│   └── 05_evaluation.py               # Kalkulasi metrik retrieval & klasifikasi
│
├── data/
│   ├── raw/                            # Dokumen PDF putusan mentah (56 berkas)
│   │   └── *.pdf
│   ├── processed/                      # Teks hasil ekstraksi OCR per dokumen
│   │   └── *.txt
│   ├── results/                        # Output pipeline retrieval
│   │   └── predictions.csv             # Hasil Top-5 retrieval & prediksi solusi
│   └── eval/                           # Berkas anotasi & hasil evaluasi
│       ├── queries.json                # Query uji + ground truth + label aktual
│       ├── retrieval_metrics.csv       # Metrik Accuracy/Precision/Recall/F1 @K per query
│       └── prediction_metrics.csv      # Metrik klasifikasi agregat (weighted)
│
├── models/                             # Artefak model terserialisasi
│   ├── tfidf_vectorizer.pkl            # Objek TfidfVectorizer yang telah di-fit
│   └── tfidf_matrix.pkl                # Matriks TF-IDF case base (44 × 5000)
│
├── requirements.txt                    # Daftar dependensi Python
├── laporan_analisis_kegagalan.md       # Laporan analisis skenario kegagalan sistem
└── README.md                           # Dokumentasi teknis proyek ini
```

---

## 4. Prasyarat & Instalasi

### 4.1 Prasyarat Sistem

| Komponen | Versi Minimum |
|---|---|
| Python | 3.10 |
| pip | 23.0 |
| RAM | 4 GB (8 GB direkomendasikan) |
| OS | Linux / macOS / Windows (WSL2) |

### 4.2 Kloning Repositori

```bash
git clone https://github.com/<username>/cbr-perceraian-pa-malang.git
cd cbr-perceraian-pa-malang
```

### 4.3 Pembangunan Virtual Environment

Sangat disarankan untuk mengisolasi dependensi proyek dalam lingkungan virtual:

```bash
# Buat virtual environment
python -m venv .venv

# Aktivasi — Linux/macOS
source .venv/bin/activate

# Aktivasi — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 4.4 Instalasi Dependensi

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Isi `requirements.txt`:**

```
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
pdfplumber>=0.11.0
pytesseract>=0.3.10
Pillow>=10.0.0
sastrawi>=1.0.1
nltk>=3.8.1
tqdm>=4.66.0
```

### 4.5 Unduhan Resource NLTK

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 5. Panduan Eksekusi Pipeline

Jalankan seluruh alur sistem secara **sekuensial** menggunakan perintah berikut dari direktori *root* proyek. Pastikan virtual environment telah diaktifkan sebelum memulai.

### Langkah 1: Ekstraksi Teks dari PDF

```bash
python src/01_extract_text.py
```

Membaca seluruh berkas PDF dari `data/raw/` dan menyimpan teks hasil OCR ke `data/processed/`.

---

### Langkah 2: Prapemrosesan Teks

```bash
python src/02_preprocessing.py
```

Menjalankan pipeline pembersihan teks: normalisasi Unicode, penghapusan noise OCR, tokenisasi, penghapusan stopword (Sastrawi + custom list), dan stemming morfologi.

---

### Langkah 3: Ekstraksi Fitur TF-IDF

```bash
python src/03_feature_extraction.py
```

Membangun matriks TF-IDF berukuran *(44 dokumen × 5000 fitur)* dari data train, lalu menyerialisasi objek `TfidfVectorizer` ke `models/tfidf_vectorizer.pkl` dan matriks ke `models/tfidf_matrix.pkl`.

---

### Langkah 4: Retrieval & Prediksi

```bash
python src/04_retrieval.py
```

Membaca 10 query uji dari `data/eval/queries.json`, menghitung Cosine Similarity antara setiap query terhadap matriks case base, menyimpan Top-5 hasil retrieval dan prediksi solusi ke `data/results/predictions.csv`.

---

### Langkah 5: Kalkulasi Metrik Evaluasi

```bash
python src/05_evaluation.py
```

Menghitung seluruh metrik performa sistem dan menyimpan hasilnya:
- `data/eval/retrieval_metrics.csv` — Metrik retrieval per query
- `data/eval/prediction_metrics.csv` — Metrik klasifikasi agregat

---

### Eksekusi Pipeline Penuh (Satu Perintah)

```bash
for script in src/01_extract_text.py src/02_preprocessing.py \
              src/03_feature_extraction.py src/04_retrieval.py \
              src/05_evaluation.py; do
    echo "==> Menjalankan: $script"
    python "$script" || { echo "[ERROR] Pipeline gagal pada: $script"; exit 1; }
done
echo "==> Pipeline selesai. Cek direktori data/eval/ untuk hasil evaluasi."
```

---

## 6. Ringkasan Performa

Tabel di bawah dihasilkan secara otomatis dari eksekusi `src/05_evaluation.py`.  
Jalankan pipeline terlebih dahulu untuk mengisi nilai metrik.

### 6.1 Metrik Retrieval Top-5 per Query

| Query ID | Query (Ringkasan) | Accuracy@5 | Precision@5 | Recall@5 | F1@5 |
|---|---|---|---|---|---|
| 1 | nafkah tidak diberikan | — | — | — | — |
| 2 | perselisihan terus-menerus | — | — | — | — |
| 3 | suami meninggalkan rumah | — | — | — | — |
| 4 | rumah tangga tidak harmonis | — | — | — | — |
| 5 | hubungan dengan perempuan lain | — | — | — | — |
| 6 | pisah tempat tinggal | — | — | — | — |
| 7 | kekerasan dalam rumah tangga | — | — | — | — |
| 8 | tidak bertanggung jawab | — | — | — | — |
| 9 | cerai talak konflik keluarga | — | — | — | — |
| 10 | gugatan cerai nafkah kurang | — | — | — | — |
| **Rata-Rata** | | **—** | **—** | **—** | **—** |

### 6.2 Metrik Klasifikasi Prediksi Solusi (Agregat Berbobot)

| Metrik | Nilai |
|---|---|
| Akurasi | — |
| Presisi (Weighted) | — |
| Recall (Weighted) | — |
| F1-Score (Weighted) | — |
| Jumlah Sampel Uji | 10 |
| Mode Evaluasi | Authentic / Self-Consistency |

> **Catatan**: Isi tabel di atas dengan nilai aktual dari output terminal `05_evaluation.py` setelah pipeline dieksekusi. Nilai `—` adalah *placeholder* yang harus diganti sebelum pengumpulan laporan.

---

## 7. Karakteristik Dataset

| Atribut | Nilai |
|---|---|
| Total Dokumen | 56 putusan PDF |
| Periode Putusan | 2011 – 2026 |
| Sumber | Pengadilan Agama Kabupaten Malang |
| Jenis Perkara | Cerai Gugat & Cerai Talak |
| Pembagian Train:Test | 80:20 (44 train : 12 test) |
| Konfigurasi TF-IDF | N-Gram (1,2), max_features=5000 |
| Metrik Kemiripan | Cosine Similarity |
| Distribusi Kelas | Sangat timpang (mayoritas: Dikabulkan) |

---

## 8. Keterbatasan & Pengembangan Lanjutan

### Keterbatasan Saat Ini

1. **Vocabulary Mismatch**: Representasi TF-IDF tidak mampu menjembatani kesenjangan leksikal antara bahasa awam dengan diksi hukum formal.
2. **Sensitivitas Noise OCR**: Token artefak OCR mendistorsi bobot IDF dan menghasilkan representasi vektor yang tidak representatif.
3. **Majority Vote Bias**: Ketimpangan kelas yang ekstrem menyebabkan prediksi solusi bias ke label dominan "Dikabulkan".

### Arah Pengembangan

| Prioritas | Solusi | Kompleksitas |
|---|---|---|
| Segera | Weighted Similarity Vote | Rendah |
| Segera | Custom stopword list untuk noise OCR | Rendah |
| Menengah | Ekspansi query via sinonim hukum | Menengah |
| Jangka Panjang | Dense Embedding (`indobenchmark/indobert-base-p1`) | Tinggi |
| Jangka Panjang | Fine-tuning LegalBERT pada korpus hukum Indonesia | Sangat Tinggi |

---

## 👥 9. Kontribusi Tim

Proyek ini disusun oleh tim yang terdiri dari 2 mahasiswa dengan pembagian tanggung jawab operasional dan teknis sebagai berikut untuk memenuhi standar penilaian tertinggi SubCPMK-3:

| Nama Anggota | NIM | Fokus Kontribusi & Tanggung Jawab Teknis |
| :--- | :---: | :--- |
| **Afllah Abdi Pratomo** | 202310370311186 | **Core Backend & Pipeline Engineering**:<br>• Implementasi fasa *Data Acquisition* & *Preprocessing* (PDF-to-Txt) (CBR-01, CBR-02).<br>• Ekstraksi fitur statistik ruang vektor (TF-IDF Vectorizer & Cosine Similarity) (CBR-04).<br>• Pengembangan modul *Case Retrieval* (`retrieval.py`) dan pengujian fungsi kedekatan kasus terdekat (CBR-06). |
| **Ahmad Nizar Rusdiwan** | 202310370311186 | **Data Engineering, Evaluation & Documentation**:<br>• Implementasi fasa *Case Representation* (`cases.csv`) dan anotasi queries uji (`queries.json`) (CBR-03, CBR-05).<br>• Pengembangan logika inferensi *Case Solution Reuse* (`predict.py`) menggunakan voting terbobot (CBR-07).<br>• Pembuatan skrip kalkulasi metrik evaluasi (`05_evaluation.py`) dan penyusunan dokumen laporan kritis (CBR-08, CBR-09). |

*Seluruh anggota tim bertanggung jawab bersama dalam penyusunan struktur repositori GitHub yang dapat direplikasi dan finalisasi berkas dokumentasi utama (README.md).*

---


*Proyek ini dikembangkan untuk memenuhi persyaratan SubCPMK-3 Mata Kuliah Penalaran Komputer, Program Studi Teknik Informatika, Universitas Muhammadiyah Malang.*

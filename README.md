# Sistem Case-Based Reasoning: Analisis Putusan Perceraian
## Pengadilan Agama Kabupaten Malang

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org/)

> **Proyek Akhir SubCPMK-3 — Penalaran Komputer**
> Universitas Muhammadiyah Malang (UMM) · 2026

# 👥 Identitas Tim

| Keterangan | Informasi |
|------------|-----------|
| **Mata Kuliah** | Penalaran Komputer |
| **Kelas** | Penalaran Komputer C |
| **Program Studi** | Teknik Informatika |
| **Universitas** | Universitas Muhammadiyah Malang |

## Anggota Tim

| No | Nama | NIM |
|:--:|------|-----|
| 1 | **Afllah Abdi Pratomo** | 202310370311186 |
| 2 | **Ahmad Nizar Rusdiawan** | 202310370311XXX |

## Deskripsi Singkat

Repositori ini berisi implementasi sistem **Case-Based Reasoning (CBR)** untuk analisis dan prediksi putusan perkara perceraian di Pengadilan Agama Kabupaten Malang. Sistem dikembangkan sebagai proyek akhir SubCPMK-3 Mata Kuliah Penalaran Komputer dengan menerapkan tahapan **Retrieve, Reuse, Revise,** dan **Retain**, serta dilengkapi evaluasi menggunakan metrik *Information Retrieval* dan *Classification*.

---

## Daftar Isi

1. [Deskripsi Proyek](#1-deskripsi-proyek)
2. [Arsitektur Sistem CBR](#2-arsitektur-sistem-cbr)
3. [Struktur Direktori Proyek](#3-struktur-direktori-proyek)
4. [Prasyarat & Instalasi](#4-prasyarat--instalasi)
5. [Panduan Eksekusi Pipeline](#5-panduan-eksekusi-pipeline)
6. [Output Proyek](#6-output-proyek)
7. [Karakteristik Dataset](#7-karakteristik-dataset)
8. [Evaluasi & Ringkasan Performa](#8-evaluasi--ringkasan-performa)
9. [Keterbatasan & Pengembangan Lanjutan](#9-keterbatasan--pengembangan-lanjutan)
10. [Kontribusi Tim](#10-kontribusi-tim)

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
- Prediksi solusi: **Majority/Weighted Vote** (fasa Reuse)
- Sumber data: 56 dokumen Putusan Pengadilan Agama Kabupaten Malang (diekstrak dari PDF via OCR)

---

## 2. Arsitektur Sistem CBR

```
Query Baru (Deskripsi Perkara)
         │
         ▼
┌─────────────────────┐
│  [1] RETRIEVE       │  TF-IDF Vectorization (tfidf_vectorizer.py)
│  Temukan Top-K Kasus│  + Cosine Similarity (retrieval.py)
│  Paling Mirip       │  → Kembalikan kasus dengan skor tertinggi
└────────┬────────────┘
         │ Top-K Kasus + Skor Kemiripan
         ▼
┌─────────────────────┐
│  [2] REUSE          │  predict.py — Voting atas label solusi
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
│  [4] RETAIN         │  Penyimpanan kasus baru ke case base
│  Pembaruan Case Base│  untuk memperkaya pengetahuan sistem
└─────────────────────┘
```

Evaluasi kuantitatif terhadap fase Retrieve dan Reuse dilakukan melalui `src/evaluation_colab.ipynb`.

---

## 3. Struktur Direktori Proyek

```
cbr-putusan-perceraian/
│
├── data/
│   ├── Dataset Putusan Perceraian.rar
│   │
│   ├── pdf/                            # 56 berkas PDF putusan asli
│   │   └── putusan_*.pdf
│   │
│   ├── raw/                             # 56 berkas teks hasil OCR mentah
│   │   └── putusan_*.txt
│   │
│   ├── cleaned/                         # 56 berkas teks hasil pembersihan
│   │   └── putusan_*.txt
│   │
│   ├── processed/
│   │   └── cases.csv                    # Case base terstruktur (konsolidasi seluruh dokumen)
│   │
│   ├── split/
│   │   ├── train.csv                    # Subset data latih (case base retrieval)
│   │   └── test.csv                      # Subset data uji (query evaluasi)
│   │
│   ├── results/
│   │   └── predictions.csv               # Hasil Top-K retrieval & prediksi solusi
│   │
│   └── eval/
│       ├── queries.json                  # Daftar query uji & metadata terkait
│       ├── ground_truth.json             # Anotasi ground truth (kasus relevan) per query
│       ├── retrieval_metrics.csv         # Metrik Accuracy/Precision/Recall/F1 @K per query
│       ├── retrieval_summary.csv         # Rata-rata metrik retrieval
│       ├── prediction_metrics.csv        # Metrik klasifikasi prediksi solusi (weighted)
│       ├── error_analysis.csv            # Query dengan F1@K terendah & kemungkinan kegagalan
│       ├── evaluation_report.md          # Laporan evaluasi otomatis (Markdown)
│       └── figures/
│           ├── retrieval_performance.png
│           ├── prediction_performance.png
│           └── confusion_matrix.png
│
├── models/
│   ├── tfidf_vectorizer.pkl              # Objek TfidfVectorizer yang telah di-fit
│   ├── tfidf_matrix.pkl                  # Matriks TF-IDF seluruh case base
│   ├── tfidf_matrix_train.pkl            # Matriks TF-IDF subset data latih
│   └── tfidf_matrix_test.pkl             # Matriks TF-IDF subset data uji
│
├── src/
│   ├── 000.ipynb                         # Notebook eksplorasi awal & EDA dataset
│   ├── src.ipynb                         # Notebook prototipe pipeline CBR terintegrasi
│   ├── extract_metadata.py               # Ekstraksi metadata putusan dari PDF
│   ├── pdf_to_text.py                    # Ekstraksi teks dari PDF via OCR
│   ├── clean_text.py                     # Pembersihan & normalisasi teks
│   ├── tfidf_vectorizer.py               # Konstruksi matriks TF-IDF & serialisasi vectorizer
│   ├── retrieval.py                      # Komputasi Cosine Similarity & retrieval Top-K
│   ├── predict.py                        # Logika inferensi prediksi solusi (fase Reuse)
│   ├── check_tfid.py                     # Utilitas pemeriksaan/validasi matriks TF-IDF
│   ├── check_result.py                   # Utilitas pemeriksaan hasil retrieval/prediksi
│   └── evaluation_colab.ipynb            # Notebook evaluasi Tahap 5 (retrieval, prediksi, visualisasi)
│
├── laporan.md                            # Laporan analisis skenario & temuan sistem
└── README.md                             # Dokumentasi teknis proyek ini
```

### Penjelasan Folder

| Folder | Fungsi |
|---|---|
| `data/pdf/` | Menyimpan berkas PDF putusan asli sebagai sumber ekstraksi. |
| `data/raw/` | Menyimpan teks mentah hasil OCR sebelum dibersihkan. |
| `data/cleaned/` | Menyimpan teks yang telah melalui proses cleaning dan normalisasi. |
| `data/processed/` | Menyimpan case base terstruktur (`cases.csv`) hasil konsolidasi seluruh dokumen. |
| `data/split/` | Menyimpan pembagian data latih (`train.csv`) dan data uji (`test.csv`). |
| `data/results/` | Menyimpan output retrieval dan prediksi solusi (`predictions.csv`). |
| `data/eval/` | Menyimpan seluruh anotasi evaluasi dan hasil metrik, laporan, serta visualisasi. |
| `models/` | Menyimpan artefak `TfidfVectorizer` dan matriks TF-IDF yang telah diserialisasi. |
| `src/` | Menyimpan seluruh skrip pipeline produksi, utilitas debugging, dan notebook. |

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
git clone https://github.com/<username>/cbr-putusan-perceraian.git
cd cbr-putusan-perceraian
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

Instal seluruh pustaka yang dibutuhkan pipeline secara langsung melalui `pip`:

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib \
            pdfplumber pytesseract Pillow \
            sastrawi nltk tqdm jupyter
```

### 4.5 Unduhan Resource NLTK

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 5. Panduan Eksekusi Pipeline

Jalankan seluruh alur sistem secara **sekuensial** menggunakan perintah berikut dari direktori *root* proyek. Pastikan virtual environment telah diaktifkan sebelum memulai.

### Langkah 1: Ekstraksi Metadata Putusan

```bash
python src/extract_metadata.py
```

Mengekstraksi metadata struktural putusan (nomor perkara, jenis perkara, tanggal, dan atribut lain) dari berkas PDF pada `data/pdf/`.

---

### Langkah 2: Ekstraksi Teks dari PDF (OCR)

```bash
python src/pdf_to_text.py
```

Membaca seluruh berkas PDF dari `data/pdf/` dan menyimpan teks hasil OCR mentah ke `data/raw/`.

---

### Langkah 3: Pembersihan Teks

```bash
python src/clean_text.py
```

Menjalankan pipeline pembersihan teks: normalisasi Unicode, penghapusan noise OCR, tokenisasi, penghapusan stopword (Sastrawi + custom list), dan stemming morfologi. Hasil disimpan ke `data/cleaned/`, kemudian dikonsolidasikan menjadi case base terstruktur pada `data/processed/cases.csv`.

---

### Langkah 4: Ekstraksi Fitur TF-IDF

```bash
python src/tfidf_vectorizer.py
```

Membangun matriks TF-IDF dari `data/split/train.csv`, lalu menyerialisasi objek `TfidfVectorizer` dan matriks ke `models/` (`tfidf_vectorizer.pkl`, `tfidf_matrix.pkl`, `tfidf_matrix_train.pkl`, `tfidf_matrix_test.pkl`).

---

### Langkah 5: Retrieval & Prediksi Solusi

```bash
python src/retrieval.py
```

Membaca query uji dari `data/split/test.csv` / `data/eval/queries.json`, menghitung Cosine Similarity antara setiap query terhadap matriks case base, mengambil Top-K kasus paling mirip, lalu memanggil `predict.py` untuk menghasilkan prediksi solusi (fase Reuse). Hasil disimpan ke `data/results/predictions.csv`.

---

### Langkah 6: Evaluasi Retrieval & Prediksi

Evaluasi dilakukan melalui notebook berikut (Google Colab atau Jupyter lokal):

```
src/evaluation_colab.ipynb
```

Pastikan `data/results/predictions.csv`, `data/eval/queries.json`, dan `data/eval/ground_truth.json` telah tersedia, lalu jalankan seluruh sel secara berurutan. Notebook ini menghitung metrik retrieval (Accuracy/Precision/Recall/F1 @K) dan metrik klasifikasi prediksi solusi menggunakan `sklearn.metrics`, menghasilkan visualisasi performa, confusion matrix, error analysis, serta laporan Markdown otomatis ke `data/eval/`.

> **Catatan**: Evaluasi prediksi solusi tidak menggunakan fallback *self-consistency*. Jika label aktual tidak tersedia pada anotasi evaluasi, notebook akan menandai evaluasi sebagai tidak valid (metrik `NaN`) beserta peringatan, bukan memalsukan skor sempurna.

---

### Eksekusi Pipeline Penuh (Satu Perintah)

```bash
for script in src/extract_metadata.py src/pdf_to_text.py \
              src/clean_text.py src/tfidf_vectorizer.py \
              src/retrieval.py; do
    echo "==> Menjalankan: $script"
    python "$script" || { echo "[ERROR] Pipeline gagal pada: $script"; exit 1; }
done
echo "==> Pipeline skrip selesai. Lanjutkan dengan src/evaluation_colab.ipynb untuk evaluasi."
```

---

## 6. Output Proyek

| Berkas | Deskripsi |
|---|---|
| `data/results/predictions.csv` | Hasil Top-K retrieval dan prediksi solusi untuk setiap query uji. |
| `data/eval/queries.json` | Daftar query uji beserta metadata terkait. |
| `data/eval/ground_truth.json` | Anotasi ground truth (kasus relevan) untuk setiap query. |
| `data/eval/retrieval_metrics.csv` | Metrik Accuracy/Precision/Recall/F1 @K per query. |
| `data/eval/retrieval_summary.csv` | Rata-rata metrik retrieval. |
| `data/eval/prediction_metrics.csv` | Metrik klasifikasi prediksi solusi (weighted). |
| `data/eval/error_analysis.csv` | Daftar query dengan F1@K terendah dan kemungkinan alasan kegagalan retrieval. |
| `data/eval/evaluation_report.md` | Laporan evaluasi otomatis dalam format Markdown. |
| `data/eval/figures/retrieval_performance.png` | Visualisasi bar chart metrik retrieval. |
| `data/eval/figures/prediction_performance.png` | Visualisasi bar chart metrik prediksi solusi. |
| `data/eval/figures/confusion_matrix.png` | Confusion matrix prediksi solusi. |

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

## 8. Evaluasi & Ringkasan Performa

Tabel di bawah dihasilkan secara otomatis dari eksekusi `src/evaluation_colab.ipynb`. Jalankan pipeline dan notebook evaluasi terlebih dahulu untuk mengisi nilai metrik.

### 8.1 Rata-Rata Metrik Retrieval Top-K per Model

| Model | Accuracy@K | Precision@K | Recall@K | F1@K |
|---|---|---|---|---|
| baseline | — | — | — | — |

> Untuk rincian per query, lihat `data/eval/retrieval_metrics.csv`. Untuk query yang gagal (ground truth tidak muncul di Top-K, atau tidak memiliki anotasi pada `data/eval/ground_truth.json`), lihat `data/eval/error_analysis.csv`.

### 8.2 Metrik Klasifikasi Prediksi Solusi (Agregat Berbobot)

| Model | Akurasi | Presisi (W) | Recall (W) | F1-Score (W) | Jumlah Sampel | Mode Evaluasi |
|---|---|---|---|---|---|---|
| baseline | — | — | — | — | — | — |

**Mode Evaluasi:**
- `Authentic` — label solusi aktual tersedia pada anotasi evaluasi; metrik mencerminkan performa nyata.
- `Invalid - Tanpa Ground Truth` — label solusi aktual tidak tersedia sama sekali; metrik diset `NaN` dan **tidak** dipalsukan menjadi skor sempurna.

> **Catatan**: Isi tabel di atas dengan nilai aktual dari `data/eval/prediction_metrics.csv` setelah notebook evaluasi dieksekusi. Nilai `—` adalah *placeholder* yang harus diganti sebelum pengumpulan laporan. Lihat juga `data/eval/evaluation_report.md` untuk laporan naratif otomatis serta `data/eval/figures/` untuk visualisasi bar chart dan confusion matrix.

---

## 9. Keterbatasan & Pengembangan Lanjutan

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

## 👥 10. Kontribusi Tim

Proyek ini disusun oleh tim yang terdiri dari 2 mahasiswa dengan pembagian tanggung jawab operasional dan teknis sebagai berikut untuk memenuhi standar penilaian tertinggi SubCPMK-3:

| Nama Anggota | NIM | Fokus Kontribusi & Tanggung Jawab Teknis |
| :--- | :---: | :--- |
| **Afllah Abdi Pratomo** | 202310370311186 | **Core Backend & Pipeline Engineering**:<br>• Implementasi fasa *Data Acquisition* & *Preprocessing* (`pdf_to_text.py`, `clean_text.py`, `extract_metadata.py`).<br>• Ekstraksi fitur statistik ruang vektor (`tfidf_vectorizer.py` — TF-IDF & Cosine Similarity).<br>• Pengembangan modul *Case Retrieval* (`retrieval.py`) dan utilitas validasi (`check_tfid.py`, `check_result.py`). |
| **Ahmad Nizar Rusdiwan** | 202310370311186 | **Data Engineering, Evaluation & Documentation**:<br>• Implementasi fasa *Case Representation* (`cases.csv`) dan anotasi query uji (`queries.json`, `ground_truth.json`).<br>• Pengembangan logika inferensi *Case Solution Reuse* (`predict.py`) menggunakan voting terbobot.<br>• Pembuatan notebook evaluasi (`evaluation_colab.ipynb`) dan penyusunan dokumen laporan kritis (`laporan.md`). |

*Seluruh anggota tim bertanggung jawab bersama dalam penyusunan struktur repositori GitHub yang dapat direplikasi dan finalisasi berkas dokumentasi utama (README.md).*

---

*Proyek ini dikembangkan untuk memenuhi persyaratan SubCPMK-3 Mata Kuliah Penalaran Komputer, Program Studi Teknik Informatika, Universitas Muhammadiyah Malang.*
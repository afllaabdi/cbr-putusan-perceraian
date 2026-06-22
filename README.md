# Progress Proyek CBR Putusan Perceraian

## Informasi Proyek

**Mata Kuliah:** Penalaran Komputer
**Sub CPMK 3:** Case-Based Reasoning (CBR)
**Domain Kasus:** Putusan Perceraian (Pengadilan Agama Kabupaten Malang)
**Jumlah Dataset:** 56 Putusan

---

# Progress Pengerjaan

## ✅ CBR-01: Membangun Case Base

### Status

Selesai

### Hasil

* Mengumpulkan 56 dokumen putusan perceraian dari Direktori Putusan Mahkamah Agung RI.
* Domain perkara: Perceraian (Cerai Gugat dan Cerai Talak).
* Dataset tersimpan dalam format PDF.

### Folder

```text
data/pdf/
```

---

## ✅ CBR-02: Cleaning dan Preprocessing

### Status

Selesai

### Hasil

* Konversi PDF ke TXT.
* Normalisasi teks.
* Pembersihan sebagian noise OCR.
* Dataset siap diproses ke tahap representasi kasus.

### Folder

```text
data/raw/
data/cleaned/
```

---

## ✅ CBR-03: Case Representation

### Status

Selesai

### Hasil

Ekstraksi metadata:

* Case ID
* Nomor Perkara
* Jenis Perkara
* Alasan Perceraian
* Putusan
* Amar Putusan (masih banyak nilai kosong)
* Ringkasan Fakta
* Full Text

### Output

```text
data/processed/cases.csv
```

Jumlah kasus:

```text
56 kasus
```

---

## ✅ CBR-04: Representasi Vektor (TF-IDF)

### Status

Selesai

### Metode

* TF-IDF Vectorizer
* N-Gram (1,2)
* Stopwords Bahasa Indonesia
* Max Features = 5000

### Train-Test Split

Sesuai spesifikasi tugas:

```text
80 : 20
```

Hasil:

```text
Train : 44 dokumen
Test  : 12 dokumen
```

### Output

```text
data/split/train.csv
data/split/test.csv

models/tfidf_vectorizer.pkl
models/tfidf_matrix_train.pkl
models/tfidf_matrix_test.pkl
```

---

## ✅ CBR-05: Query Uji dan Ground Truth

### Status

Selesai

### Hasil

Membuat 10 query uji yang mewakili:

* Kurang nafkah
* Perselisihan
* Pertengkaran
* Meninggalkan rumah
* Pisah tempat tinggal
* Perempuan lain
* KDRT
* Konflik keluarga
* Cerai talak

### File

```text
data/eval/queries.json
```

---

## ✅ CBR-06: Case Retrieval

### Status

Selesai

### Metode

* TF-IDF
* Cosine Similarity
* Top-K Retrieval (K = 5)

### Fungsi

```python
retrieve(query, k=5)
```

### Hasil

Sistem mampu:

* Mengubah query menjadi vektor TF-IDF.
* Menghitung cosine similarity.
* Mengambil 5 kasus paling mirip.

### File

```text
src/retrieval.py
```

---

## ✅ CBR-07: Case Solution Reuse

### Status

Selesai

### Metode

1. Majority Vote
2. Weighted Similarity Vote

### Fungsi

```python
predict_outcome(query)
```

### Hasil

Untuk setiap query:

* Mengambil Top 5 kasus paling mirip.
* Mengambil label putusan kasus lama.
* Melakukan voting.
* Menghasilkan prediksi putusan.

### Output

```text
data/results/predictions.csv
```

Jumlah prediksi:

```text
10 query
```

Kolom:

```text
query_id
query
predicted_solution
top_5_case_ids
```

---

# Tahap Yang Belum Dikerjakan

## ⏳ CBR-08 Evaluasi

Yang perlu dilakukan:

* Accuracy
* Precision
* Recall
* F1 Score

Output:

```text
data/eval/retrieval_metrics.csv
data/eval/prediction_metrics.csv
```

---

## ⏳ CBR-09 Analisis Kegagalan Model

Yang perlu dilakukan:

* Analisis retrieval yang kurang relevan.
* Analisis hasil prediksi.
* Menjelaskan penyebab kesalahan.
* Memberikan rekomendasi perbaikan.

---

## ⏳ CBR-10 Dokumentasi Repository

Yang perlu dilakukan:

* Rapikan struktur repository.
* Lengkapi README utama.
* Tambahkan screenshot hasil.
* Tambahkan cara menjalankan project.

---

# Catatan Penting

1. Sebagian hasil OCR masih mengandung noise seperti:

```text
aadalah
abahwa
aakan
```

Namun retrieval dan prediksi tetap berjalan.

2. Mayoritas label putusan dalam dataset adalah:

```text
Dikabulkan
```

Sehingga sebagian besar hasil prediksi juga menghasilkan label:

```text
Dikabulkan
```

3. Semua file model berada di folder:

```text
models/
```

4. Dataset utama berada di:

```text
data/processed/cases.csv
```

progres...

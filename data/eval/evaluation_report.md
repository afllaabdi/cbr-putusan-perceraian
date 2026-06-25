# Laporan Evaluasi Tahap 5 — Retrieval & Prediksi Solusi

## Ringkasan Tujuan Evaluasi
Mengukur performa retrieval Top-5 dan prediksi solusi sistem CBR menggunakan metrik Accuracy, Precision, Recall, dan F1-score, dibandingkan antar model yang tersedia pada `predictions.csv`.

## Tabel Metrik Retrieval (Rata-Rata per Model)
| model_name | accuracy_at_k | precision_at_k | recall_at_k | f1_at_k |
| --- | --- | --- | --- | --- |
| baseline | 0.22 | 0.22 | 1.0 | 0.3571 |

## Tabel Metrik Prediksi Solusi (Klasifikasi Berbobot)
| model_name | accuracy | precision_weighted | recall_weighted | f1_weighted | support | evaluation_mode |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.0 | 1.0 | 1.0 | 1.0 | 10 | Authentic |

## Ringkasan Error Analysis (Top 5 F1@K Terendah)
| query_id | model_name | query | ground_truth_ids | retrieved_ids | f1_at_k | reason |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | baseline | terjadi perselisihan dan pertengkaran terus menerus | putusan_6071_pdt.g_2022_pa.kab.mlg_20260617083217 | putusan_6071_pdt.g_2022_pa.kab.mlg_20260617083217;putusan_6620_pdt.g_2020_pa.kab.mlg_20260617083642;putusan_3071_pdt.g_2016_pa.kab.mlg_20260617081632;putusan_4885_pdt.g_2019_pa.kab.mlg_20260617083806;putusan_4578_pdt.g_2020_pa.kab.mlg_20260617083825 | 0.3333 | Sebagian kasus relevan ter-retrieve namun precision/recall masih rendah |
| 3 | baseline | suami meninggalkan rumah dan tidak memberi kabar | putusan_5210_pdt.g_2022_pa.kab.mlg_20260617083637 | putusan_5210_pdt.g_2022_pa.kab.mlg_20260617083637;putusan_4885_pdt.g_2019_pa.kab.mlg_20260617083806;putusan_4589_pdt.g_2020_pa.kab.mlg_20260617083702;putusan_4586_pdt.g_2020_pa.kab.mlg_20260617082453;putusan_7092_pdt.g_2019_pa.kab.mlg_20260617083935 | 0.3333 | Sebagian kasus relevan ter-retrieve namun precision/recall masih rendah |
| 4 | baseline | hubungan rumah tangga tidak harmonis | putusan_5280_pdt.g_2023_pa.kab.mlg_20260617084813 | putusan_5280_pdt.g_2023_pa.kab.mlg_20260617084813;putusan_7018_pdt.g_2022_pa.kab.mlg_20260617083434;putusan_3109_pdt.g_2019_pa.kab.mlg_20260617081614;putusan_6771_pdt.g_2022_pa.kab.mlg_20260617084833;putusan_6071_pdt.g_2022_pa.kab.mlg_20260617083217 | 0.3333 | Sebagian kasus relevan ter-retrieve namun precision/recall masih rendah |
| 5 | baseline | suami memiliki hubungan dengan perempuan lain | putusan_1038_pdt.g_2017_pa.kab.mlg_20260617082811 | putusan_1038_pdt.g_2017_pa.kab.mlg_20260617082811;putusan_7220_pdt.g_2019_pa.kab.mlg_20260617084231;putusan_4244_pdt.g_2016_pa.kab.mlg._20260617085200;putusan_1832_pdt.g_2018_pa.kab.mlg_20260614185407;putusan_1247_pdt.g_2026_pa.kab.mlg_20260617080613 | 0.3333 | Sebagian kasus relevan ter-retrieve namun precision/recall masih rendah |
| 7 | baseline | terjadi kekerasan dalam rumah tangga | putusan_5657_pdt.g_2020_pa.kab.mlg_20260617082330 | putusan_5657_pdt.g_2020_pa.kab.mlg_20260617082330;putusan_4885_pdt.g_2019_pa.kab.mlg_20260617083806;putusan_4578_pdt.g_2020_pa.kab.mlg_20260617083825;putusan_3071_pdt.g_2016_pa.kab.mlg_20260617081632;putusan_4244_pdt.g_2016_pa.kab.mlg._20260617085200 | 0.3333 | Sebagian kasus relevan ter-retrieve namun precision/recall masih rendah |

## Kesimpulan
- Model retrieval dengan performa terbaik: **baseline** (F1@5 = 0.3571).
- Seluruh model memiliki label aktual yang valid untuk evaluasi prediksi.
- Kegagalan retrieval pada beberapa query mengindikasikan potensi *vocabulary mismatch* antara deskripsi query dan diksi hukum formal pada dokumen kasus.

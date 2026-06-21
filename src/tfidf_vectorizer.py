import pandas as pd
import re
import os
import joblib
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================
# DOWNLOAD STOPWORDS (sekali saja)
# =====================================

nltk.download('stopwords')

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("data/processed/cases.csv")

# =====================================
# STOPWORDS INDONESIA
# =====================================

indo_stopwords = stopwords.words("indonesian")

# Tambahan kata yang sering muncul di putusan
custom_stopwords = [
    "pemohon",
    "termohon",
    "penggugat",
    "tergugat",
    "putusan",
    "nomor",
    "halaman",
    "agama",
    "kabupaten",
    "malang",
    "mahkamah",
    "agung",
    "republik",
    "indonesia",
    "kepaniteraan",
    "disclaimer",
    "sidang",
    "hakim",
    "perkara",
    "pasal",
    "huruf",
    "ayat",
    "rupiah",
    "rp",
]

all_stopwords = indo_stopwords + custom_stopwords

# =====================================
# PREPROCESSING
# =====================================

def preprocess_text(text):

    text = str(text).lower()

    # hapus url
    text = re.sub(r'http\S+', ' ', text)

    # hapus angka
    text = re.sub(r'\d+', ' ', text)

    # hapus karakter selain huruf
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # hapus huruf tunggal
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)

    # hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# =====================================
# PILIH TEKS
# =====================================

texts = df["text_full"].fillna("").apply(preprocess_text)

# =====================================
# TF-IDF
# =====================================

vectorizer = TfidfVectorizer(
    stop_words=all_stopwords,
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90
)

tfidf_matrix = vectorizer.fit_transform(texts)

# =====================================
# INFO
# =====================================

print("=" * 50)
print("HASIL TF-IDF")
print("=" * 50)

print("Jumlah Dokumen :", tfidf_matrix.shape[0])
print("Jumlah Fitur   :", tfidf_matrix.shape[1])

# =====================================
# SIMPAN
# =====================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

joblib.dump(
    tfidf_matrix,
    "models/tfidf_matrix.pkl"
)

print("\nTF-IDF berhasil disimpan")
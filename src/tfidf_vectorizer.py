import pandas as pd
import numpy as np
import re
import os
import joblib
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# =====================================
# DOWNLOAD STOPWORDS
# =====================================

nltk.download('stopwords')

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("data/processed/cases.csv")

print("=" * 50)
print("DATASET")
print("=" * 50)
print(f"Jumlah Data Total : {len(df)}")

# =====================================
# TRAIN TEST SPLIT
# =====================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"Jumlah Data Train : {len(train_df)}")
print(f"Jumlah Data Test  : {len(test_df)}")

# =====================================
# BUAT FOLDER
# =====================================

os.makedirs("data/split", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =====================================
# SIMPAN TRAIN TEST
# =====================================

train_df.to_csv(
    "data/split/train.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    "data/split/test.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================
# STOPWORDS
# =====================================

indo_stopwords = stopwords.words("indonesian")

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
    "rp"
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
# PREPROCESS TRAIN DAN TEST
# =====================================

texts_train = (
    train_df["text_full"]
    .fillna("")
    .apply(preprocess_text)
)

texts_test = (
    test_df["text_full"]
    .fillna("")
    .apply(preprocess_text)
)

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

tfidf_matrix_train = vectorizer.fit_transform(texts_train)

tfidf_matrix_test = vectorizer.transform(texts_test)

# =====================================
# INFO HASIL
# =====================================

print("\n" + "=" * 50)
print("HASIL TF-IDF")
print("=" * 50)

print("Shape Train :", tfidf_matrix_train.shape)
print("Shape Test  :", tfidf_matrix_test.shape)

# =====================================
# SIMPAN MODEL
# =====================================

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

joblib.dump(
    tfidf_matrix_train,
    "models/tfidf_matrix_train.pkl"
)

joblib.dump(
    tfidf_matrix_test,
    "models/tfidf_matrix_test.pkl"
)

print("\nModel berhasil disimpan")

# =====================================
# CEK FITUR
# =====================================

features = vectorizer.get_feature_names_out()

print("\nJumlah fitur:", len(features))

print("\n20 fitur pertama:")

for feature in features[:20]:
    print(feature)

print("\nSelesai.")

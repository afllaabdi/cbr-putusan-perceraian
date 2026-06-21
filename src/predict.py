import pandas as pd
import joblib
import os

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# LOAD DATA TRAIN
# =====================================

df = pd.read_csv("data/split/train.csv")

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    "models/tfidf_matrix_train.pkl"
)

# =====================================
# RETRIEVAL
# =====================================

def retrieve(query, k=5):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarities.argsort()[::-1][:k]

    results = []

    for idx in top_indices:

        results.append({
            "case_id": df.iloc[idx]["case_id"],
            "similarity": float(similarities[idx]),
            "putusan": str(df.iloc[idx]["putusan"])
        })

    return results


# =====================================
# MAJORITY VOTE
# =====================================

def majority_vote(results):

    putusan_list = [
        r["putusan"]
        for r in results
    ]

    voting = Counter(putusan_list)

    return voting.most_common(1)[0][0]


# =====================================
# WEIGHTED VOTE
# =====================================

def weighted_vote(results):

    scores = {}

    for r in results:

        label = r["putusan"]

        if label not in scores:
            scores[label] = 0

        scores[label] += r["similarity"]

    return max(scores, key=scores.get)


# =====================================
# PREDICT OUTCOME
# =====================================

def predict_outcome(query):

    top_k = retrieve(query, k=5)

    return {
        "query": query,
        "top_k": top_k,
        "majority_vote": majority_vote(top_k),
        "weighted_vote": weighted_vote(top_k)
    }


# =====================================
# 10 QUERY UJI
# =====================================

queries = [
    "suami tidak memberikan nafkah kepada istri",
    "terjadi perselisihan dan pertengkaran terus menerus",
    "suami meninggalkan rumah dan tidak memberi kabar",
    "hubungan rumah tangga tidak harmonis",
    "suami memiliki hubungan dengan perempuan lain",
    "pasangan telah pisah tempat tinggal",
    "terjadi kekerasan dalam rumah tangga",
    "suami tidak bertanggung jawab terhadap keluarga",
    "permohonan cerai talak karena konflik keluarga",
    "istri mengajukan gugatan cerai karena nafkah kurang"
]

# =====================================
# SIMPAN HASIL
# =====================================

all_predictions = []

for i, query in enumerate(queries, start=1):

    result = predict_outcome(query)

    print("=" * 70)
    print(f"QUERY {i}")
    print(query)

    print("\nTOP 5 KASUS:")

    for case in result["top_k"]:

        print(
            f"{case['case_id']} | "
            f"{case['similarity']:.4f} | "
            f"{case['putusan']}"
        )

    print(
        "\nPrediksi:",
        result["weighted_vote"]
    )

    all_predictions.append({
        "query_id": i,
        "query": query,
        "predicted_solution": result["weighted_vote"],
        "top_5_case_ids": ";".join(
            [x["case_id"] for x in result["top_k"]]
        )
    })

# =====================================
# EXPORT CSV
# =====================================

os.makedirs(
    "data/results",
    exist_ok=True
)

pred_df = pd.DataFrame(
    all_predictions
)

pred_df.to_csv(
    "data/results/predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===================================")
print("predictions.csv berhasil dibuat")
print("Jumlah prediksi:", len(pred_df))
print("===================================")

import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("data/processed/cases.csv")

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    "models/tfidf_matrix.pkl"
)

# ==========================
# RETRIEVE FUNCTION
# ==========================

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
            "similarity": round(float(similarities[idx]), 4),
            "putusan": df.iloc[idx]["putusan"]
        })

    return results


# ==========================
# TEST
# ==========================

query = "suami tidak memberikan nafkah kepada istri"

results = retrieve(query)

print("\nQUERY:")
print(query)

print("\nTOP 5 KASUS MIRIP:\n")

for i, r in enumerate(results, start=1):

    print(
        f"{i}. {r['case_id']} "
        f"(Similarity={r['similarity']}) "
        f"Putusan={r['putusan']}"
    )
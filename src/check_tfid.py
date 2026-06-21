import joblib

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

features = vectorizer.get_feature_names_out()

print("Jumlah fitur:", len(features))

print("\n20 fitur pertama:\n")

for word in features[:20]:
    print(word)
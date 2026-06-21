import ast
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# KONSTANTA PATH
# ---------------------------------------------------------------------------
PATH_PREDICTIONS_CSV = os.path.join("data", "results", "predictions.csv")
PATH_QUERIES_JSON    = os.path.join("data", "eval", "queries.json")
PATH_OUT_RETRIEVAL   = os.path.join("data", "eval", "retrieval_metrics.csv")
PATH_OUT_PREDICTION  = os.path.join("data", "eval", "prediction_metrics.csv")

K = 5  # Kedalaman retrieval yang dievaluasi


# ---------------------------------------------------------------------------
# FUNGSI UTILITAS PARSING
# ---------------------------------------------------------------------------

def parse_top_k_ids(raw_value: str) -> list[str]:

    if pd.isna(raw_value) or str(raw_value).strip() == "":
        return []

    raw_str = str(raw_value).strip()

    # Upaya 1: Interpretasi sebagai list literal Python (e.g., "['a', 'b']")
    if raw_str.startswith("["):
        try:
            parsed = ast.literal_eval(raw_str)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except (ValueError, SyntaxError):
            pass

    # Upaya 2: Interpretasi sebagai string delimiter titik-koma (e.g., "a;b;c")
    if ";" in raw_str:
        return [item.strip() for item in raw_str.split(";") if item.strip()]

    # Upaya 3: Delimiter koma sebagai fallback
    if "," in raw_str:
        return [item.strip() for item in raw_str.split(",") if item.strip()]

    # Upaya 4: Nilai tunggal non-delimited
    return [raw_str]


def load_predictions(path: str) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Berkas prediksi tidak ditemukan: '{path}'\n"
            f"        Pastikan pipeline retrieval (04_retrieval.py) telah dieksekusi."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig mengatasi BOM karakter

    required_cols = {"query_id", "query", "predicted_solution", "top_5_case_ids"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(
            f"[ERROR] Kolom berikut tidak ditemukan pada CSV: {missing}\n"
            f"        Kolom tersedia: {list(df.columns)}"
        )

    df["query_id"]       = df["query_id"].astype(str).str.strip()
    df["top_5_case_ids"] = df["top_5_case_ids"].apply(parse_top_k_ids)

    return df


def load_queries(path: str) -> dict:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Berkas queries tidak ditemukan: '{path}'\n"
            f"        Pastikan direktori data/eval/ telah dibuat."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    query_map = {}
    for item in data:
        qid = str(item.get("query_id", "")).strip()

        # Mendukung kunci alternatif untuk ground truth
        ground_truth_raw = (
            item.get("ground_truth")
            or item.get("relevant_cases")
            or item.get("relevant_ids")
            or []
        )
        ground_truth = [str(x).strip() for x in ground_truth_raw] if ground_truth_raw else []

        # Mendukung kunci alternatif untuk label solusi aktual
        actual_solution = (
            item.get("actual_solution")
            or item.get("label")
            or item.get("outcome")
            or None
        )

        query_map[qid] = {
            "ground_truth":     ground_truth,
            "actual_solution":  actual_solution,
        }

    return query_map


# ---------------------------------------------------------------------------
# FUNGSI METRIK RETRIEVAL
# ---------------------------------------------------------------------------

def compute_accuracy_at_k(ground_truth: list, retrieved: list) -> float:

    if not ground_truth or not retrieved:
        return 0.0
    intersection = set(ground_truth) & set(retrieved)
    return 1.0 if len(intersection) >= 1 else 0.0


def compute_precision_at_k(ground_truth: list, retrieved: list, k: int = K) -> float:

    if not retrieved:
        return 0.0
    retrieved_k   = retrieved[:k]
    intersection  = set(ground_truth) & set(retrieved_k)
    return len(intersection) / k


def compute_recall_at_k(ground_truth: list, retrieved: list) -> float:

    if not ground_truth or not retrieved:
        return 0.0
    intersection = set(ground_truth) & set(retrieved)
    return len(intersection) / len(ground_truth)


def compute_f1_at_k(precision: float, recall: float) -> float:

    denominator = precision + recall
    if denominator == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / denominator


# ---------------------------------------------------------------------------
# PIPELINE EVALUASI RETRIEVAL
# ---------------------------------------------------------------------------

def evaluate_retrieval(df_pred: pd.DataFrame, query_map: dict) -> pd.DataFrame:

    records = []
    missing_gt_count = 0

    for _, row in df_pred.iterrows():
        qid       = str(row["query_id"]).strip()
        retrieved = row["top_5_case_ids"]

        meta         = query_map.get(qid, {})
        ground_truth = meta.get("ground_truth", [])

        if not ground_truth:
            missing_gt_count += 1

        acc  = compute_accuracy_at_k(ground_truth, retrieved)
        prec = compute_precision_at_k(ground_truth, retrieved, k=K)
        rec  = compute_recall_at_k(ground_truth, retrieved)
        f1   = compute_f1_at_k(prec, rec)

        records.append({
            "query_id":       qid,
            "accuracy_top_k": acc,
            "precision_top_k": prec,
            "recall_top_k":   rec,
            "f1_top_k":       f1,
        })

    if missing_gt_count > 0:
        print(
            f"\n  [PERINGATAN] {missing_gt_count} dari {len(df_pred)} query tidak memiliki "
            f"anotasi ground_truth pada queries.json.\n"
            f"  Metrik retrieval untuk query tersebut akan bernilai 0.0 secara default.\n"
            f"  Tambahkan kolom 'ground_truth' pada queries.json untuk hasil evaluasi penuh."
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# PIPELINE EVALUASI PREDIKSI SOLUSI
# ---------------------------------------------------------------------------

def evaluate_prediction(df_pred: pd.DataFrame, query_map: dict) -> pd.DataFrame:

    y_pred = df_pred["predicted_solution"].fillna("UNKNOWN").astype(str).tolist()

    y_true = []
    for _, row in df_pred.iterrows():
        qid  = str(row["query_id"]).strip()
        meta = query_map.get(qid, {})
        sol  = meta.get("actual_solution")
        # Fallback: jika actual_solution tidak tersedia, gunakan predicted_solution
        # (skenario self-consistency check)
        y_true.append(str(sol) if sol else str(row["predicted_solution"]))

    # Deteksi modus evaluasi
    has_actual = any(
        query_map.get(str(row["query_id"]), {}).get("actual_solution") is not None
        for _, row in df_pred.iterrows()
    )

    if not has_actual:
        print(
            "\n  [PERINGATAN] Kolom 'actual_solution' tidak ditemukan pada queries.json.\n"
            "  Evaluasi klasifikasi menggunakan mode SELF-CONSISTENCY:\n"
            "  predicted_solution dibandingkan dengan predicted_solution itu sendiri.\n"
            "  Semua metrik klasifikasi akan bernilai 1.0 — ini bukan cerminan performa riil.\n"
            "  Tambahkan kolom 'actual_solution' pada queries.json untuk evaluasi otentik."
        )

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Laporan klasifikasi terperinci per kelas
    print("\n  === Laporan Klasifikasi Per Kelas ===")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
            target_names=sorted(set(y_true + y_pred)),
        )
    )

    return pd.DataFrame([{
        "metric":           "Classification (Weighted)",
        "accuracy":         round(acc,  4),
        "precision_weighted": round(prec, 4),
        "recall_weighted":  round(rec,  4),
        "f1_weighted":      round(f1,   4),
        "support":          len(y_true),
        "evaluation_mode":  "Authentic" if has_actual else "Self-Consistency",
    }])


# ---------------------------------------------------------------------------
# FUNGSI OUTPUT TERMINAL
# ---------------------------------------------------------------------------

def print_summary_table(df_retrieval: pd.DataFrame, df_prediction: pd.DataFrame) -> None:
    separator = "=" * 70

    print(f"\n{separator}")
    print("  RINGKASAN EVALUASI SISTEM CBR — PENGADILAN AGAMA KAB. MALANG")
    print(separator)

    # ---- Tabel Retrieval Per Query ----
    print("\n  [1] METRIK RETRIEVAL Top-5 PER QUERY\n")
    header = f"  {'Query ID':<12} {'Acc@5':>8} {'Prec@5':>8} {'Rec@5':>8} {'F1@5':>8}"
    print(header)
    print("  " + "-" * 48)

    for _, row in df_retrieval.iterrows():
        line = (
            f"  {str(row['query_id']):<12}"
            f"  {row['accuracy_top_k']:>6.4f}"
            f"  {row['precision_top_k']:>6.4f}"
            f"  {row['recall_top_k']:>6.4f}"
            f"  {row['f1_top_k']:>6.4f}"
        )
        print(line)

    print("  " + "-" * 48)
    mean_acc  = df_retrieval["accuracy_top_k"].mean()
    mean_prec = df_retrieval["precision_top_k"].mean()
    mean_rec  = df_retrieval["recall_top_k"].mean()
    mean_f1   = df_retrieval["f1_top_k"].mean()

    print(
        f"  {'RATA-RATA':<12}"
        f"  {mean_acc:>6.4f}"
        f"  {mean_prec:>6.4f}"
        f"  {mean_rec:>6.4f}"
        f"  {mean_f1:>6.4f}"
    )

    # ---- Tabel Prediksi Solusi ----
    print(f"\n{separator}")
    print("\n  [2] METRIK PREDIKSI SOLUSI (KLASIFIKASI BERBOBOT)\n")

    for _, row in df_prediction.iterrows():
        print(f"  Akurasi          : {row['accuracy']:.4f}  ({row['accuracy']*100:.2f}%)")
        print(f"  Presisi Berbobot : {row['precision_weighted']:.4f}")
        print(f"  Recall Berbobot  : {row['recall_weighted']:.4f}")
        print(f"  F1 Berbobot      : {row['f1_weighted']:.4f}")
        print(f"  Jumlah Sampel    : {row['support']}")
        print(f"  Mode Evaluasi    : {row['evaluation_mode']}")

    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main() -> None:

    print("\n" + "=" * 70)
    print("  SISTEM EVALUASI CBR — PUTUSAN PERCERAIAN PENGADILAN AGAMA")
    print("  Universitas Muhammadiyah Malang — Penalaran Komputer")
    print("=" * 70)

    # --- Langkah 1: Pemuatan Data ---
    print(f"\n  [LANGKAH 1/4] Memuat berkas predictions dari: {PATH_PREDICTIONS_CSV}")
    try:
        df_pred = load_predictions(PATH_PREDICTIONS_CSV)
    except (FileNotFoundError, KeyError) as e:
        print(str(e))
        sys.exit(1)
    print(f"  Berhasil memuat {len(df_pred)} baris prediksi.")

    print(f"\n  [LANGKAH 2/4] Memuat berkas queries dari: {PATH_QUERIES_JSON}")
    try:
        query_map = load_queries(PATH_QUERIES_JSON)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Gagal memuat queries.json: {e}")
        sys.exit(1)
    print(f"  Berhasil memuat metadata untuk {len(query_map)} query.")

    # --- Langkah 2: Evaluasi Retrieval ---
    print(f"\n  [LANGKAH 3/4] Menghitung metrik retrieval Top-{K}...")
    df_retrieval = evaluate_retrieval(df_pred, query_map)

    # --- Langkah 3: Evaluasi Prediksi ---
    print(f"\n  [LANGKAH 4/4] Menghitung metrik klasifikasi prediksi solusi...")
    df_prediction = evaluate_prediction(df_pred, query_map)

    # --- Langkah 4: Penyimpanan Hasil ---
    os.makedirs(os.path.dirname(PATH_OUT_RETRIEVAL), exist_ok=True)

    df_retrieval.to_csv(PATH_OUT_RETRIEVAL, index=False, encoding="utf-8")
    print(f"\n  [SIMPAN] Metrik retrieval  → {PATH_OUT_RETRIEVAL}")

    df_prediction.to_csv(PATH_OUT_PREDICTION, index=False, encoding="utf-8")
    print(f"  [SIMPAN] Metrik klasifikasi → {PATH_OUT_PREDICTION}")

    # --- Langkah 5: Ringkasan Terminal ---
    print_summary_table(df_retrieval, df_prediction)


if __name__ == "__main__":
    main()

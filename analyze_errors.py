import pandas as pd
import numpy as np
from src.plagiarism_engine.preprocessing import TextPreprocessor
from src.plagiarism_engine.minhash import MinHash
from src.plagiarism_engine.simhash import SimHash
from src.plagiarism_engine.evaluation import EngineEvaluator

def analyze_errors(
    pairs_file="data/raw/quora/questions.csv",
    text_col_a="question1",
    text_col_b="question2",
    label_col="is_duplicate",
    limit=5000,
    shingle_size=3,
    num_hashes=128,
    threshold=0.4,
    output_csv="outputs/error_analysis.csv"
):

    print("Loading Quora dataset...")
    df = pd.read_csv(pairs_file)
    if limit and limit < len(df):
        df = df.head(limit)
    print(f"Loaded {len(df)} pairs.")


    prep = TextPreprocessor(shingle_size=shingle_size)
    simhash_engine = SimHash(bits=64)
    evaluator = EngineEvaluator(threshold=threshold, num_hashes=num_hashes)

    print("Computing similarities...")
    results = []

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"   Progress: {idx}/{len(df)}")

        text_a = str(row[text_col_a])
        text_b = str(row[text_col_b])
        true_label = int(row[label_col])

        tokens_a = prep.tokenize(prep.clean_text(text_a))
        tokens_b = prep.tokenize(prep.clean_text(text_b))
        tokens_a = prep.remove_stopwords(tokens_a)
        tokens_b = prep.remove_stopwords(tokens_b)

        weights_a = TextPreprocessor.compute_tf(tokens_a)
        weights_b = TextPreprocessor.compute_tf(tokens_b)
        if weights_a and weights_b:
            fp_a = simhash_engine.compute_fingerprint(weights_a)
            fp_b = simhash_engine.compute_fingerprint(weights_b)
            simhash_sim = simhash_engine.similarity(fp_a, fp_b)
        else:
            simhash_sim = 0.0
        simhash_pred = 1 if simhash_sim >= threshold else 0

        shingles_a = prep.process(text_a)
        shingles_b = prep.process(text_b)
        if shingles_a and shingles_b:
            mh_a = MinHash(num_hashes=num_hashes)
            mh_a.add_shingles(shingles_a)
            mh_b = MinHash(num_hashes=num_hashes)
            mh_b.add_shingles(shingles_b)
            minhash_sim = mh_a.jaccard_similarity(mh_b)
        else:
            minhash_sim = 0.0
        minhash_pred = 1 if minhash_sim >= threshold else 0

        results.append({
            "text_a": text_a[:150] + "..." if len(text_a) > 150 else text_a,
            "text_b": text_b[:150] + "..." if len(text_b) > 150 else text_b,
            "true_label": true_label,
            "simhash_sim": round(simhash_sim, 4),
            "simhash_pred": simhash_pred,
            "minhash_sim": round(minhash_sim, 4),
            "minhash_pred": minhash_pred,
            "simhash_error": "FP" if (simhash_pred == 1 and true_label == 0) else ("FN" if (simhash_pred == 0 and true_label == 1) else "OK"),
            "minhash_error": "FP" if (minhash_pred == 1 and true_label == 0) else ("FN" if (minhash_pred == 0 and true_label == 1) else "OK")
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    print("\n" + "="*80)
    print("ERROR ANALYSIS SUMMARY")
    print("="*80)

    total = len(df_results)
    simhash_fp = len(df_results[df_results["simhash_error"] == "FP"])
    simhash_fn = len(df_results[df_results["simhash_error"] == "FN"])
    minhash_fp = len(df_results[df_results["minhash_error"] == "FP"])
    minhash_fn = len(df_results[df_results["minhash_error"] == "FN"])

    print(f"\nTotal pairs analyzed: {total}")
    print(f"\n SimHash Errors:")
    print(f"   False Positives (FP): {simhash_fp} ({simhash_fp/total*100:.2f}%)")
    print(f"   False Negatives (FN): {simhash_fn} ({simhash_fn/total*100:.2f}%)")
    print(f"\n MinHash Errors:")
    print(f"   False Positives (FP): {minhash_fp} ({minhash_fp/total*100:.2f}%)")
    print(f"   False Negatives (FN): {minhash_fn} ({minhash_fn/total*100:.2f}%)")


    def show_samples(df, error_type, method, n=2):
        col = f"{method}_error"
        samples = df[df[col] == error_type].head(n)
        if len(samples) == 0:
            print(f"\n No {error_type} samples found for {method}.")
            return
        print(f"\n {n} samples of {error_type} for {method}:")
        for i, (_, row) in enumerate(samples.iterrows(), 1):
            print(f"\n  Sample {i}:")
            print(f"    Text A: {row['text_a']}")
            print(f"    Text B: {row['text_b']}")
            print(f"    True Label: {row['true_label']}")
            print(f"    {method.capitalize()} Similarity: {row[f'{method}_sim']}")
            print(f"    {method.capitalize()} Prediction: {row[f'{method}_pred']}")

    show_samples(df_results, "FP", "simhash")
    show_samples(df_results, "FN", "simhash")
    show_samples(df_results, "FP", "minhash")
    show_samples(df_results, "FN", "minhash")

    print("\n" + "="*80)
    print("INTERESTING CASES (Deep Dive)")
    print("="*80)

    mixed = df_results[(df_results["simhash_error"] == "OK") & (df_results["minhash_error"] != "OK")]
    if len(mixed) > 0:
        print("\nSimHash OK, but MinHash Error:")
        for i, (_, row) in enumerate(mixed.head(3).iterrows(), 1):
            print(f"\n  Case {i}:")
            print(f"    Text A: {row['text_a']}")
            print(f"    Text B: {row['text_b']}")
            print(f"    True Label: {row['true_label']}")
            print(f"    SimHash Sim: {row['simhash_sim']} (Pred: {row['simhash_pred']})")
            print(f"    MinHash Sim: {row['minhash_sim']} (Pred: {row['minhash_pred']})")
            print(f"    Error Type (MinHash): {row['minhash_error']}")

    mixed = df_results[(df_results["minhash_error"] == "OK") & (df_results["simhash_error"] != "OK")]
    if len(mixed) > 0:
        print("\n MinHash OK, but SimHash Error:")
        for i, (_, row) in enumerate(mixed.head(3).iterrows(), 1):
            print(f"\n  Case {i}:")
            print(f"    Text A: {row['text_a']}")
            print(f"    Text B: {row['text_b']}")
            print(f"    True Label: {row['true_label']}")
            print(f"    MinHash Sim: {row['minhash_sim']} (Pred: {row['minhash_pred']})")
            print(f"    SimHash Sim: {row['simhash_sim']} (Pred: {row['simhash_pred']})")
            print(f"    Error Type (SimHash): {row['simhash_error']}")

    print("\n" + "="*80)
    print("Analysis complete. Full results saved to:", output_csv)

if __name__ == "__main__":
    analyze_errors(
        pairs_file="data/raw/quora/questions.csv",
        limit=5000,          
        shingle_size=3,       
        num_hashes=128,        
        threshold=0.4,         
        output_csv="outputs/error_analysis_results.csv"
    )
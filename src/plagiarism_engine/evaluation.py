import time
import os
import math
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict

from src.plagiarism_engine.simhash import SimHash
from src.plagiarism_engine.minhash import MinHash
from src.plagiarism_engine.preprocessing import TextPreprocessor

class EngineEvaluator:
    def __init__(self, threshold=0.5, num_hashes=128):
        self.threshold = threshold
        self.num_hashes = num_hashes
        self.simhash_engine = SimHash(bits=64)

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"Precision": round(precision, 4), "Recall": round(recall, 4), "F1-Score": round(f1, 4)}

    def evaluate_on_pairs(self, df_pairs, text_col_a, text_col_b, label_col, text_processor):
        y_true = df_pairs[label_col].astype(int).tolist()

        doc_freq = Counter()
        total_docs = 0
        for _, row in df_pairs.iterrows():
            text_a = str(row[text_col_a])
            text_b = str(row[text_col_b])
            tokens_a = text_processor.tokenize(text_processor.clean_text(text_a))
            tokens_b = text_processor.tokenize(text_processor.clean_text(text_b))
            tokens_a = text_processor.remove_stopwords(tokens_a)
            tokens_b = text_processor.remove_stopwords(tokens_b)
            for tokens in (tokens_a, tokens_b):
                if tokens:
                    unique = set(tokens)
                    doc_freq.update(unique)
                    total_docs += 1

        if total_docs == 0:
            idf = {}
        else:
            idf = {}
            for token, df in doc_freq.items():
                idf[token] = math.log((total_docs + 1) / (df + 1)) + 1

        start = time.time()
        simhash_preds = []
        for _, row in df_pairs.iterrows():
            text_a = str(row[text_col_a])
            text_b = str(row[text_col_b])
            tokens_a = text_processor.tokenize(text_processor.clean_text(text_a))
            tokens_b = text_processor.tokenize(text_processor.clean_text(text_b))
            tokens_a = text_processor.remove_stopwords(tokens_a)
            tokens_b = text_processor.remove_stopwords(tokens_b)
            weights_a = TextPreprocessor.compute_tfidf_weights(tokens_a, idf)
            weights_b = TextPreprocessor.compute_tfidf_weights(tokens_b, idf)
            if weights_a and weights_b:
                fp_a = self.simhash_engine.compute_fingerprint(weights_a)
                fp_b = self.simhash_engine.compute_fingerprint(weights_b)
                sim = self.simhash_engine.similarity(fp_a, fp_b)
                simhash_preds.append(1 if sim >= self.threshold else 0)
            else:
                simhash_preds.append(0)
        simhash_time = time.time() - start
        simhash_metrics = self.calculate_metrics(y_true, simhash_preds)

        start = time.time()
        minhash_preds = []
        for _, row in df_pairs.iterrows():
            text_a = str(row[text_col_a])
            text_b = str(row[text_col_b])
            shingles_a = text_processor.process(text_a)
            shingles_b = text_processor.process(text_b)
            m1 = MinHash(num_hashes=self.num_hashes)
            m1.add_shingles(shingles_a)
            m2 = MinHash(num_hashes=self.num_hashes)
            m2.add_shingles(shingles_b)
            sim = m1.jaccard_similarity(m2)
            minhash_preds.append(1 if sim >= self.threshold else 0)
        minhash_time = time.time() - start
        minhash_metrics = self.calculate_metrics(y_true, minhash_preds)

        df = pd.DataFrame({
            "Method": ["SimHash", "MinHash"],
            "Precision": [simhash_metrics["Precision"], minhash_metrics["Precision"]],
            "Recall": [simhash_metrics["Recall"], minhash_metrics["Recall"]],
            "F1-Score": [simhash_metrics["F1-Score"], minhash_metrics["F1-Score"]],
            "Execution_Time_Sec": [round(simhash_time, 4), round(minhash_time, 4)]
        })
        return df

def save_metrics(df, output_path):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Metrics saved to {output_path}")
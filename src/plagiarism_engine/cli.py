import argparse
import sys
import os
import json
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.plagiarism_engine.preprocessing import TextPreprocessor
from src.plagiarism_engine.minhash import MinHash
from src.plagiarism_engine.lsh import LSH
from src.plagiarism_engine.simhash import SimHash
from src.plagiarism_engine.dataset import PANPC11Loader
from src.plagiarism_engine.evaluation import EngineEvaluator, save_metrics

def load_text_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def compare_command(args):
    text_a = load_text_file(args.file_a)
    text_b = load_text_file(args.file_b)
    prep = TextPreprocessor(shingle_size=args.shingle_size)

    shingles_a = prep.process(text_a)
    shingles_b = prep.process(text_b)
    mh_a = MinHash(num_hashes=args.num_hashes)
    mh_a.add_shingles(shingles_a)
    mh_b = MinHash(num_hashes=args.num_hashes)
    mh_b.add_shingles(shingles_b)
    minhash_sim = mh_a.jaccard_similarity(mh_b)

    tokens_a = prep.tokenize(prep.clean_text(text_a))
    tokens_b = prep.tokenize(prep.clean_text(text_b))
    tokens_a = prep.remove_stopwords(tokens_a)
    tokens_b = prep.remove_stopwords(tokens_b)
    
    all_tokens = [tokens_a, tokens_b]
    idf = TextPreprocessor.compute_idf(all_tokens)
    weights_a = TextPreprocessor.compute_tfidf_weights(tokens_a, idf)
    weights_b = TextPreprocessor.compute_tfidf_weights(tokens_b, idf)
    
    simhash_engine = SimHash(bits=64)
    fp_a = simhash_engine.compute_fingerprint(weights_a) if weights_a else 0
    fp_b = simhash_engine.compute_fingerprint(weights_b) if weights_b else 0
    simhash_sim = simhash_engine.similarity(fp_a, fp_b) if weights_a and weights_b else 0.0

    result = {
        "file_a": args.file_a,
        "file_b": args.file_b,
        "shingle_size": args.shingle_size,
        "num_hashes": args.num_hashes,
        "MinHash_similarity": round(minhash_sim, 4),
        "SimHash_similarity": round(simhash_sim, 4)
    }
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[+] Compare result saved to {args.output}")

def corpus_command(args):
    print(f"[*] Loading corpus from {args.data}")
    loader = PANPC11Loader(args.data)
    prep = TextPreprocessor(shingle_size=args.shingle_size)
    lsh = LSH(num_hashes=args.num_hashes, num_bands=args.num_bands)

    doc_paths = []
    for path in loader.iter_document_paths():
        doc_paths.append(path)
        if args.max_docs and len(doc_paths) >= args.max_docs:
            break

    print(f"[*] Total documents: {len(doc_paths)}")
    start = time.time()

    print("[*] Building LSH index...")
    doc_count = 0
    for idx, doc_path in enumerate(doc_paths):
        if idx % 1000 == 0:
            print(f"    Progress: {idx}/{len(doc_paths)}")
        doc_id = os.path.basename(doc_path).replace('.txt', '')
        content = loader.load_document(doc_path)
        shingles = prep.process(content)
        if shingles:
            mh = MinHash(num_hashes=args.num_hashes)
            mh.add_shingles(shingles)
            lsh.add_document(doc_id, mh.get_signature())
            doc_count += 1

    candidates = lsh.get_candidates()
    elapsed = time.time() - start
    print(f"[*] LSH found {len(candidates)} candidate pairs in {elapsed:.2f}s")

    reduction_ratio = LSH.candidate_reduction_ratio(doc_count, candidates)
    print(f"[*] Candidate reduction ratio: {reduction_ratio:.2%}")

    pairs = [(lsh.doc_ids[i], lsh.doc_ids[j]) for i, j in candidates]
    df = pd.DataFrame(pairs, columns=['doc1', 'doc2'])
    df['method'] = 'LSH'
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[+] Candidates saved to {args.output}")

    summary = {
        "total_docs": doc_count,
        "num_candidates": len(candidates),
        "num_hashes": args.num_hashes,
        "num_bands": args.num_bands,
        "shingle_size": args.shingle_size,
        "time_sec": round(elapsed, 2),
        "reduction_ratio": round(reduction_ratio, 4)
    }
    summary_path = args.output.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Summary saved to {summary_path}")

def pairs_command(args):
    df = pd.read_csv(args.pairs)
    if args.limit and args.limit < len(df):
        df = df.head(args.limit)

    prep = TextPreprocessor(shingle_size=args.shingle_size)
    evaluator = EngineEvaluator(threshold=args.threshold, num_hashes=args.num_hashes)
    metrics_df = evaluator.evaluate_on_pairs(
        df, args.text_col_a, args.text_col_b, args.label_col, prep
    )
    save_metrics(metrics_df, args.output)
    print(metrics_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Plagiarism Detection CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_compare = subparsers.add_parser('compare')
    p_compare.add_argument('--file-a', required=True)
    p_compare.add_argument('--file-b', required=True)
    p_compare.add_argument('--shingle-size', type=int, default=3, choices=[1, 2, 3, 4, 5])
    p_compare.add_argument('--num-hashes', type=int, default=128, choices=[128, 256])
    p_compare.add_argument('--output', default='outputs/compare.json')
    p_compare.set_defaults(func=compare_command)

    p_corpus = subparsers.add_parser('corpus')
    p_corpus.add_argument('--data', required=True)
    p_corpus.add_argument('--num-hashes', type=int, default=128, choices=[128, 256])
    p_corpus.add_argument('--num-bands', type=int, default=16)
    p_corpus.add_argument('--shingle-size', type=int, default=3, choices=[1, 2, 3, 4, 5])
    p_corpus.add_argument('--max-docs', type=int, default=None)
    p_corpus.add_argument('--output', default='outputs/candidates.csv')
    p_corpus.set_defaults(func=corpus_command)

    p_pairs = subparsers.add_parser('pairs')
    p_pairs.add_argument('--pairs', required=True)
    p_pairs.add_argument('--text-col-a', required=True)
    p_pairs.add_argument('--text-col-b', required=True)
    p_pairs.add_argument('--label-col', required=True)
    p_pairs.add_argument('--limit', type=int, default=5000)
    p_pairs.add_argument('--threshold', type=float, default=0.5)
    p_pairs.add_argument('--shingle-size', type=int, default=3, choices=[1, 2, 3, 4, 5])
    p_pairs.add_argument('--num-hashes', type=int, default=128, choices=[128, 256])
    p_pairs.add_argument('--output', default='outputs/metrics.csv')
    p_pairs.set_defaults(func=pairs_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
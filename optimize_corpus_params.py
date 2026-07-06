import os
import time
import random
import pandas as pd
import subprocess
from src.plagiarism_engine.dataset import PANPC11Loader

def get_random_docs(data_path, num_samples=5000, seed=42):
    random.seed(seed)
    loader = PANPC11Loader(data_path)
    all_docs = list(loader.iter_document_paths())
    selected = random.sample(all_docs, min(num_samples, len(all_docs)))
    return selected

def run_corpus(doc_paths, params, output_dir="outputs/optimize_temp"):
    shingle, hashes, bands = params
    os.makedirs(output_dir, exist_ok=True)
    
    temp_corpus_dir = os.path.join(output_dir, f"temp_corpus_{shingle}_{hashes}_{bands}")
    os.makedirs(temp_corpus_dir, exist_ok=True)
    
    import shutil
    for src_path in doc_paths:
        dst_path = os.path.join(temp_corpus_dir, os.path.basename(src_path))
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
    
    output_file = os.path.join(output_dir, f"corpus_{shingle}_{hashes}_{bands}.csv")
    
    cmd = [
        "python", "-m", "src.plagiarism_engine.cli", "corpus",
        "--data", temp_corpus_dir,
        "--num-hashes", str(hashes),
        "--num-bands", str(bands),
        "--shingle-size", str(shingle),
        "--max-docs", str(len(doc_paths)),
        "--output", output_file
    ]
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"Error for params {params}: {result.stderr[:200]}")
            return None, elapsed
        summary_path = output_file.replace('.csv', '_summary.json')
        if os.path.exists(summary_path):
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            candidates = summary.get('num_candidates', 0)
            total_docs = summary.get('total_docs', 0)
            reduction = 1.0 - (candidates / (total_docs * (total_docs - 1) / 2)) if total_docs > 1 else 0.0
            return {
                'candidates': candidates,
                'reduction': reduction,
                'total_docs': total_docs,
                'time': elapsed,
                'output_file': output_file
            }, elapsed
        else:
            return None, elapsed
    except subprocess.TimeoutExpired:
        print(f"Timeout for params {params}")
        return None, 999.0
    except Exception as e:
        print(f"Exception for params {params}: {e}")
        return None, 999.0
    finally:
        import shutil
        if os.path.exists(temp_corpus_dir):
            shutil.rmtree(temp_corpus_dir)

def grid_search_corpus(doc_paths):
    shingle_sizes = [3, 4, 5]
    num_hashes_list = [128, 256]
    results = []
    best_score = None
    best_params = None
    
    total = len(shingle_sizes) * len(num_hashes_list) * 6 
    tested = 0
    
    for shingle in shingle_sizes:
        for hashes in num_hashes_list:
            bands_options = []
            for b in [4, 8, 16, 32, 64]:
                if hashes % b == 0:
                    bands_options.append(b)
            if not bands_options:
                bands_options = [4]
            
            for bands in bands_options:
                tested += 1
                params = (shingle, hashes, bands)
                print(f"\n[{tested}/{total}] Testing: shingle={shingle}, hashes={hashes}, bands={bands}")
                
                result, elapsed = run_corpus(doc_paths, params)
                if result is None:
                    continue
                reduction = result['reduction']
                candidates = result['candidates']
                time_val = elapsed
                
                time_score = max(0, 1 - (time_val / 100))
                score = 0.7 * reduction + 0.3 * time_score
                
                results.append({
                    'shingle_size': shingle,
                    'num_hashes': hashes,
                    'num_bands': bands,
                    'candidates': candidates,
                    'reduction': reduction,
                    'time_sec': round(time_val, 2),
                    'score': round(score, 4)
                })
                
                print(f"   Candidates: {candidates}, Reduction: {reduction:.2%}, Time: {time_val:.2f}s, Score: {score:.4f}")
                
                if best_score is None or score > best_score:
                    best_score = score
                    best_params = params
    
    df = pd.DataFrame(results)
    df.to_csv("outputs/corpus_param_optimization_results.csv", index=False)
    print("\nResults saved to: outputs/corpus_param_optimization_results.csv")
    
    return best_params, best_score, df

if __name__ == "__main__":
    data_path = "data/raw/pan-plagiarism-corpus-2011/external-detection-corpus"
    print("Loading PAN corpus...")
    doc_paths = get_random_docs(data_path, num_samples=5000)
    print(f"Selected {len(doc_paths)} documents randomly (seed=42).")
    
    best_params, best_score, df = grid_search_corpus(doc_paths)
    
    if best_params:
        shingle, hashes, bands = best_params
        print("\n" + "="*60)
        print("BEST PARAMETERS FOUND:")
        print(f"   shingle_size = {shingle}")
        print(f"   num_hashes   = {hashes}")
        print(f"   num_bands    = {bands}")
        print(f"   Score        = {best_score:.4f}")
        print("="*60)
        print("\nRecommended command for full PAN corpus:")
        print(f"python -m src.plagiarism_engine.cli corpus \\")
        print(f"  --data \"{data_path}\" \\")
        print(f"  --num-hashes {hashes} --num-bands {bands} --shingle-size {shingle} \\")
        print(f"  --max-docs 26939 --output outputs/corpus_optimized.csv")
    else:
        print("No valid parameters found. Check your data and try again.")
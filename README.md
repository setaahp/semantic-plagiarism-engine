# Semantic Plagiarism Detection Engine

A semantic plagiarism detection engine implemented **from scratch** using **MinHash + Locality Sensitive Hashing (LSH)** and **TF-IDF Weighted SimHash**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

# Overview

This project detects similar and plagiarized documents using two complementary approaches:

- **MinHash + LSH**
  - Efficient candidate generation
  - Reduces document comparisons from **O(n²)** to approximately **O(n)**

- **TF-IDF Weighted SimHash**
  - Generates 64-bit semantic fingerprints
  - Fast similarity estimation using Hamming distance

All algorithms are implemented **from scratch** without relying on plagiarism detection libraries.

---

# Project Structure

```text
semantic-plagiarism-engine/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   └── plagiarism_engine/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── dataset.py
│       ├── minhash.py
│       ├── lsh.py
│       ├── simhash.py
│       ├── evaluation.py
│       └── cli.py
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   └── test_engine.py
└── outputs/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/setaahp/semantic-plagiarism-engine.git
cd semantic-plagiarism-engine
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

# Datasets

## Quora Question Pairs

Download from:

https://www.kaggle.com/c/quora-question-pairs

Place the dataset at:

```text
data/raw/quora/questions.csv
```

## PAN-PC-11

Download from:

https://pan.webis.de

Place it inside:

```text
data/raw/pan-plagiarism-corpus-2011/
```

---

# Command Line Interface

## Compare Two Documents

```bash
python -m src.plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc1.txt \
    --file-b data/sample_corpus/doc2.txt \
    --shingle-size 3 \
    --num-hashes 128 \
    --output outputs/compare.json
```

Example output

```json
{
  "file_a": "doc1.txt",
  "file_b": "doc2.txt",
  "shingle_size": 3,
  "num_hashes": 128,
  "MinHash_similarity": 0.1562,
  "SimHash_similarity": 0.7031
}
```

---

## Find Similar Documents in a Corpus

```bash
python -m src.plagiarism_engine.cli corpus \
    --data data/raw/pan-plagiarism-corpus-2011/external-detection-corpus \
    --num-hashes 128 \
    --num-bands 16 \
    --shingle-size 1 \
    --output outputs/candidates.csv
```

Outputs:

- Candidate pairs (CSV)
- Summary statistics (JSON)

---

## Evaluate on Labeled Document Pairs

```bash
python -m src.plagiarism_engine.cli pairs \
    --pairs data/raw/quora/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --threshold 0.5 \
    --shingle-size 3 \
    --num-hashes 128 \
    --output outputs/metrics.csv
```

Example metrics

| Method | Precision | Recall | F1 | Time (s) |
|---------|----------:|-------:|---:|---------:|
| SimHash | 0.4074 | 0.9942 | 0.5780 | 2.02 |
| MinHash | 0.5681 | 0.6349 | 0.5997 | 1.29 |

---

# Experimental Results

## Quora Question Pairs (5,000 pairs)

| Method | Best Parameters | Precision | Recall | F1 | Time |
|---------|----------------|----------:|-------:|---:|------:|
| SimHash | shingle=1, hashes=128, threshold=0.5 | 0.4074 | 0.9942 | 0.5780 | 2.02 s |
| MinHash | shingle=1, hashes=128, threshold=0.4 | 0.5681 | 0.6349 | 0.5997 | 1.29 s |

---

## PAN-PC-11 Corpus (26,939 documents)

| Metric | Value |
|---------|------:|
| Total documents | 26,939 |
| Possible pairs | ≈362,800,000 |
| Candidate pairs after LSH | 26,197 |
| Reduction | 99.99% |
| Runtime | ~68 minutes |

---

# Recommended Parameters

| Command | Shingle Size | Hashes | Bands | Threshold |
|----------|-------------:|-------:|------:|----------:|
| compare | 3 | 128 | - | - |
| corpus | 3 | 128 | 16 | - |
| pairs | 1 or 3 | 128 | - | 0.4 (MinHash) / 0.5 (SimHash) |

---

# Running Tests

```bash
pytest tests/test_engine.py -v
```

---

# License

This project is licensed under the MIT License.

---

# Author

**Setayesh**

GitHub:

https://github.com/setaahp
https://github.com/iamhamidhosseini

---

Developed as a university project for semantic plagiarism detection using MinHash, LSH, and SimHash.

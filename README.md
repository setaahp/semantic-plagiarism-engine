# Semantic Plagiarism Engine

Implementation of two methods for plagiarism and near-duplicate document detection:
- **MinHash + LSH** for approximate similarity and candidate reduction
- **SimHash with TF‑IDF weighting** for fast fingerprinting

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
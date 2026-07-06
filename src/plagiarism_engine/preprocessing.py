import re
import math
from collections import Counter
from typing import List, Set, Dict

ENGLISH_STOPWORDS = {
    'the', 'is', 'at', 'which', 'on', 'of', 'for', 'a', 'an', 'and', 'or',
    'in', 'to', 'with', 'by', 'from', 'as', 'are', 'was', 'were', 'has', 'have',
    'be', 'been', 'being', 'am', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must'
}

class TextPreprocessor:
    def __init__(self, shingle_size=3):
        self.shingle_size = shingle_size
        self.stopwords = ENGLISH_STOPWORDS

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        return [t for t in tokens if t not in self.stopwords]

    def generate_shingles(self, tokens: List[str]) -> Set[str]:
        if len(tokens) < self.shingle_size:
            return set()
        shingles = set()
        for i in range(len(tokens) - self.shingle_size + 1):
            shingle = ' '.join(tokens[i:i+self.shingle_size])
            shingles.add(shingle)
        return shingles

    def process(self, text: str) -> Set[str]:
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        return self.generate_shingles(filtered)

    @staticmethod
    def compute_tf(tokens: List[str]) -> Dict[str, float]:
        freq = Counter(tokens)
        total = len(tokens)
        if total == 0:
            return {}
        return {w: count / total for w, count in freq.items()}

    @staticmethod
    def compute_idf(corpus_tokens: List[List[str]]) -> Dict[str, float]:
        doc_count = len(corpus_tokens)
        if doc_count == 0:
            return {}
        doc_freq = Counter()
        for tokens in corpus_tokens:
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)
        idf = {}
        for token, df in doc_freq.items():
            idf[token] = math.log((doc_count + 1) / (df + 1)) + 1
        return idf

    @staticmethod
    def compute_tfidf_weights(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
        tf = TextPreprocessor.compute_tf(tokens)
        weights = {}
        for token, tf_val in tf.items():
            if token in idf:
                weights[token] = tf_val * idf[token]
        return weights
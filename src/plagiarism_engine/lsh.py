import numpy as np
import hashlib
from collections import defaultdict
from typing import List, Set, Tuple, Dict

class LSH:
    def __init__(self, num_hashes: int = 128, num_bands: int = 16):
        if num_hashes % num_bands != 0:
            raise ValueError("num_hashes must be divisible by num_bands")
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.num_rows = num_hashes // num_bands
        self.buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.doc_ids: List[str] = []
        self.signatures: List[np.ndarray] = []

    @staticmethod
    def _hash_band(band_values: np.ndarray) -> int:
        return int(hashlib.md5(band_values.tobytes()).hexdigest(), 16)

    def add_document(self, doc_id: str, signature: np.ndarray):
        if len(signature) != self.num_hashes:
            raise ValueError("Signature length mismatch")
        idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.signatures.append(signature)

        for band_idx in range(self.num_bands):
            start = band_idx * self.num_rows
            end = start + self.num_rows
            band = signature[start:end]
            band_hash = self._hash_band(band)
            key = (band_idx, band_hash)
            self.buckets[key].append(idx)

    def get_candidates(self) -> Set[Tuple[int, int]]:
        candidates = set()
        for bucket in self.buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i+1, len(bucket)):
                        a, b = bucket[i], bucket[j]
                        if a < b:
                            candidates.add((a, b))
                        else:
                            candidates.add((b, a))
        return candidates

    def query(self, signature: np.ndarray) -> List[int]:
        if len(signature) != self.num_hashes:
            raise ValueError("Signature length mismatch")
        result = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.num_rows
            end = start + self.num_rows
            band = signature[start:end]
            band_hash = self._hash_band(band)
            key = (band_idx, band_hash)
            if key in self.buckets:
                result.update(self.buckets[key])
        return list(result)

    @staticmethod
    def candidate_reduction_ratio(total_docs: int, candidate_pairs: Set[Tuple[int, int]]) -> float:
        total_pairs = total_docs * (total_docs - 1) // 2
        if total_pairs == 0:
            return 0.0
        return 1.0 - (len(candidate_pairs) / total_pairs)
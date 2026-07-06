import numpy as np
import hashlib
from typing import Set

class MinHash:
    def __init__(self, num_hashes: int = 128, seed: int = 42):
        self.num_hashes = num_hashes
        self.seed = seed
        self.hash_values = np.full(num_hashes, 2**64-1, dtype=np.uint64)
        self._generate_hash_functions()
        self.has_content = False

    def _generate_hash_functions(self):
        np.random.seed(self.seed)
        self.a = np.random.randint(1, 2**32-1, size=self.num_hashes, dtype=np.uint64)
        self.b = np.random.randint(0, 2**32-1, size=self.num_hashes, dtype=np.uint64)
        self.mod = 2**32 - 1

    @staticmethod
    def _hash_shingle(shingle: str) -> int:
        return int(hashlib.md5(shingle.encode('utf-8')).hexdigest(), 16) & 0xFFFFFFFFFFFFFFFF

    def add_shingle(self, shingle: str):
        h = self._hash_shingle(shingle)
        hashes = (self.a * h + self.b) % self.mod
        self.hash_values = np.minimum(self.hash_values, hashes)
        self.has_content = True

    def add_shingles(self, shingles: Set[str]):
        for shingle in shingles:
            self.add_shingle(shingle)

    def get_signature(self) -> np.ndarray:
        return self.hash_values.astype(np.uint64)

    def jaccard_similarity(self, other: 'MinHash') -> float:
        if not self.has_content and not other.has_content:
            return 0.0
        if not self.has_content or not other.has_content:
            return 0.0
        sig1 = self.get_signature()
        sig2 = other.get_signature()
        equal = np.sum(sig1 == sig2)
        return equal / self.num_hashes
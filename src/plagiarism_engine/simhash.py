import hashlib
import numpy as np
from typing import Dict

class SimHash:
    def __init__(self, bits: int = 64):
        self.bits = bits

    @staticmethod
    def _hash_token(token: str) -> np.ndarray:
        hex_hash = hashlib.md5(token.encode('utf-8')).hexdigest()
        int_val = int(hex_hash[:16], 16)
        bits = np.array([(int_val >> i) & 1 for i in range(64)], dtype=np.int8)
        return np.where(bits == 1, 1, -1)

    def compute_fingerprint(self, token_weights: Dict[str, float]) -> int:
        if not token_weights:
            return 0
        vector = np.zeros(self.bits, dtype=np.float64)
        for token, weight in token_weights.items():
            vector += self._hash_token(token) * weight
        fingerprint = 0
        for i in range(self.bits):
            if vector[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    @staticmethod
    def similarity(hash1: int, hash2: int) -> float:
        xor = hash1 ^ hash2
        distance = bin(xor).count('1')
        return 1.0 - (distance / 64)
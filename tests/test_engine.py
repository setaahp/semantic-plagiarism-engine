import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plagiarism_engine.preprocessing import TextPreprocessor
from src.plagiarism_engine.minhash import MinHash
from src.plagiarism_engine.lsh import LSH
from src.plagiarism_engine.simhash import SimHash
from src.plagiarism_engine.evaluation import EngineEvaluator
from src.plagiarism_engine.dataset import PANPC11Loader

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor(shingle_size=3)

    def test_clean_text(self):
        text = "  This is a SAMPLE text, with punctuation!  "
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "this is a sample text with punctuation")

    def test_tokenize(self):
        text = "this is a sample"
        tokens = self.preprocessor.tokenize(text)
        self.assertEqual(tokens, ["this", "is", "a", "sample"])

    def test_remove_stopwords(self):
        tokens = ["this", "is", "a", "sample", "text"]
        filtered = self.preprocessor.remove_stopwords(tokens)
        self.assertEqual(filtered, ["sample", "text"])

    def test_generate_shingles_returns_set(self):
        tokens = ["the", "quick", "brown", "fox", "jumps"]
        shingles = self.preprocessor.generate_shingles(tokens)
        expected = {"the quick brown", "quick brown fox", "brown fox jumps"}
        self.assertEqual(shingles, expected)
        self.assertIsInstance(shingles, set)

    def test_process(self):
        text = "The quick brown fox jumps over the lazy dog"
        shingles = self.preprocessor.process(text)
        self.assertTrue(len(shingles) > 0)
        for sh in shingles:
            self.assertFalse(any(stop in sh for stop in ['the', 'over', 'is']))

class TestMinHash(unittest.TestCase):
    def setUp(self):
        self.mh1 = MinHash(num_hashes=128, seed=42)
        self.mh2 = MinHash(num_hashes=128, seed=42)

    def test_add_shingle(self):
        self.mh1.add_shingle("hello world")
        self.mh2.add_shingle("hello world")
        sim = self.mh1.jaccard_similarity(self.mh2)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_different_shingles(self):
        self.mh1.add_shingle("hello world")
        self.mh2.add_shingle("goodbye world")
        sim = self.mh1.jaccard_similarity(self.mh2)
        self.assertLess(sim, 0.3)

    def test_signature_length(self):
        self.mh1.add_shingle("test")
        sig = self.mh1.get_signature()
        self.assertEqual(len(sig), 128)

    def test_empty_documents(self):
        sim = self.mh1.jaccard_similarity(self.mh2)
        self.assertEqual(sim, 0.0)
        self.mh2.add_shingle("test")
        sim = self.mh1.jaccard_similarity(self.mh2)
        self.assertEqual(sim, 0.0)

class TestLSH(unittest.TestCase):
    def setUp(self):
        self.lsh = LSH(num_hashes=128, num_bands=16)

    def test_add_and_candidates(self):
        sig1 = np.random.randint(0, 2**32, size=128, dtype=np.uint64)
        sig2 = sig1.copy()
        sig3 = np.random.randint(0, 2**32, size=128, dtype=np.uint64)

        self.lsh.add_document("doc1", sig1)
        self.lsh.add_document("doc2", sig2)
        self.lsh.add_document("doc3", sig3)

        candidates = self.lsh.get_candidates()
        self.assertIn((0,1), candidates)
        self.assertNotIn((0,2), candidates)

    def test_query(self):
        sig = np.random.randint(0, 2**32, size=128, dtype=np.uint64)
        self.lsh.add_document("doc1", sig)
        self.lsh.add_document("doc2", sig)
        self.lsh.add_document("doc3", np.random.randint(0, 2**32, size=128, dtype=np.uint64))

        candidates = self.lsh.query(sig)
        self.assertEqual(set(candidates), {0,1})

    def test_reduction_ratio(self):
        ratio = LSH.candidate_reduction_ratio(10, {(0,1), (2,3)})
        total_pairs = 45
        expected = 1 - (2/45)
        self.assertAlmostEqual(ratio, expected, places=4)

class TestSimHash(unittest.TestCase):
    def setUp(self):
        self.simhash = SimHash(bits=64)

    def test_fingerprint(self):
        weights = {"hello": 0.5, "world": 0.5}
        fp = self.simhash.compute_fingerprint(weights)
        self.assertIsInstance(fp, int)
        self.assertTrue(0 <= fp < 2**64)

    def test_similarity(self):
        weights1 = {"hello": 0.6, "world": 0.4}
        weights2 = {"hello": 0.6, "world": 0.4}
        weights3 = {"goodbye": 1.0}

        fp1 = self.simhash.compute_fingerprint(weights1)
        fp2 = self.simhash.compute_fingerprint(weights2)
        fp3 = self.simhash.compute_fingerprint(weights3)

        sim_same = self.simhash.similarity(fp1, fp2)
        sim_diff = self.simhash.similarity(fp1, fp3)

        self.assertAlmostEqual(sim_same, 1.0, places=4)
        self.assertLess(sim_diff, 0.5)

    def test_empty_weights(self):
        fp = self.simhash.compute_fingerprint({})
        self.assertEqual(fp, 0)
        sim = self.simhash.similarity(0, 0)
        self.assertEqual(sim, 1.0)

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.evaluator = EngineEvaluator(threshold=0.5, num_hashes=128)

    def test_calculate_metrics(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        metrics = self.evaluator.calculate_metrics(y_true, y_pred)
        expected_precision = round(2/3, 4)
        expected_recall = round(2/3, 4)
        expected_f1 = round(2/3, 4)
        self.assertEqual(metrics["Precision"], expected_precision)
        self.assertEqual(metrics["Recall"], expected_recall)
        self.assertEqual(metrics["F1-Score"], expected_f1)

    def test_evaluate_on_pairs(self):
        df = pd.DataFrame({
            'text1': ["hello world", "goodbye world", "test data"],
            'text2': ["hello world", "hello world", "different data"],
            'label': [1, 0, 0]
        })
        prep = TextPreprocessor(shingle_size=2)
        metrics_df = self.evaluator.evaluate_on_pairs(
            df, 'text1', 'text2', 'label', prep
        )
        self.assertEqual(len(metrics_df), 2)
        self.assertIn('Precision', metrics_df.columns)
        self.assertIn('Recall', metrics_df.columns)
        self.assertIn('F1-Score', metrics_df.columns)
        self.assertIn('Execution_Time_Sec', metrics_df.columns)

if __name__ == '__main__':
    unittest.main()
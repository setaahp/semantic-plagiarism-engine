import os
from typing import Generator

class PANPC11Loader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.source_dirs = []
        self.suspicious_dirs = []

        for root, dirs, files in os.walk(data_path):
            if "source-document" in dirs:
                self.source_dirs.append(os.path.join(root, "source-document"))
            if "suspicious-document" in dirs:
                self.suspicious_dirs.append(os.path.join(root, "suspicious-document"))

        if not self.source_dirs:
            for root, dirs, files in os.walk(data_path):
                if "source-documents" in dirs:
                    self.source_dirs.append(os.path.join(root, "source-documents"))
        if not self.suspicious_dirs:
            for root, dirs, files in os.walk(data_path):
                if "suspicious-documents" in dirs:
                    self.suspicious_dirs.append(os.path.join(root, "suspicious-documents"))

    def _list_txt(self, dirs):
        files = []
        for d in dirs:
            for root, _, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith(".txt") and f not in ["readme.txt", "retrieval-task.txt"]:
                        files.append(os.path.join(root, f))
        return files

    def list_source_documents(self):
        return self._list_txt(self.source_dirs)

    def list_suspicious_documents(self):
        return self._list_txt(self.suspicious_dirs)

    def load_document(self, filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def iter_document_paths(self) -> Generator[str, None, None]:
        for path in self.list_source_documents():
            yield path
        for path in self.list_suspicious_documents():
            yield path

    def load_all_documents(self):
        docs = {}
        for path in self.iter_document_paths():
            doc_id = os.path.basename(path).replace('.txt', '')
            docs[doc_id] = self.load_document(path)
        return docs
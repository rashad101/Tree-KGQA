"""
FAISS-based index components. Original from
https://github.com/facebookresearch/DPR/blob/master/dpr/indexer/faiss_indexers.py
https://github.com/facebookresearch/BLINK/blob/master/elq/index/faiss_indexer.py
"""

import faiss
import numpy as np

class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def index_data(self, data: np.array):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int):
        raise NotImplementedError

    def serialize(self, index_file: str):
        print("Serializing index to ", index_file)
        faiss.write_index(self.index, index_file)

    def deserialize_from(self, index_file: str):
        print(f"Loading index from {index_file}")
        self.index = faiss.read_index(index_file)
        print(f"Loaded index of type {type(self.index)} and size {self.index.ntotal}")


# DenseHNSWFlatIndexer does approximate search
class DenseHNSWFlatIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
        self,
        vector_sz: int,
        buffer_size: int = 50000,
        store_n: int = 128,
        ef_search: int = 256,
        ef_construction: int = 200,
    ):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        index = faiss.IndexHNSWFlat(vector_sz, store_n, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index

    def index_data(self, data: np.array):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        print("Indexing data, this may take a while.")
        self.index.add(data)
        print(f"Total data indexed {n}")

    def search_knn(self, query_vectors, top_k):
        scores, indexes = self.index.search(query_vectors, top_k)
        return scores, indexes

    def deserialize_from(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = 1

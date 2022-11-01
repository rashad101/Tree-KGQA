import json
from sentence_transformers import SentenceTransformer
from utils.faiss_util import DenseHNSWFlatIndexer


class Indexer:
    def __init__(self):
        self.indexer = self.load_indexer()

    @staticmethod
    def load_indexer():
        """
        loads required files for indexing
        :return: required mapping and encoder for indexing
        """
        i2e = json.load(open("data/wikidata/i2e.json"))
        i2id = json.load(open("data/wikidata/i2id.json"))
        encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')

        indexer = DenseHNSWFlatIndexer(1)
        indexer.deserialize_from("data/wikidata/indexed_wikidata_entities.pkl")

        return {
            "i2e": i2e,
            "i2id": i2id,
            "indexer": indexer,
            "encoder": encoder
        }

    def lookup(self, text, topk=1):
        """
        Perform faiss_hnsw lookup
        :param topk: number of candidate entities
        :param text: text chunk to be looked up
        :return:  [labels],[ids] --> list of entity labels and entity ids
        """
        query_vector = self.indexer["encoder"].encode([text])
        sc, e_id = self.indexer["indexer"].search_knn(query_vector, topk)  # 1 -> means top entity
        return [self.indexer["i2e"][str(e_id[0][i])] for i in range(topk)],\
               [self.indexer["i2id"][str(e_id[0][i])] for i in range(topk)]


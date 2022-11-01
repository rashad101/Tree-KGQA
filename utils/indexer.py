from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
import json
from utils.faiss_util import DenseHNSWFlatIndexer
from datetime import datetime


def get_args():
    parser = ArgumentParser(description="Entity indexer")
    parser.add_argument("--output_path", required=True, type=str, help="output path")
    parser.add_argument("--faiss_index", type=str, default="hnsw", help='hnsw index')
    parser.add_argument('--index_buffer', type=int, default=50000)
    parser.add_argument("--save_index", action='store_true', help='save indexed file')
    parsed_args = parser.parse_args()
    parsed_args = parsed_args.__dict__
    return parsed_args


def main(args):

    data, idx2entity, idx2id = list(), dict(), dict()

    start_time = datetime.now()
    # entity.jsonl is available here: wget http://dl.fbaipublicfiles.com/elq/entity.jsonl
    with open("data/entity.jsonl") as f:
        for i,aline in enumerate(f):
            info = json.loads(aline.strip())
            data.append(info["entity"])
            idx2entity[i] = info["entity"]

            if "kb_idx" in info:
                idx2id[i] = info["kb_idx"]
            else:
                idx2id[i] = None

    print('Data loading Duration: {}'.format(datetime.now() - start_time))
    start_time = datetime.now()
    json.dump(idx2entity,open("data/wikidata/i2e.json","w"), ensure_ascii=False)
    json.dump(idx2id, open("data/wikidata/i2id.json", "w"), ensure_ascii=False)
    print('Data saving Duration: {}'.format(datetime.now() - start_time))

    start_time = datetime.now()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    print('Sentence Transformer loading Duration: {}'.format(datetime.nowe() - start_time))

    start_time = datetime.now()
    encoded_data = model.encode(['lionel'])
    print('Encoding Duration: {}'.format(datetime.nowe() - start_time))

    print("Using HNSW index in FAISS")
    vector_size = 768
    index = DenseHNSWFlatIndexer(vector_size, len(data))
    print("Building index.")
    start_time = datetime.now()
    index.index_data(encoded_data)
    print("Done indexing data.")
    print('Indexing Duration: {}'.format(datetime.now() - start_time))

    if args.save_index:                                                 # saving index
        print("Saving index file")
        index.serialize(args.output_path)
        print("Done")

if __name__ == '__main__':

    args = get_args()
    main(args)
import json
from gensim.models import fasttext
from sentence_transformers import SentenceTransformer
from components.bart_zeroshot_classifier import BARTClassifier
from components.relation_linking_embedding import WikidataPropertiesMatcher

class RelationLinker:
    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.prediction = list()

        self.rels = json.load(open("data/wikidata/relations.json"))["rows"]

        if args.vec!="sentencetransformer":
            self.vec = fasttext.load_facebook_vectors("data/wiki.simple.bin")
        else:
            self.vec = SentenceTransformer('distilbert-base-nli-mean-tokens')

        self.onehop = json.load(open("data/wikidata/onehop_comps.json"))
        self.entmap = json.load(open("data/wikidata/id2ent_mapping.json"))
        self.reltimer = 0
        self.kgqatimer = 0

        self.id2r = {r[0]: r[1] for r in self.rels}
        self.r2id = {r[1]: r[0] for r in self.rels}

        self.prop_matcher = WikidataPropertiesMatcher(all_relations=self.rels, word_vectors=self.vec, args=args, onehop=self.onehop)
        self.zs_classifier = BARTClassifier(args=args)

    def perform_RL_single(self, single_data):
        rl_output = self.prop_matcher.predict_rel(data=single_data, classifier=self.zs_classifier)
        return rl_output

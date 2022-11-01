import json
import pickle
from tqdm import tqdm
from time import time
from os.path import join as add_paths
from datetime import datetime
from components.tree import Tree
from utils.metrics import eval
from gensim.models import fasttext
from sentence_transformers import SentenceTransformer
from components.bart_zeroshot_classifier import BARTClassifier
from components.relation_linking_embedding import WikidataPropertiesMatcher
from components.relation_linking_graphlaplacian import GraphLaplacian

class KGQA:
    def __init__(self, entity_linker, data, args):
        self.el = entity_linker
        self.data = data
        self.args = args
        self.prediction = list()
        if self.args.use_kge:
            self.kge = self.load_kge()
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

        if self.args.RL_method!="graph-laplacian":
            self.prop_matcher = WikidataPropertiesMatcher(all_relations=self.rels, word_vectors=self.vec, args=args, onehop=self.onehop)
            self.zs_classifier = BARTClassifier(args=args)
            print("Selecting embedding based relation matcher")
        else:
            self.prop_matcher = GraphLaplacian(self.args, self.vec, self.id2r, self.r2id, onehop = self.onehop)
            self.zs_classifier = None
            print("Selecting Graph-Laplacian based relation matcher")



    def load_kge(self):
        with open("data/wikidata/kge/relemb_quatE.pkl","rb") as f:
            kge = pickle.load(f)
            return kge

    def process_question(self,question, mentions=[]):
        ques = question
        if question.endswith(".") or question.endswith("?"):
            ques = ques[:len(question)-1]
        for ment in mentions:
            ques = ques.replace(ment,"")
        return ques

    def perform_KGQA_single(self, single_data, el_output=None, rl_output=None):
        forest = list()  # list of trees
        answers = list()
        final_ans = single_data.copy()
        final_ans["pred_objects"] = list()
        final_ans["pred_relations"] = list()
        final_ans["tree_output"] = list()
        for entid in el_output["id"]:
            a_tree = Tree(args=self.args, root_node_id=entid, vec=self.vec, n_hop=self.args.n_hop, kge=self.kge if self.args.use_kge else None,onehop=self.onehop, rels=self.rels, ent_map=self.entmap)
            ques = self.process_question(single_data["question"], mentions=el_output["mentions"])

            if rl_output is None:      # when relation is not found, compute the relation with simple embedding
                a_tree.perform_tree_walk(ques, n_pass=True)
            else:                      # if relation is found, compute embedding-based similarity
                for a_rel in rl_output:
                    a_tree.perform_tree_walk(self.id2r[a_rel], n_pass=False, use_kg_emb=self.args.use_kge)

            answer = a_tree.get_max_edge(max_hop=self.args.n_hop)
            final_ans["pred_relations"].append(answer["relation_id"])
            final_ans["pred_objects"].extend(answer["object_entities"])
            answers.append(answer)
            forest.append(a_tree.tree)


        final_ans["forest"] = forest
        return final_ans.copy()

    def perform_KGQA(self):
        intersect = 0
        pred_intersect = 0
        for ii,d in tqdm(enumerate(self.data)):
            # entity linking step
            rl_output = list()
            if self.args.ablation and ii>299:
                break

            if self.args.true_EL:  # run with already linked entity (mainly for ablation)
                mentions = list()
                if "original_mentions" in d:
                    mentions = d["original_mentions"]
                el_output = {"id": d["wikidata_id"],"mentions": mentions}
                output = self.perform_KGQA_single(d, el_output=el_output)
            else:
                # Entity linking
                single_el_output = self.el.perform_EL_single(d)
                ids = [ent for ent in single_el_output["pred_wikiid"] if ent is not None]
                el_output = {"id": ids, "mentions": single_el_output["mention"]}
                temp_data = d.copy()
                temp_data.update(single_el_output)

                # relation linking
                st = time()
                rl_output = self.prop_matcher.predict_rel(data=temp_data, classifier=self.zs_classifier)
                en = time()
                self.reltimer+=(en-st)
                st = time()
                output = self.perform_KGQA_single(temp_data.copy(), el_output=el_output, rl_output=rl_output)
                en = time()
                self.kgqatimer+=(en-st)

            output["pred_objects"] = [out[out.rfind("/")+1:] if "/" in out else out for out in output["pred_objects"]]

            output["pred_relations"] = list(set(output["pred_relations"]))
            if len(set(d["wikidata_relation_id"]).intersection(output["pred_relations"]))>0:
                pred_intersect+=1
            print("REL: ",pred_intersect, output["wikidata_relation_id"], output["pred_relations"])
            if len(set(output["wikidata_objects"]).intersection(output["pred_objects"]))>0:
                intersect+=1
            print("ANS: ",intersect,d["wikidata_objects"], output["pred_objects"])
            self.prediction.append(output)

            # if ii==99:
            #     print("rel: ",self.reltimer)
            #     print("kgtime: ", self.kgqatimer)
            #     break


    def evaluate(self):
        eval(dataset=self.args.dataset, predictions=self.prediction, task=self.args.task)


    def save_predictions(self):
        prediction_dir = f"outputs/{'ablation' if self.args.ablation else 'kgqa'}/"
        dt_fmt = datetime.now().strftime("%d-%b-%Y_%H:%M:%S.%f")
        filename = f"KGQA_{self.args.dataset}_RL-{self.args.RL_method}_ELdisamb-{self.args.EL_disamb}_API-{'yes' if self.args.use_api else 'no'}_Indexing-{'yes' if self.args.use_indexing else 'no'}_paraphrase-{'yes' if self.args.paraphrase_q else 'no'}_uncased-{'yes' if self.args.uncased else 'no'}_KG-{self.args.kg}_{dt_fmt}.json"
        filename = filename.replace(":","-")
        json.dump(self.prediction, open(add_paths(prediction_dir,filename),"w"), indent=3)

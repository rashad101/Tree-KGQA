import json
from tqdm import tqdm
from os.path import join as add_paths
from components.indexer import Indexer
from components.named_entity_recognizer import NER_predictor
from utils.api import API
from utils.miscellaneous import get_fuzzy_match
from utils.metrics import eval
from datetime import datetime


class EntityLinker:
    def __init__(self, args, data):
        self.data = data
        self.args = args
        self.topk = 1
        self.intersect = 0
        self.em = 0
        self.prediction = list()
        self.ner = NER_predictor(self.args)
        if self.args.use_api:
            self.api = API()
        if self.args.use_indexing:
            self.indexer = Indexer()


    def detect_spans(self, single_data, candidate_entity_info, lower=False):
        # step 1: NER mention span detection
        cand_mentions, ner_found = self.ner.get_entity_spans(single_data["question"].lower() if lower else single_data["question"], lower=lower)

        # if span not detected by the NER, then paraphrase the question and perform NER again
        paraphrased = False

        # step 2: candidate entity retrievalx
        if self.args.use_api and self.args.use_indexing:       # entity linking using API call then indexing

            # first apply api call
            if ner_found:                                      # if found entity mention by NER over the original question or the paraphrased one
                for cand_ment in cand_mentions:
                    candidate_entity_info, api_status = self.get_api_result(cand_ment, candidate_entity_info,single_data, paraphrased, lower=lower)

                    # if entity not found using api call, then check in the indexed file
                    if not api_status:
                        candidate_entity_info = self.get_indexed_result(cand_ment, candidate_entity_info,single_data, paraphrased, lower=lower)
            else:
                candidate_entity_info = self.get_indexed_result(single_data["question"], candidate_entity_info, single_data, paraphrased, lower=lower)

        elif self.args.use_indexing:                           # only using indexing method for candidate generation
            candidate_entity_info = self.get_indexed_result(single_data["question"], candidate_entity_info, single_data, paraphrased, lower=lower, force=False)

        elif self.args.use_api:                                # only using API call method for candidate generation
            if ner_found:                                      # if found entity by NER or NER over paraphrased question
                for cand_ment in cand_mentions:
                    candidate_entity_info, _ = self.get_api_result(cand_ment,candidate_entity_info,single_data,paraphrased, lower=lower)

        else:                                                  # if nothing is specified, by default it performs indexing
            candidate_entity_info = self.get_indexed_result(single_data["question"], candidate_entity_info, single_data, paraphrased, lower=lower)

        return candidate_entity_info.copy()


    def perform_EL_single(self, single_data):
        """
        :param single_data: Single data point including the question and other relevant information
        :return: List of dictionary (Linked entity ids with labels)
        """
        candidate_entity_info = {
            "pred_wikiid": list(),
            "pred_wikilabel": list(),
            "mention": list(),
            "mention_boundary": list()
        }
        candidate_entity_info = self.detect_spans(single_data,candidate_entity_info)

        if len(candidate_entity_info["pred_wikiid"])==0:
            candidate_entity_info = self.detect_spans(single_data,candidate_entity_info, lower=True)

        # output template
        pred_output = {
            "pred_wikiid": list(),
            "pred_wikilabel": list(),
            "mention": list(),
            "mention_boundary": list()
        }

        # step 3: entity disambiguation
        if self.args.EL_disamb == "relation":                   # disambiguate using relation information
            pass
        else:                                                   # no disambiguation
            pred_output.update(candidate_entity_info)

        pred_output["pred_wikiid"] = [pd for pd in pred_output["pred_wikiid"] if pd is not None]
        if len(set(pred_output["pred_wikiid"]).intersection(set(single_data["wikidata_id"])))>0:
            self.intersect+=1
        if set(pred_output["pred_wikiid"])==set(single_data["wikidata_id"]):
            self.em+=1
        print("ENT: ",self.em,self.intersect,pred_output["pred_wikiid"],single_data["wikidata_id"])
        return pred_output                                     # returns a dictionary of results with the original data

    def perform_EL(self):
        """perform entity linking over the whole dataset"""
        for d in tqdm(self.data, desc=f"Testing {self.args.dataset}: "):

            if not self.args.true_EL:
                pred_result = self.perform_EL_single(d)
                save_result = d.copy()
                save_result.update(pred_result)
                self.prediction.append(save_result)
            else:
                pass
                # return reference entities

    def get_api_result(self, cand_ment, candidate_entity_info, single_data, paraphrased,lower=False):
        api_result = self.api.fetch_entity(cand_ment)
        if "search" in api_result:  # if API call returns any result
            if len(api_result["search"]) > 0:  # handle cases when API call returns result but an empty one
                wikilabel = api_result["search"][0]["label"]
                wikiid = api_result["search"][0]["id"]
                question = single_data["question"].lower() if lower else single_data["question"]
                # identify mention boundary
                if paraphrased:
                    # perform fuzzy match to get the paraphrased mention span from the original question
                    score, mention = get_fuzzy_match(cand_ment, question)     # look for the entity boundary in the original question
                    boundary = [question.find(mention), question.find(mention) + len(mention)]
                else:
                    mention = cand_ment
                    boundary = [question.find(cand_ment), question.find(cand_ment) + len(cand_ment)]

                candidate_entity_info["pred_wikiid"].append(wikiid)
                candidate_entity_info["pred_wikilabel"].append(wikilabel)
                candidate_entity_info["mention"].append(mention)
                candidate_entity_info["mention_boundary"].append(boundary)
            else:
                return candidate_entity_info, False
        else:
            return candidate_entity_info, False
        return candidate_entity_info, True


    def get_indexed_result(self, cand_ment, candidate_entity_info, single_data, paraphrased, lower=False, force=False):

        indexed_labels, indexed_ids = self.indexer.lookup(cand_ment, topk=self.topk)
        question  = single_data["question"].lower() if lower else single_data["question"]
        found_ya = False
        # identify mention boundary
        if paraphrased:
            # perform fuzzy match to get the paraphrased mention span from the original question
            if cand_ment==single_data["question"]:
                if single_data["original_question"].lower().find(indexed_labels[0].lower()) != -1:
                    mention = indexed_labels[0].lower()
                    boundary = [question.lower().find(mention),
                                question.lower().find(mention) + len(mention)]
                    found_ya = True
            else:
                score,mention = get_fuzzy_match(cand_ment, question)  # look for the entity boundary in the original question
                boundary = [question.find(mention),
                            question.find(mention) + len(mention)]

            candidate_entity_info["pred_wikiid"].append(indexed_ids[0])
            candidate_entity_info["pred_wikilabel"].append(indexed_labels[0])
            candidate_entity_info["mention"].append(mention)
            candidate_entity_info["mention_boundary"].append(boundary)
        else:
            if single_data["original_question"].lower().find(indexed_labels[0].lower())!=-1:
                mention = indexed_labels[0].lower()
                boundary = [question.lower().find(mention),
                            question.lower().find(mention) + len(mention)]
                found_ya = True
                candidate_entity_info["pred_wikiid"].append(indexed_ids[0])
                candidate_entity_info["pred_wikilabel"].append(indexed_labels[0])
                candidate_entity_info["mention"].append(mention)
                candidate_entity_info["mention_boundary"].append(boundary)

        if not found_ya and force:
            mention = indexed_labels[0].lower()

            score, mention = get_fuzzy_match(mention.lower(),
                                             question.lower())  # look for the entity boundary in the original question

            boundary = [question.lower().find(mention),
                        question.lower().find(mention) + len(mention)]
            candidate_entity_info["pred_wikiid"].append(indexed_ids[0])
            candidate_entity_info["pred_wikilabel"].append(indexed_labels[0])
            candidate_entity_info["mention"].append(mention)
            candidate_entity_info["mention_boundary"].append(boundary)

        return candidate_entity_info


    def evaluate(self):
        """evaluates the predicted outputs"""
        eval(dataset=self.args.dataset, predictions=self.prediction, task=self.args.task)


    def save_predictions(self):
        """save prediction in a file"""
        prediction_dir = f"outputs/{'ablation' if self.args.ablation else 'EL'}/"
        dt_fmt = datetime.now().strftime("%d-%b-%Y_%H:%M:%S.%f")
        filename = f"EL_{self.args.dataset}_RL-{self.args.RL_method}_ELdisamb-{self.args.EL_disamb}_API-{'yes' if self.args.use_api else 'no'}_Indexing-{'yes' if self.args.use_indexing else 'no'}_paraphrase-{'yes' if self.args.paraphrase_q else 'no'}_uncased-{'yes' if self.args.uncased else 'no'}_KG-{self.args.kg}_{dt_fmt}.json"
        filename = filename.replace(":","-")
        json.dump(self.prediction, open(add_paths(prediction_dir, filename),"w"), indent=3)

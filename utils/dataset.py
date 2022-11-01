import json,re
from tqdm import tqdm
from utils.api import API
from utils.KGQA_Exception import DatasetNotFoundException


class Dataset:
    def __init__(self, args, dataset_name: str):
        self.args = args
        self.api = API()

        if dataset_name == "webqsp":
            self.data = self.load_webqsp()
        elif dataset_name == "lcquad-2.0":
            self.data = self.load_lcquad_v2()
        elif dataset_name == "lcquad-2.0-kbpearl":
            self.data = self.load_lcquad_v2_kbpearl()
        elif dataset_name == "qald-7":
            self.data = self.load_qald7()
        elif dataset_name == "webqsp-wd":
            self.data = self.load_webqsp_wd()
        else:
            raise DatasetNotFoundException(dataset=dataset_name)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_webqsp(self):
        d = json.load(open("data/webqsp/webqsp_test.json"))
        webqsp = list()
        for item in tqdm(d, desc="Reading WebQSP: "):
            if type(item["text"])==str and len(item["text"])>0:
                webqsp.append({
                    "question": self.process_question(item["text"], lower=True if self.args.uncased else False),
                    "original_question": item["text"],
                    "wikidata_id": item["wikidata_id"],
                    "wikidata_entity": item["entity"],
                    "mentions": [item["text"][boundary[0]:boundary[1]] for boundary in item["mentions"]],
                    "mentions_boundary": item["mentions"]
                })
        return webqsp

    def get_ids(self,query):
        query = query.replace("?", "").replace(".", "").replace("{", "").replace("}", "").replace("(", "").replace(")","")
        _ents = list()
        for w in query.split():
            if ":Q" in w:
                _ents.append(w.split(":")[1].strip())
        return [ent for ent in _ents if ent is not None]

    def load_lcquad_v2(self):
        d = json.load(open("data/lcquad-2.0/lcquad2_test.json"))
        lcquad2 = list()
        for item in tqdm(d, desc="Reading LcQuAD 2.0: "):
            if type(item["question"])==str and len(item["question"])>0:
                if item["question"] != "n/a" and item["question"]!="gsdfhgdfh" and item["question"]!="na":
                    wikidata_id = self.get_ids(item["sparql_wikidata"])
                    lcquad2.append({
                        "question": self.process_question(item["question"], lower=True if self.args.uncased else False).replace("{","").replace("}",""),
                        "original_question": item["question"],
                        "wikidata_id": wikidata_id,
                        "sparql_wikidata": item["sparql_wikidata"]
                    })
        return lcquad2

    def load_lcquad_v2_kbpearl(self):
        dt = json.load(open('data/lcquad-2.0-kbpearl/lcquad2_ground_truth_final.json'))
        data_idx = [aline.strip() for aline in open("data/lcquad-2.0-kbpearl/lcquad2.0_gt_id").readlines()]
        lcquad2_kbpearl = list()
        for idx in tqdm(data_idx, desc="Reading LcQuAD v2.0 KB-pearl: "):
            item = dt[idx]
            if type(item["text"])==str and len(item["text"])>0:
                if item["text"]!="n/a"  and item["text"]!="gsdfhgdfh" and item["text"]!="na":
                    lcquad2_kbpearl.append({
                        "question": self.process_question(item["text"], lower=True if self.args.uncased else False).replace("{","").replace("}",""),
                        "original_question": item["text"],
                        "wikidata_id": [anid.replace("}","").replace("{","") for anid in item["result"]["Wikidata"]["entities"]],
                        "wikidata_relation_id": item["result"]["Wikidata"]["relations"]
                    })
        return lcquad2_kbpearl


    def get_qald7_ids(self, qd):
        entlist = list()
        if "results" in qd[0]:
            for an_ent in qd[0]["results"]["bindings"]:
                if "uri" in an_ent:
                    lab = an_ent["uri"]["value"]
                    lab = lab[lab.rfind("/")+1:]
                    entlist.append(lab)
        return entlist

    def load_qald7(self):
        dt = json.load(open('data/qald-7/qald-7-test-en-wikidata.json'))["questions"]
        qald7 = list()
        for item in (tqdm(dt, desc="Reading QALD-7: ")):
            try:
                qald7.append({
                    "question": self.process_question(item["question"][0]["string"], lower=True if self.args.uncased else False).replace("{","").replace("}",""),
                    "original_question": item["question"][0]["string"],
                    "wikidata_id": item["wikidata_id"],
                    "wikidata_objects": self.api.execute_query(item["query"]["sparql"])
                })
            except Exception as es:
                print(item)
                print(es)

        return qald7

    def load_webqsp_wd(self):
        dt = json.load(open('data/webqsp/webqsp-wd.json'))
        webqsp_wd = list()
        for item in (tqdm(dt, desc="Reading WebQSP_WD: ")):
            ids = list()
            mentions = list()
            for linkedid in item["entities"]:
                labs = list()
                for alink in linkedid["linkings"]:
                    ids.append(alink[0])
                    labs.append(alink[1])

                q_toks = item["utterance"].split()
                for a_mention in linkedid["token_ids"]:
                    try:
                        mentions.append(q_toks[a_mention])
                    except:
                        for m in labs:
                            mentions.append(m.lower())

            webqsp_wd.append({
                "question": self.process_question(item["utterance"], lower=True if self.args.uncased else False).replace("{","").replace("}",""),
                "original_question": item["utterance"],
                "wikidata_id": ids,
                "wikidata_objects": item["answers"],
                "original_mentions": mentions
            })
        return webqsp_wd


    def process_question(self, text, lower=True):
        """
        :param text: question
        :return: modified question with trailing punctuation mark removed and lower-cased
        """
        if text.endswith("?q"):
            text = text.replace("?q","")

        if text.endswith(".") or text.endswith("?"):
            return text[:len(text) - 1].strip().lower() if lower else text[:len(text) - 1].strip()
        else:
            return text.strip().lower() if lower else text.strip()
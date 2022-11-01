import numpy as np
import heapq
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataPropertiesMatcher:
    def __init__(self, all_relations, word_vectors=None, args=None, onehop=None):
        if args.RL_method=="sentencetransformer":
            self.encoder = word_vectors
            self.dim = 768
            self.method = "transformer"
        else:
            self.method = "fasttext"
            self.de_wvec = word_vectors
            self.dim = 300

        self.onehop = onehop
        self.wiki_properties = all_relations
        self.id2r = {r[0]: r[1] for r in self.wiki_properties}
        self.r2id = {r[1]: r[0] for r in self.wiki_properties}
        self.idx2propkey = defaultdict()
        self.property_vecs = np.zeros((len(self.wiki_properties), self.dim))  # keep a matrix of entity vectors
        for j, (p, l) in enumerate(self.wiki_properties):
            self.property_vecs[j] = self.get_chunk_vector(l)
            self.idx2propkey[j] = p
        self.property2idx = {v: k for k, v in self.idx2propkey.items()}  # property to index mapping

    def get_property2idx(self, property_id):
        """
        get index given a property/relation id
        :param property_id: string relation wikidata property starting with P
        :return: index
        """
        try:
            return self.property2idx[property_id]
        except KeyError:
            return 0

    def get_most_probable_property(self, query, top_k=20, entity_label=None, remove_entity=False, entity_id=None,method="fasttext"):
        """
        get the most probable relation given a query
        :param query: question
        :param top_k: top 10 properties
        :param entity_label: label of the entity in the question
        :param remove_entity: remove entity from the query
        :param entity_id: wikidata id of the incoming entity
        :return:
        """
        if remove_entity:
            if entity_label:
                query = query.replace(entity_label, '')
        property_mask = np.zeros(len(self.wiki_properties))
        if entity_id:  # get 1-hop relations in case the entity id is given
            one_hop_property = self.get_relations(entity_id)
        entity_properties = [self.get_property2idx(p) for p in list(one_hop_property)]
        property_mask[entity_properties] = 1.0
        query_vec = np.expand_dims(self.get_chunk_vector(query, method=method), 0)
        cosine_sim = self.get_cosine_similarity(self.property_vecs, query_vec)
        property_scores = [p[0] for p in cosine_sim]  # get into a single list
        property_scores = np.array(property_scores)*property_mask
        best_matches = heapq.nlargest(min(top_k,20), range(len(property_scores)), property_scores.__getitem__)
        top_k_properties = [self.wiki_properties[b_m] for b_m in best_matches]  # get top k matches
        return top_k_properties

    def get_candidate_properties(self, query, top_k=25, entity_properties=list()):
        """
        get the most probable relation given a query
        :param query: question
        :param top_k: top 10 properties
        :param entity_label: label of the entity in the question
        :param remove_entity: remove entity from the query
        :param entity_id: wikidata id of the incoming entity
        :return:
        """
        property_mask = np.zeros(len(self.wiki_properties))
        entity_properties = [self.get_property2idx(p) for p in entity_properties]
        property_mask[entity_properties] = 1.0
        query_vec = np.expand_dims(self.get_chunk_vector(query), 0)
        # property_mask = np.expand_dims(property_mask, 1)
        cosine_sim = self.get_cosine_similarity(self.property_vecs, query_vec)
        # property_scores = np.sort(cosine_sim)
        property_scores = [p[0] for p in cosine_sim]  # get into a single list
        property_scores = np.array(property_scores)*property_mask
        best_matches = heapq.nlargest(min(top_k,len(entity_properties)), range(len(property_scores)), property_scores.__getitem__)
        top_k_properties = [self.wiki_properties[b_m] for b_m in best_matches]  # get top k matches
        return top_k_properties

    def get_chunk_vector(self, chunk):
        """
        Returns a vector for each chunk
        :param chunk: tokens for a chunk
        :return: an averaged vector
        """
        if self.method=="fasttext":
            chunks = chunk.lower().split()  # split the chunk into words
            chunk_vector = []
            for t in chunks:
                t_vec = self.de_wvec.get_vector(t).reshape(1,-1)  # Get the vector for each word
                chunk_vector.append(t_vec)
            return np.average(chunk_vector, 0)
        else:
            return self.encoder.encode([chunk])


    def get_top1(self, query, cands, classifier):
        """ get top 1 relation from top10 relations"""
        results = classifier.classify(sequence=query, candidate_labels=cands)
        toprel = results['labels'][0]
        toprel_score = results["scores"][0]

        return {
            "toprel": toprel,
            "toprel_id": self.r2id[toprel],
            "toprel_score": toprel_score,
            "rels_with_scores": {lb: results["scores"][i] for i, lb in enumerate(results['labels'])}
        }

    def predict_rel(self,data, classifier):
        temp_p = []
        qu = data["question"]
        ent_labs = [data["original_question"][m[0]:m[1]] for m in data["mention_boundary"]] # [start_idx, end_idx] entity mention boundary
        for elab in ent_labs:
            qu = qu.replace(elab, "")

        for elab in data["mention"]:
            qu = qu.replace(elab,"")
        tot = 0

        for entid in data["pred_wikiid"]:
            cand_rel = set()
            if entid not in self.onehop:
                print(f"{entid}  not found in the file")
            else:
                if len(self.onehop[entid]) == 0:
                    print(f"No relations for {entid}")
                else:
                    for kk, vv in self.onehop[entid].items():
                        if vv in self.id2r:
                            cand_rel.add(self.id2r[vv])
                    tot += len(self.onehop[entid])

            if len(data["pred_wikiid"])>0:
                if len(cand_rel) < 40 and len(cand_rel) > 0:
                    result = self.get_top1(qu, cands=list(cand_rel), classifier=classifier)
                elif len(cand_rel) > 0:
                    tt = self.get_candidate_properties(query=qu, top_k=40, entity_properties=list(cand_rel))
                    result = self.get_top1(qu, cands=[r[1] for r in tt], classifier=classifier)
                else:
                    tt = self.get_candidate_properties(query=qu, top_k=13, entity_properties=list(self.id2r.values()))
                    result = self.get_top1(qu, cands=[r[1] for r in tt], classifier=classifier)

                temp_p.append(result["toprel_id"])

        return list(set(temp_p))

    def get_relations(self, entity_id, direct=False):
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setQuery(
            "SELECT ?item WHERE { wd:" + entity_id + "?item ?obj. SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'.}}")
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        relations = [r['item']['value'].split('/')[-1] for r in results['results']['bindings'] if
                     'direct/P' in r['item']['value']]
        return set(relations)

    @staticmethod
    def get_cosine_similarity(u, v):
        """
        cosine similarity between a matrix and a vector
        :param u: a matrix of size B X N
        :param v: a vector of size 1 X N
        :return: cosine similarity
        """
        u = np.expand_dims(u, 1)
        n = np.sum(u * v, axis=2)
        d = np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=1)
        return n / d

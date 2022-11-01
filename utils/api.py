from time import sleep
import requests
from qwikidata.sparql import return_sparql_query_results

class API:

    @staticmethod
    def fetch_entity(entity):
        """
        :param entity: predicted entity
        :return: API call results related to input entity
        """
        sleep(1)
        url_template = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&continue=0&search=ENTITY"
        wiki_ent_search_url = url_template.replace("ENTITY", '%20'.join(entity.split()))
        fetched_data = requests.get(url=wiki_ent_search_url).json()
        return fetched_data

    @staticmethod
    def get_relations(entity_id, direct=False):
        """
        :param entity_id: WIKIDATA entity ID
        :param direct:
        :return: Set of connected relations to given entity_id
        """
        sleep(1)
        sparql_query = "SELECT ?item WHERE { wd:" + entity_id + "?item ?obj. SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'.}}"
        results = return_sparql_query_results(sparql_query)
        relations = [r['item']['value'].split('/')[-1] for r in results['results']['bindings'] if
                     'direct/P' in r['item']['value']]
        return list(set(relations))


    def filter_triples(self,results):
        """only extract dicrect entities and relations"""
        filtered_results = list()
        for res in results:
            if "direct/P" in res["pred"]["value"] and "entity/Q" in res["obj"]["value"]:
                pred = res["pred"]["value"]
                pred = pred[pred.find("direct/P") + len("direct/"):]
                # if pred == "P31":
                #     continue
                obj = res["obj"]["value"]
                obj = obj[obj.rfind("/")+1:]
                obj = obj[obj.find("entity/Q") + len("entity/"):]
                #print("OBJ: ", obj)
                obj_lab = res["objLabel"]["value"]
                #print(f"{pred} {obj}:{obj_lab}")
                filtered_results.append((pred, obj, obj_lab))
        return filtered_results


    def get_connected_comps(self, entity_id):
        "get connected nodes and realtions"
        try:
            sleep(2)
            sparql_query = "Select distinct ?pred ?obj ?predLabel ?objLabel where { wd:" + entity_id + " ?pred ?obj FILTER (CONTAINS(str(?pred),'wikidata')) SERVICE wikibase:label {bd:serviceParam wikibase:language 'en' .}}"
            results = return_sparql_query_results(sparql_query)
            results = self.filter_triples(results["results"]["bindings"])
        except:
            print(f"ERROR: one hop info not found for: {entity_id}")
            results = {}
        ent_info, con_comps = {entity_id:{}}, {entity_id:{}}

        for res in results:
            p,o,o_l = res
            ent_info[entity_id][o] = o_l
            con_comps[entity_id][o] = p

        return ent_info.copy(), con_comps.copy()

    def extract_ids(self, results):
        ids = list()
        for res in results:
            if "uri" in res:
                val = res["uri"]["value"]
                val = val[val.rfind("/")+1:]
                ids.append(val)
        return ids

    def execute_query(self,query):
        results = return_sparql_query_results(query)
        results = self.extract_ids(results["results"]["bindings"])
        return results
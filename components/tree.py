import json
from utils.miscellaneous import generate_ngrams, get_cosine_similarity, get_one_hop_rels
from sklearn.metrics.pairwise import cosine_similarity
from utils.api import API

class Tree:
    """
    This class represents a tree, where the root node is a predicted entity.
    This tree spans till k-hop (2 is the default value). Nodes represents
    entities and edge represent a relation between two connecting
    entities (subject, object). Each node holds a state. A state contains 4 state variables:
        "max_score": maximum similarity score between the given text chunk and connected relations
        "max_text_chunk": the text chunk for which the max_score is achieved,
        "max_edge": the relation for which the max_score is achieved
        "max_edge_label": label of the max_edge (relation label)
    structure of the entity info looks like below:
    {
        "entity_id": {
            "id_231": "rel_231",
            "id_232": "rel_231",        # we put entity_ids as the keys as a relation might appear multiple time in a particular hop
            "id_233": "rel_233"
        }
    }
    """
    def __init__(self, args=None, vec=None, root_node_id=None, n_hop=2, kge=None, onehop=None, rels=None, ent_map = None):
        super(Tree, self).__init__()

        self.args = args
        self.states = dict()
        self.root = root_node_id
        self.api = API()
        self.vec = vec
        self.kge = kge
        self.n_nop = n_hop
        self.data = onehop
        self.rel = rels
        self.rel = {r[0]:r[1] for r in self.rel}

        self.entity_mapping = ent_map
        self.tree = dict()

        # creates the tree hop-wise, with default state values for each of the nodes in the tree
        self.build_tree(n=1, nodeid=self.root, parent_node=None, max_n=n_hop)  # n -> current level/hop in the tree


    def build_tree(self, n=1, nodeid=None, parent_node=None, max_n = 2):
        """Build n-level (n-hop) tree recursively"""
        if n<=max_n:
            # creates the hop if its already not there
            if f"hop-{n}" not in self.tree:
                self.tree[f"hop-{n}"] = dict()

            # fetch one-hop info by api call if the entity is not in the data already and update the data
            if nodeid not in self.data:
                self.update_data(nodeid)

            # creating the state of the current node and storing the connected nodes and relations
            self.tree[f"hop-{n}"][nodeid] = {
                "connections": {k:v for k,v in self.data[nodeid].copy().items() if k!=parent_node},
                "state": {
                    "max_score": -1,
                    "max_text_chunk": None,
                    "max_edge": None,   # max scoring connected edge
                    "max_edge_label": None
                }
            }
        else:
            return
        # populating next hop of the tree
        for leaf in self.data[nodeid].keys():
            if leaf!=parent_node:           # getting rid of cycle
                self.build_tree(n=n+1, nodeid=leaf, parent_node=nodeid, max_n=max_n)


    def update_tree(self, running_text, n=1, max_n=2, use_kg_emb=False):

        if n<=max_n:
            # perform update operation in the current hop
            for node, connections in self.tree[f"hop-{n}"].items():
                for next_node, edge in connections["connections"].items():
                    #print(connections)
                    max_score = self.tree[f"hop-{n}"][node]["state"]["max_score"]
                    if edge in self.rel:  # skip the edges that are not in the whole relation list
                        if not use_kg_emb: # use fasttext/wikipedia2vec embedding
                            if self.args.vec=="sentencetransformer":
                                current_score = cosine_similarity(self.vec.encode([running_text]).reshape(1,-1), self.vec.encode([self.rel[edge]]).reshape(1,-1)).tolist()[0][0]
                            else:
                                current_score = cosine_similarity(self.vec.get_vector(running_text).reshape(1,-1), self.vec.get_vector(self.rel[edge]).reshape(1,-1)).tolist()[0][0]
                        else:              # use knowledge graph embedding based similarity
                            current_score = self.kge_similarity(running_text, self.rel[edge])

                        # update the state of the parent node
                        if current_score>max_score:
                            self.tree[f"hop-{n}"][node]["state"]["max_score"] = current_score
                            self.tree[f"hop-{n}"][node]["state"]["max_text_chunk"] = running_text
                            self.tree[f"hop-{n}"][node]["state"]["max_edge"] = edge
                            self.tree[f"hop-{n}"][node]["state"]["max_edge_label"] = self.rel[edge]

            # pass the running text to the next hop
            self.update_tree(running_text, n=n+1, max_n=self.n_nop)
        else:
            return

    def update_state(self, node_id, new_value=None):
        """
        Updates the state of a node
        :param node_id: Entity ID of a node
        :param new_value: new similarity score with the chunk
        """
        for n in range(self.n_nop):
            if node_id in self.tree[f"hop-{n + 1}"]:
                if new_value["score"]>self.tree[f"hop-{n + 1}"][node_id]["state"]["score"]:
                    self.tree[f"hop-{n + 1}"][node_id]["state"].update(new_value)

    def get_state(self, node_id):
        """
        :param node_id: Entity ID of a node
        :return: state of that node
        """
        for n in range(self.n_nop):
            if node_id in self.tree[f"hop-{n+1}"]:
                return self.tree[f"hop-{n+1}"][node_id]["state"]
        return None

    def update_data(self,nodeid):
        ent_info, con_comps = {nodeid: {}}, {nodeid: {}}
        try:
            ent_info, con_comps = self.api.get_connected_comps(nodeid)
            self.data.update(con_comps)
            self.entity_mapping.update(ent_info)
            self.save_connected_info()
        except Exception as e:
            print(f"ERROR: 1-hop info for {nodeid} could not be fetched")
            self.data.update(con_comps)
            self.entity_mapping.update(ent_info)
            self.save_connected_info()

    def save_connected_info(self):
        json.dump(self.data,open("data/wikidata/onehop_comps.json","w"),indent=3)
        json.dump(self.entity_mapping, open("data/wikidata/id2ent_mapping.json", "w"), indent=3)

    def get_max_edge(self, max_hop=1):
        # if two of the hops have the same max edge, take the lowest hop edge
        # take all the edges with the same
        level, relation_label, relation_id, max_relation_score = 1, None, None, -1
        object_nodes, subject_node = list(), None
        for n in range(self.n_nop):
            #get results till max_hop
            if n+1<=max_hop:    # because of this its possible to only get results till k=max_hop, set max_hop=1 is you want to get the result till hop-1
                current_level, current_relation_label, current_relation_id,current_max_relation_score = n+1, None, None,-1
                current_object_nodes, current_subject_node = list(), None

                # get max scoring edge of the current hop
                hop_max_rel, hop_max_rel_score = None, -1
                for nn in self.tree[f"hop-{n + 1}"].keys():
                    curr_sc = self.tree[f"hop-{n+1}"][nn]["state"]["max_score"]
                    if curr_sc>hop_max_rel_score:
                        hop_max_rel_score=curr_sc
                        hop_max_rel=self.tree[f"hop-{n+1}"][nn]["state"]["max_edge"]

                for node_id,conns in self.tree[f"hop-{n + 1}"].items():
                    node_id_score_max = self.tree[f"hop-{n+1}"][node_id]["state"]["max_score"]
                    node_id_edge_max = self.tree[f"hop-{n+1}"][node_id]["state"]["max_edge"]
                    node_id_edge_max_label = self.tree[f"hop-{n+1}"][node_id]["state"]["max_edge_label"]

                    for next_node, edge in conns["connections"].items():
                        if edge==hop_max_rel:
                            current_level = n+1
                            current_relation_label = node_id_edge_max_label
                            current_relation_id = node_id_edge_max
                            current_max_relation_score = node_id_score_max
                            current_object_nodes.append(next_node)
                            current_subject_node = node_id
                if current_max_relation_score>max_relation_score:
                    level = current_level
                    relation_label = current_relation_label
                    relation_id = current_relation_id
                    max_relation_score = current_max_relation_score
                    object_nodes = current_object_nodes
                    subject_node = current_subject_node

        return {
            "hop": level,
            "relation_label": relation_label,
            "relation_id": relation_id,
            "score": max_relation_score,
            "object_entities": object_nodes,
            "subject": subject_node
        }


    def perform_tree_walk(self, question, n_pass=False, use_kg_emb=False):
        """
        :param question:
        :param relation: if True use the KGE embedding for capturing the context better. (set n_pass=False if you put relation=True
        :return: Update states of all the nodes
        """
        if n_pass:
            ngram = generate_ngrams(question)
            for a_gram in ngram:
                self.update_tree(a_gram, n=1, max_n=self.n_nop)
        else:
            self.update_tree(question, n=1, max_n=self.n_nop, use_kg_emb= use_kg_emb)


    def kge_similarity(self, candidate_relation, kg_relation):
        """ Knowledge Graph embedding based similarity"""
        return cosine_similarity(self.kge[candidate_relation].reshape(1,-1),self.kge[kg_relation].reshape(1,-1)).tolist()[0][0]



if __name__=="__main__":
    # dummy tree: after building the tree from the data
    # tree= {
    #     "hop-01": {
    #         "root_id": {
    #             "id_111": "rel_111",
    #             "id_112": "rel_112",
    #             "id_113": "rel_113"
    #         }
    #     },
    #     "hop-02": {
    #         "id_111": {
    #             "id_211": "rel_211",
    #             "id_212": "rel_212",
    #             "id_213": "rel_213"
    #         },
    #         "id_112": {
    #             "id_221": "rel_221",
    #             "id_222": "rel_222",
    #             "id_223": "rel_223"
    #         },
    #         "id_113": {
    #             "id_231": "rel_231",
    #             "id_232": "rel_232",
    #             "id_233": "rel_233"
    #         }
    #     }
    # }

    tree = Tree(root_node_id="root_id", n_hop=2)
    print(tree.tree)
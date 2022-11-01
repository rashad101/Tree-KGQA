from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Unsupervised KGQA v1.0")
    parser.add_argument('--task', type=str, default="EL", help="For entity linking 'EL', for question answering 'KGQA' ")
    parser.add_argument('--kg',type=str, required=False, default="wikidata", help="Only wikidata is available for v1.0. In version 2.0 Freebase, DBpedia will be available")
    parser.add_argument('--dataset', type=str, default="webqsp", choices=["webqsp","lcquad-2.0","lcquad-2.0-kbpearl","qald-7","webqsp-wd"], help="Dataset name")
    parser.add_argument('--RL_method', type=str, required=False, default="None", help="None, fuzzy ,kge, fasttext_emb, sentencetransformer ,graph-laplacian")
    parser.add_argument('--vec', type=str, required=False, default="sentencetransformer",choices=["sentencetransformer","fasttext"])
    parser.add_argument('--EL_disamb',type=str, default="None", choices=["None","relation"], help="Entity disambiguation method")
    parser.add_argument('--QAtype', type=str, default="complex", choices=["complex","simple"], help="Dataset type")
    parser.add_argument('--use_api', action='store_true')
    parser.add_argument('--use_indexing', action='store_true')
    parser.add_argument('--true_EL', action='store_true')
    parser.add_argument('--true_RL', action='store_true')
    parser.add_argument('--uncased', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parsed_args = parser.parse_args()
    return parsed_args
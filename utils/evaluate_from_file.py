import json
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Unsupervised KGQA v1.0")
    parser.add_argument('--task', type=str, default="EL", choices=["EL","RL","KGQA"])
    parser.add_argument('--f', type=str)
    parser.add_argument('--true_EL', action='store_true')
    parsed_args = parser.parse_args()
    return parsed_args
args = get_args()


def calculate_scores(pred,gold, eval_type = "entity"):
    pred = [list(set(p)) for p in pred]
    gold = [list(set(g)) for g in gold]
    tpentity = 0
    fpentity = 0
    fnentity = 0
    totalentchunks = 0

    onnocount = 0
    for queryitem, golditem in zip(pred, gold):
        queryentities = queryitem

        if len(set(golditem).intersection(queryentities)) > 0:
            onnocount += 1
        for goldentity in set(golditem):
            totalentchunks += 1
            if goldentity in queryentities:
                tpentity += 1
            else:
                fnentity += 1
        for queryentity in set(queryentities):
            if queryentity not in golditem:
                fpentity += 1

    precisionentity = tpentity / float(tpentity + fpentity)
    recallentity = tpentity / float(tpentity + fnentity)
    f1entity = 2 * (precisionentity * recallentity) / (precisionentity + recallentity)
    print(f"precision {eval_type} = ", precisionentity)
    print(f"recall {eval_type} = ", recallentity)
    print(f"f1 {eval_type} = ", f1entity)
    return {
        "precision": precisionentity,
        "recall": recallentity,
        "f1-score": f1entity
    }

def eval_lcquad(predictions, task="EL"):
    pred, gold = list(), list()
    pred_kgqa, gold_kgqa = list(),list()

    if task=="EL" or task =="KGQA":
        for single_pred in predictions:
            if args.true_EL:
                pred.append(list(set(single_pred["wikidata_id"])))
            else:
                pred.append(list(set(single_pred["pred_wikiid"])))
            gold.append(list(set(single_pred["wikidata_id"])))
        print("Entity Linking: ")
        el_result = calculate_scores(pred, gold)

    if task=="RL":
        for single_pred in predictions:

            pred.append(list(set(single_pred["pred_relations"])))
            gold.append(list(set(single_pred["wikidata_relation_id"])))
        print("Relation Linking: ")
        el_result = calculate_scores(pred, gold)



    if task=="KGQA":
        for single_pred in predictions:
            pred_kgqa.append(list(set(single_pred["pred_objects"])))
            gold_kgqa.append(list(set(single_pred["wikidata_objects"])))
        print("KGQA Scores: ")
        kgqa_result = calculate_scores(pred_kgqa,gold_kgqa,eval_type="KGQA")

    return {"EL": el_result, "KGQA": kgqa_result} if task=="KGQA" else {"EL": el_result}


if __name__ == "__main__":
    data = json.load(open(args.f))
    eval_lcquad(data, task=args.task)
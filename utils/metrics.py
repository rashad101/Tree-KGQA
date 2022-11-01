# partially adopted from https://github.com/facebookresearch/BLINK/blob/master/elq/vcg_utils/measures.py
# and from https://github.com/debayan/pnel/blob/master/eval/lcquad2/judge.py


from argparse import ArgumentParser
import json
import numpy as np
from utils.args import get_args

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

        #print(set(golditem), set(queryentities))
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

def eval_sqv2(predictions):
    pred = list()
    gold = list()
    for single_pred in predictions:
        pred.append(list(set(single_pred["pred_wikiid"])))
        gold.append(list(set(single_pred["wikidata_id"])))
    tpentity = 0
    fpentity = 0
    fnentity = 0
    totalentchunks = 0

    for queryitem, golditem in zip(pred, gold):
        if len(queryitem) == 0:
            continue
        queryentities = queryitem
        for goldentity in golditem:
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
    print("precision entity = ", precisionentity)
    print("recall entity = ", recallentity)
    print("f1 entity = ", f1entity)



def eval_lcquad(predictions, task="EL"):
    pred, gold = list(), list()
    pred_kgqa, gold_kgqa = list(),list()

    for single_pred in predictions:
        if args.true_EL:
            pred.append(list(set(single_pred["wikidata_id"])))
        else:
            pred.append(list(set(single_pred["pred_wikiid"])))
        gold.append(list(set(single_pred["wikidata_id"])))

    print("Entity Linking: ")
    el_result = calculate_scores(pred,gold)

    if task=="KGQA":
        for single_pred in predictions:
            pred_kgqa.append(list(set(single_pred["pred_objects"])))
            gold_kgqa.append(list(set(single_pred["wikidata_objects"])))
        print("KGQA Scores: ")
        kgqa_result = calculate_scores(pred_kgqa,gold_kgqa,eval_type="KGQA")

    return {"EL": el_result, "KGQA": kgqa_result} if task=="KGQA" else {"EL": el_result}


def entity_linking_tp_with_overlap(gold, predicted):
    """
    Counts weak and strong matches
    :param gold:
    :param predicted:
    :return:
    >>> entity_linking_tp_with_overlap([('Q7366', 14, 18), ('Q780394', 19, 35)], [('Q7366', 14, 16), ('Q780394', 19, 35)])
    2, 1
    >>> entity_linking_tp_with_overlap([('Q7366', 14, 18), ('Q780394', 19, 35)], [('Q7366', 14, 16)])
    1, 0
    >>> entity_linking_tp_with_overlap([(None, 14, 18), ('Q780394', 19, 35)], [('Q7366', 14, 16)])
    0, 0
    >>> entity_linking_tp_with_overlap([(None, 14, 18), (None, )], [(None,)])
    1, 0
    >>> entity_linking_tp_with_overlap([('Q7366', ), ('Q780394', )], [('Q7366', 14, 16)])
    1, 0
    >>> entity_linking_tp_with_overlap([], [('Q7366', 14, 16)])
    0, 0
    """
    if not gold or not predicted:
        return 0, 0
    # Add dummy spans, if no spans are given, everything is overlapping per default
    if any(len(e) != 3 for e in gold):
        gold = [(e[0], 0, 1) for e in gold]
        predicted = [(e[0], 0, 1) for e in predicted]
    # Replace None KB ids with empty strings
    gold = [("",) + e[1:] if e[0] is None else e for e in gold]
    predicted = [("",) + e[1:] if e[0] is None else e for e in predicted]

    gold = sorted(gold, key=lambda x: x[2])
    predicted = sorted(predicted, key=lambda x: x[2])

    # tracks weak matches
    lcs_matrix_weak = np.zeros((len(gold), len(predicted)), dtype=np.int16)
    # tracks strong matches
    lcs_matrix_strong = np.zeros((len(gold), len(predicted)), dtype=np.int16)
    for g_i in range(len(gold)):
        for p_i in range(len(predicted)):
            gm = gold[g_i]
            pm = predicted[p_i]

            # increment lcs_matrix_weak
            if not (gm[1] >= pm[2] or pm[1] >= gm[2]) and (gm[0].lower() == pm[0].lower()):
                if g_i == 0 or p_i == 0:
                    lcs_matrix_weak[g_i, p_i] = 1
                else:
                    lcs_matrix_weak[g_i, p_i] = 1 + lcs_matrix_weak[g_i - 1, p_i - 1]
            else:
                if g_i == 0 and p_i == 0:
                    lcs_matrix_weak[g_i, p_i] = 0
                elif g_i == 0 and p_i != 0:
                    lcs_matrix_weak[g_i, p_i] = max(0, lcs_matrix_weak[g_i, p_i - 1])
                elif g_i != 0 and p_i == 0:
                    lcs_matrix_weak[g_i, p_i] = max(lcs_matrix_weak[g_i - 1, p_i], 0)
                elif g_i != 0 and p_i != 0:
                    lcs_matrix_weak[g_i, p_i] = max(lcs_matrix_weak[g_i - 1, p_i], lcs_matrix_weak[g_i, p_i - 1])

            # increment lcs_matrix_strong
            if (gm[1] == pm[1] and pm[2] == gm[2]) and (gm[0].lower() == pm[0].lower()):
                if g_i == 0 or p_i == 0:
                    lcs_matrix_strong[g_i, p_i] = 1
                else:
                    lcs_matrix_strong[g_i, p_i] = 1 + lcs_matrix_strong[g_i - 1, p_i - 1]
            else:
                if g_i == 0 and p_i == 0:
                    lcs_matrix_strong[g_i, p_i] = 0
                elif g_i == 0 and p_i != 0:
                    lcs_matrix_strong[g_i, p_i] = max(0, lcs_matrix_strong[g_i, p_i - 1])
                elif g_i != 0 and p_i == 0:
                    lcs_matrix_strong[g_i, p_i] = max(lcs_matrix_strong[g_i - 1, p_i], 0)
                elif g_i != 0 and p_i != 0:
                    lcs_matrix_strong[g_i, p_i] = max(lcs_matrix_strong[g_i - 1, p_i], lcs_matrix_strong[g_i, p_i - 1])

    weak_match_count = lcs_matrix_weak[len(gold) - 1, len(predicted) - 1]
    strong_match_count = lcs_matrix_strong[len(gold) - 1, len(predicted) - 1]
    return weak_match_count, strong_match_count


def display_metrics(num_correct, num_predicted, num_gold, prefix="",):
    p = 0 if num_predicted == 0 else float(num_correct) / float(num_predicted)
    r = 0 if num_gold == 0 else float(num_correct) / float(num_gold)
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    print("{0}precision = {1} / {2} = {3}".format(prefix, num_correct, num_predicted, p))
    print("{0}recall = {1} / {2} = {3}".format(prefix, num_correct, num_gold, r))
    print("{0}f1 = {1}".format(prefix, f1))


def calc_indices(q, entity_mention):
    if entity_mention is not None:
        try:
            return q.index(entity_mention), q.index(entity_mention)+len(entity_mention)
        except:
            return None, None
    else:
        return None, None


def eval_webqsp(data, task=None):
    num_correct_weak, num_correct_strong, num_predicted, num_gold = 0, 0, 0, 0

    for d in data:
        num_predicted+= sum([ 1 if e is not None else 0 for e in d["pred_wikiid"]])
        pred_ent = d["pred_wikiid"]

        num_gold+= len(d["wikidata_id"])
        gold_ent = d["wikidata_id"]

        gold_info = [(g_e,d["mentions_boundary"][i][0],d["mentions_boundary"][i][1]) for i, g_e in enumerate(gold_ent)]
        pred_info = [(p_e,d["mention_boundary"][i][0],d["mention_boundary"][i][1]) for i, p_e in enumerate(pred_ent)]

        num_overlap_weak, num_overlap_strong = entity_linking_tp_with_overlap(gold_info, pred_info)
        num_correct_weak += num_overlap_weak
        num_correct_strong += num_overlap_strong


    print("Weak Matching:")
    print(num_correct_weak, num_correct_strong)
    display_metrics(num_correct_weak, num_predicted, num_gold)
    print("Strong Matching:")
    display_metrics(num_correct_strong, num_predicted, num_gold)


def eval(dataset=None, predictions=None, task=None):
    if dataset=="webqsp":
        return eval_webqsp(predictions, task=task)
    elif dataset=="sqv2":
        return eval_sqv2(predictions,task=task)
    elif dataset=="lcquad-2.0-kbpearl" or dataset=="lcquad-2.0" or dataset =="qald-7" or dataset=="t-rex" or dataset=="webqsp-wd":
        return eval_lcquad(predictions, task=task)


if __name__== "__main__":
    args = get_args()
    if args.f!="":
        data = json.load(open(args.f))
        eval(dataset=args.dataset, predictions=data, task=args.task)
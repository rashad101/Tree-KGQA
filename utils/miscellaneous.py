import numpy as np
import json
from utils.api import API
from fuzzywuzzy import fuzz

onehoprel = json.load(open("data/wikidata/onehoprels.json"))
api_util = API()


def get_cosine_similarity(u,v):
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


def generate_ngrams(text):
    """ generates n-gram chunks given a text"""
    words_list = text.split()
    n = len(words_list)
    ngrams_list = []

    for num in range(0, len(words_list)):
        for l in range(n):
            ngram = ' '.join(words_list[num:num + l])
            ngrams_list.append(ngram)
    return ngrams_list


def get_match(w2vec, chunk, question):
    ngrams = generate_ngrams(question)
    q_vec = get_chunk_vector(w2vec,question)
    matched_chunk = ""
    sc = 0
    for agram in ngrams:
        score = get_cosine_similarity(get_chunk_vector(w2vec, agram), q_vec)
        if score > sc :
            matched_chunk=agram
            sc = score
    return matched_chunk


def get_one_hop_rels(entity_id):
    """
    :param entity_id: entity id
    :return: one hop connected relation
    """
    if entity_id in onehoprel:
        return onehoprel[entity_id]

    # get one hop relation with API call and update the file
    onehop_rel = api_util.get_relations(entity_id=entity_id)
    if onehop_rel:
        onehoprel[entity_id] = onehop_rel
        json.dump(onehoprel, open("data/wikidata/onehoprels.json","w"), indent=3)
    return onehop_rel


def get_chunk_vector(wvec, chunk):
    """
    Returns a vector for each chunk
    :param chunk: tokens for a chunk
    :return: an averaged vector
    """
    chunks = chunk.lower().split()  # split the chunk into words
    chunk_vector = []
    for t in chunks:
        t_vec = wvec.get_vector(t)  # Get the vector for each word
        chunk_vector.append(t_vec)
    return np.average(chunk_vector, 0)


def get_fuzzy_match(object, answer, threshold=80):
    """get phrase with highest match in answer"""
    answer_phrase = generate_ngrams(answer)
    if answer_phrase:
        best_match = [fuzz.ratio(object, phr) for phr in answer_phrase]
        return np.max(best_match), answer_phrase[np.argmax(best_match)]
    else:
        return 0, ''
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import numpy as np
from fuzzywuzzy import fuzz

class NER_predictor:
    def __init__(self, args):
        self.device = 0 if args.use_gpu else -1
        if args.uncased:
            self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
            self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
            self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
        self.tokenizer_lower = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        self.model_lower = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
        self.lower_nlp = pipeline("ner", model=self.model_lower, tokenizer=self.tokenizer_lower, device=self.device)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=self.device)

    @staticmethod
    def process_perdiction(result):
        """
        :param result: predicted entity (tokens)
        :return: contatenated longest entity spans
        """
        ent = []
        running_ent = ""

        for r in result:
            if r["word"].startswith("##"):
                running_ent += r["word"].replace("##", "")
            else:
                if running_ent != "":
                    ent.append(running_ent)
                    running_ent = r["word"]
                else:
                    running_ent = r["word"]
            #print(r["word"],running_ent, ent)
        if len(ent) > 0 and ent[-1] != running_ent:
            ent.append(running_ent)
        elif running_ent != "":
            ent.append(running_ent)
        return ent

    @staticmethod
    def eliminate_duplicates(candidates):
        """
        :param candidates:
        :return: non-overlapping candidates
        """
        final_cands = np.ones(len(candidates))
        for n in range(len(candidates)):
            for i, cand in enumerate(candidates):
                for j, mtc in enumerate(candidates):
                    if mtc.find(cand) != -1 and i != j and final_cands[i]:
                        final_cands[i] = 0
        return [candidates[i] for i, cand in enumerate(final_cands) if cand]


    @staticmethod
    def generate_ngrams(s, n=range(8)):
        """ generates n-gram chunks given a sentence/question"""
        words_list = s.split()
        ngrams_list = []

        for num in range(0, len(words_list)):
            for l in n:
                ngram = ' '.join(words_list[num:num + l])
                ngrams_list.append(ngram)
        return ngrams_list

    def get_spans(self, q, results):
        """
            why: this is required to find the longest NER span because sometimes the predicted NERs
                 appears side by side in the question but predicted seperately.
            what it does: creates a sequence by concatenating NER chunks and then generates n-gram chunks;
            returns the n-gram chunks of NERs present in the question
        """
        ss,spans = self.next_token_merge_check(q, results)
        #print("step 1: ",spans)
        for i in range(50):
            m_status,spans = self.next_token_merge_check(q, spans)

        #print("step 2: ", spans)
        for i in range(len(q.split())):
            spans = self.fuzzyfy(spans, question=q)

        for i in range(50):
            m_status, spans = self.next_token_merge_check(q, spans)

        return spans

    def get_fuzzy_match(self,answer, sentence, threshold=80):
        """get phrase with highest match in answer"""
        answer_phrase = self.generate_ngrams(sentence)
        if answer_phrase:
            best_match = [fuzz.ratio(answer, phr) for phr in answer_phrase]
            return answer_phrase[np.argmax(best_match)]
        else:
            return ''

    def fuzzyfy(self,temp_pred, question):
        newlist = [self.get_fuzzy_match(ped, question) for ped in temp_pred]
        temp_newlist = list()
        for anitem in newlist:
            if anitem not in temp_newlist:
                temp_newlist.append(anitem)
        newlist = self.eliminate_duplicates(temp_newlist)
        return newlist


    def sanity_check(self,oldlist, newlist):
        finallist = []
        for item in newlist:
            ok = False
            for oitem in oldlist:
                if oitem in item:
                    ok = True
                    break
            if ok:
                finallist.append(item)
        return finallist


    def predict_NER_spans(self, processed_result, ques):
        """
        :param ques:
        :param processed_result:
        :return:
        """
        spanned = self.get_spans(ques, processed_result)  # check for largest NER spans
        predicted_spans = self.eliminate_duplicates(spanned)  # remove overlapping spans
        return (predicted_spans, True) if len(predicted_spans) > 0 else ([], False)


    def next_token_merge_check(self, text, tokens):
        tokens = [tok for tok in tokens if tok is not None]
        if len(tokens) <= 1:
            return False, tokens

        new_tok = list()
        found = False
        idx = 0
        while True:
            if idx == len(tokens) - 1:
                new_tok.append(tokens[idx])
                break
            if idx > len(tokens) - 1:
                break
            if idx < len(tokens) - 1:
                if text.find(tokens[idx] + " " + tokens[idx + 1]) != -1:
                    new_tok.append(tokens[idx] + " " + tokens[idx + 1])
                    found = True
                    idx += 1
                else:
                    new_tok.append(tokens[idx])
            else:
                new_tok.append(tokens[idx])
            idx += 1

        if not found:
            return False, tokens
        return True, self.eliminate_duplicates(new_tok)

    def is_numbers(self, atoken):
        for a_char in "abcdefghijklmnopqrstuvwxyz":
            if a_char in atoken:
                return False
        return True

    def get_clean_token(self,tokens):
        return [t["word"].replace("##","") for t in tokens]

    def get_entity_spans(self, question, lower=False):
        WHwords = ["Who", "What", "When", "Which", "Who", "How"]

        if not lower:
            spans = self.nlp(question)  # getting NER tokens (sometime contains ## in split parts)
        else:
            spans = self.lower_nlp(question)

        clean_tokens = self.get_clean_token(spans)
        processed_result = self.process_perdiction(spans)  # getting processed and final NER words

        temp_pred, status = self.predict_NER_spans(processed_result, question)  # get predicted final NER spans

        temp_pred = [pp for pp in temp_pred if pp!="and" and pp!="in" and pp!="hyphen" and pp not in WHwords]
        temp_pred = self.sanity_check(clean_tokens,temp_pred)

        temp_pred = [anitem.replace("'s", "") if anitem.endswith("'s") else anitem for anitem in temp_pred]

        if len(temp_pred)>0:
            status=True

        merge_status, merged_span = self.next_token_merge_check(question, temp_pred)

        if merge_status: # 2nd round of merging (this is required since an entity may contain more than 2 tokens
            merge_status, merged_span = self.next_token_merge_check(question, merged_span)
        merged_span = [tp[:len(tp) - 1] if tp.endswith(",") else tp for tp in merged_span]  # removing tailing zeros
        merged_span = [tok for tok in merged_span if not self.is_numbers(tok)]

        return merged_span, status



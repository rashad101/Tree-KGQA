from transformers import pipeline


class BARTClassifier:
    def __init__(self, args):
        self.classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli', device=0 if args.use_gpu else -1)

    def classify(self, sequence, candidate_labels, multi_class=None):
        return self.classifier(sequence, candidate_labels) if multi_class is None else self.classifier(sequence, candidate_labels, multi_class=True)
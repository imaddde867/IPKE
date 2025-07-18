from transformers import pipeline

class Classifier:
    def __init__(self):
        self.classifier = None

    def _load_model(self):
        if self.classifier is None:
            # Using a zero-shot classification model from Hugging Face
            # This allows classifying text into categories without explicit training on those categories.
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify_document(self, text: str, candidate_labels: list[str]) -> dict:
        if not text or not isinstance(text, str):
            return {"category": "unclassified", "score": 0.0}
        
        self._load_model()
        # The model returns a dictionary with 'sequence', 'labels', and 'scores'
        result = self.classifier(text, candidate_labels)
        
        # We'll return the top predicted label and its score
        if result and result['labels'] and result['scores']:
            return {"category": result['labels'][0], "score": result['scores'][0]}
        else:
            return {"category": "unclassified", "score": 0.0}

# Initialize the classifier globally or as a singleton
classifier = Classifier()

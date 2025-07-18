from keybert import KeyBERT

class KeyphraseExtractor:
    def __init__(self):
        self.kw_model = None

    def _load_model(self):
        if self.kw_model is None:
            # Initialize KeyBERT model. It uses a Sentence-BERT model internally.
            # The default model is 'all-MiniLM-L6-v2', which is good for general purpose.
            self.kw_model = KeyBERT()

    def extract_keyphrases(self, text: str, top_n: int = 10) -> list[str]:
        if not text or not isinstance(text, str):
            return []
        
        self._load_model()
        # Extract keywords/keyphrases
        # The output is a list of tuples: (keyword, score)
        keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
        
        # Return just the keywords
        return [keyword for keyword, score in keywords]

# Initialize the extractor globally or as a singleton
keyphrase_extractor = KeyphraseExtractor()

from transformers import pipeline

class NERExtractor:
    def __init__(self):
        # Using a pre-trained NER model from Hugging Face
        # This model identifies entities like PER (person), ORG (organization), LOC (location), etc.
        # For industrial documentation, we might need to fine-tune a model or use a more specialized one later.
        self.nlp = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

    def extract_entities(self, text: str) -> list[dict]:
        if not text or not isinstance(text, str):
            return []
        entities = self.nlp(text)
        # Filter out unwanted entity types or reformat if necessary
        # For example, we might only care about specific types of entities relevant to industrial processes.
        return entities

# Initialize the extractor globally or as a singleton to avoid reloading the model multiple times
ner_extractor = NERExtractor()

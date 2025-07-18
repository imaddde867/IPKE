from elasticsearch import Elasticsearch, BadRequestError
import time

ELASTICSEARCH_URL = "http://elasticsearch:9200"
INDEX_NAME = "documents"

class ElasticsearchClient:
    def __init__(self):
        self.es = None
        self._initialized = False

    def _initialize(self, max_retries=10, retry_delay=5):
        if self._initialized:
            return
        
        for i in range(max_retries):
            try:
                self.es = Elasticsearch(ELASTICSEARCH_URL)
                # Ping to check connection
                if self.es.ping():
                    print(f"Successfully connected to Elasticsearch at {ELASTICSEARCH_URL}")
                    self.create_index(max_retries=max_retries, retry_delay=retry_delay)
                    self._initialized = True
                    return
                else:
                    print(f"Elasticsearch ping failed. Retrying in {retry_delay} seconds...")
            except Exception as e:
                print(f"Error connecting to Elasticsearch (attempt {i+1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        raise ConnectionError("Could not connect to Elasticsearch after multiple retries.")

    def create_index(self, max_retries=10, retry_delay=5):
        for i in range(max_retries):
            try:
                if not self.es.indices.exists(index=INDEX_NAME):
                    self.es.indices.create(index=INDEX_NAME, ignore=400, body={
                        "settings": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0
                        },
                        "mappings": {
                            "properties": {
                                "document_id": {"type": "integer"},
                                "filename": {"type": "keyword"},
                                "file_type": {"type": "keyword"},
                                "extracted_text": {"type": "text"},
                                "classification_category": {"type": "keyword"},
                                "extracted_entities": {"type": "nested", "properties": {
                                    "text": {"type": "text"},
                                    "entity_type": {"type": "keyword"}
                                }},
                                "key_phrases": {"type": "keyword"},
                                "processing_timestamp": {"type": "date"},
                                "document_sections": {"type": "object"} # Added for sections
                            }
                        }
                    })
                    print(f"Elasticsearch index '{INDEX_NAME}' created.")
                return
            except BadRequestError as e:
                print(f"Elasticsearch BadRequestError during index creation (attempt {i+1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except Exception as e:
                print(f"Unexpected error during index creation: {e}")
                raise
        raise ConnectionError("Could not create Elasticsearch index after multiple retries.")

    def index_document(self, document_data: dict):
        self._initialize()
        try:
            self.es.index(index=INDEX_NAME, id=document_data['document_id'], document=document_data)
            print(f"Document {document_data['document_id']} indexed in Elasticsearch.")
        except Exception as e:
            print(f"Error indexing document {document_data['document_id']}: {e}")

    def search_documents(self, query: str, field: str = "extracted_text", size: int = 10) -> list:
        self._initialize()
        search_body = {
            "query": {
                "match": {
                    field: query
                }
            },
            "size": size
        }
        response = self.es.search(index=INDEX_NAME, body=search_body)
        return [hit['_source'] for hit in response['hits']['hits']]

# Initialize the Elasticsearch client globally
es_client = ElasticsearchClient()

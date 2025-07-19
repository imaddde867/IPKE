from elasticsearch import Elasticsearch, BadRequestError
from src.logging_config import get_logger
from src.exceptions import SearchError, ServiceUnavailableError
from src.config import config_manager
import time

logger = get_logger(__name__)

# Get configuration from config manager
ELASTICSEARCH_URL = config_manager.get_elasticsearch_url()
INDEX_NAME = config_manager.get_config().elasticsearch.index_name

class ElasticsearchClient:
    def __init__(self):
        self.es = None
        self._initialized = False
    
    def _ensure_connection(self, max_retries=None, retry_delay=None):
        """Lazy initialization of Elasticsearch connection."""
        if max_retries is None:
            max_retries = config_manager.get_config().processing.max_retry_attempts
        if retry_delay is None:
            retry_delay = config_manager.get_config().processing.retry_delay_seconds
        if self._initialized and self.es is not None:
            return
        
        logger.info(f"Initializing Elasticsearch connection to {ELASTICSEARCH_URL}")
        
        for i in range(max_retries):
            try:
                self.es = Elasticsearch(ELASTICSEARCH_URL)
                # Ping to check connection
                if self.es.ping():
                    logger.info(f"Successfully connected to Elasticsearch at {ELASTICSEARCH_URL}")
                    self._create_index()
                    self._initialized = True
                    return
                else:
                    logger.warning(f"Elasticsearch ping failed. Retrying in {retry_delay} seconds...")
            except Exception as e:
                logger.warning(f"Error connecting to Elasticsearch (attempt {i+1}/{max_retries}): {e}")
                if i < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(retry_delay)
        
        raise ServiceUnavailableError(
            "Could not connect to Elasticsearch after multiple retries",
            service_name="Elasticsearch",
            service_url=ELASTICSEARCH_URL,
            retry_count=max_retries
        )

    def _create_index(self, max_retries=None, retry_delay=None):
        """Create the Elasticsearch index if it doesn't exist."""
        if max_retries is None:
            max_retries = config_manager.get_config().processing.max_retry_attempts
        if retry_delay is None:
            retry_delay = config_manager.get_config().processing.retry_delay_seconds
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
                                "document_sections": {"type": "object"}
                            }
                        }
                    })
                    logger.info(f"Elasticsearch index '{INDEX_NAME}' created")
                else:
                    logger.info(f"Elasticsearch index '{INDEX_NAME}' already exists")
                return
            except BadRequestError as e:
                logger.warning(f"Elasticsearch BadRequestError during index creation (attempt {i+1}/{max_retries}): {e}")
                if i < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error during index creation: {e}")
                raise SearchError(
                    f"Failed to create Elasticsearch index: {str(e)}",
                    index_name=INDEX_NAME,
                    operation="create_index"
                ) from e
        
        raise SearchError(
            "Could not create Elasticsearch index after multiple retries",
            index_name=INDEX_NAME,
            operation="create_index"
        )

    def index_document(self, document_data: dict):
        """Index a document in Elasticsearch."""
        try:
            self._ensure_connection()
            self.es.index(index=INDEX_NAME, id=document_data['document_id'], document=document_data)
            logger.info(f"Document {document_data['document_id']} indexed in Elasticsearch")
        except Exception as e:
            raise SearchError(
                f"Failed to index document {document_data.get('document_id', 'unknown')}",
                index_name=INDEX_NAME,
                operation="index_document",
                details={'document_id': document_data.get('document_id')}
            ) from e

    def search_documents(self, query: str, field: str = "extracted_text", size: int = 10) -> dict:
        """Search documents in Elasticsearch."""
        try:
            self._ensure_connection()
            search_body = {
                "query": {
                    "match": {
                        field: query
                    }
                },
                "size": size
            }
            response = self.es.search(index=INDEX_NAME, body=search_body)
            logger.info(f"Search completed: query='{query}', field='{field}', results={len(response['hits']['hits'])}")
            return response
        except Exception as e:
            raise SearchError(
                f"Search failed for query: {query}",
                query=query,
                index_name=INDEX_NAME,
                operation="search",
                details={'field': field, 'size': size}
            ) from e

# Initialize the Elasticsearch client globally (lazy initialization)
es_client = ElasticsearchClient()

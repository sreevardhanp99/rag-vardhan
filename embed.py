import os
import json
import re
from typing import List, Dict, Any
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile, SemanticConfiguration,
    SemanticSearch, SemanticPrioritizedFields, SemanticField
)
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        self.embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT")
        
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.search_key)
        )
        
        self.index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=AzureKeyCredential(self.search_key)
        )
        
        # Initialize Azure Blob Storage
        self.blob_service_client = BlobServiceClient.from_connection_string(
            os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
        self.chunks_container = "processed-chunks"
        self.files_container = "uploaded-files"
    
    def sanitize_key(self, key: str) -> str:
        """Sanitize key for Azure Search compatibility"""
        # Keep only letters, digits, underscore, dash, equal sign
        sanitized = re.sub(r'[^a-zA-Z0-9_\-=]', '_', key)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized or "document"
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=self.embedding_deployment
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return []
    
    def delete_index_if_exists(self):
        """Delete the index if it exists"""
        try:
            self.index_client.delete_index(self.index_name)
            logger.info(f"Deleted existing index: {self.index_name}")
        except Exception as e:
            logger.info(f"Index {self.index_name} doesn't exist or couldn't be deleted: {str(e)}")

    def create_search_index(self):
        """Create or update the search index"""
        try:
            # Delete existing index first
            self.delete_index_if_exists()
            
            fields = [
                SearchField(name="id", type=SearchFieldDataType.String, key=True),
                SearchField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="filename", type=SearchFieldDataType.String, filterable=True, searchable=True),
                SearchField(name="blob_url", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(name="char_count", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="default-vector-profile"
                )
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="default-hnsw-config")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw-config"
                    )
                ]
            )
            
            # Configure semantic search
            semantic_config = SemanticConfiguration(
                name="default-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="blob_url")]
                )
            )
            
            semantic_search = SemanticSearch(configurations=[semantic_config])
            
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            self.index_client.create_index(index)
            logger.info(f"Search index '{self.index_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Error creating search index: {str(e)}")
            raise
    
    def upload_chunk_to_blob(self, chunk_id: str, chunk_text: str) -> str:
        """Upload chunk text to Azure Blob Storage and return blob URL"""
        try:
            blob_name = f"{chunk_id}.txt"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.chunks_container, 
                blob=blob_name
            )
            
            # Upload the chunk text
            blob_client.upload_blob(chunk_text, overwrite=True)
            
            # Return the blob URL
            blob_url = blob_client.url
            logger.info(f"Uploaded chunk to blob: {blob_url}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Error uploading chunk to blob: {str(e)}")
            return ""

    def get_chunk_from_blob(self, blob_url: str) -> str:
        """Retrieve chunk text from Azure Blob Storage"""
        try:
            # Extract blob name from URL
            blob_name = blob_url.split('/')[-1]
            blob_client = self.blob_service_client.get_blob_client(
                container=self.chunks_container, 
                blob=blob_name
            )
            
            # Download and return the chunk text
            chunk_text = blob_client.download_blob().readall().decode('utf-8')
            return chunk_text
            
        except Exception as e:
            logger.error(f"Error retrieving chunk from blob: {str(e)}")
            return ""

    def upload_documents(self, chunks: List[Dict[str, str]]) -> bool:
        """Upload document chunks with embeddings to Azure Search"""
        try:
            if not chunks:
                logger.warning("No chunks to upload")
                return False
            
            # Ensure index exists
            self.create_search_index()
            
            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks]
            logger.info(f"Creating embeddings for {len(texts)} chunks...")
            
            # Create embeddings in batches to avoid rate limits
            batch_size = 10
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.create_embeddings(batch_texts)
                embeddings.extend(batch_embeddings)
                logger.info(f"Created embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            if len(embeddings) != len(chunks):
                logger.error("Mismatch between chunks and embeddings count")
                return False
            
            # Prepare documents for upload
            documents = []
            for i, chunk in enumerate(chunks):
                # Sanitize the chunk_id to ensure Azure Search compatibility
                sanitized_chunk_id = self.sanitize_key(chunk["chunk_id"])
                
                # Upload chunk text to blob storage
                blob_url = self.upload_chunk_to_blob(sanitized_chunk_id, chunk["text"])
                
                if not blob_url:
                    logger.error(f"Failed to upload chunk {sanitized_chunk_id} to blob storage")
                    continue
                
                doc = {
                    "id": sanitized_chunk_id,
                    "chunk_id": sanitized_chunk_id,
                    "filename": chunk["filename"],
                    "blob_url": blob_url,
                    "chunk_index": chunk["chunk_index"],
                    "char_count": chunk["char_count"],
                    "embedding": embeddings[i]
                }
                documents.append(doc)
            
            # Upload to Azure Search
            logger.info(f"Uploading {len(documents)} documents to search index...")
            result = self.search_client.upload_documents(documents=documents)
            
            success_count = sum(1 for r in result if r.succeeded)
            logger.info(f"Successfully uploaded {success_count}/{len(documents)} documents")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error uploading documents: {str(e)}")
            return False
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using hybrid search"""
        try:
            # Create query embedding
            query_embedding = self.create_embeddings([query])
            if not query_embedding:
                return []
            
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding[0],
                k_nearest_neighbors=top_k,
                fields="embedding"
            )
            
            # Perform hybrid search (vector + keyword)
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["chunk_id", "filename", "blob_url", "chunk_index"],
                top=top_k
            )
            
            search_results = []
            for result in results:
                # Get chunk text from blob storage
                chunk_text = self.get_chunk_from_blob(result["blob_url"])
                
                search_results.append({
                    "chunk_id": result["chunk_id"],
                    "filename": result["filename"],
                    "text": chunk_text,
                    "chunk_index": result["chunk_index"],
                    "score": result.get("@search.score", 0),
                    "blob_url": result["blob_url"]
                })
            
            logger.info(f"Found {len(search_results)} relevant chunks for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
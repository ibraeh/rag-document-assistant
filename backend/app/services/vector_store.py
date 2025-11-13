"""
Vector store service using ChromaDB for semantic search
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
from app.config import settings
from app.services.embeddings import embedding_service
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB"""

    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            logger.info("Initializing ChromaDB client")
            self.client = chromadb.PersistentClient(
                path=settings.vector_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            self.collection = self.client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                f"ChromaDB initialized. Collection: {settings.collection_name}, "
                f"Documents: {self.collection.count()}"
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise

    def reset_collection(self):
        """Reset the entire collection (delete all data)"""
        try:
            self.client.delete_collection(settings.collection_name)
            self.collection = self.client.create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def add_documents(self, chunks: List[Dict], document_id: str) -> int:
        """Add document chunks to vector store"""
        try:
            if not chunks:
                raise ValueError("No chunks provided")

            ids, documents, metadatas = [], [], []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])

            logger.info(f"Generating embeddings for {len(documents)} chunks")
            embeddings = embedding_service.embed_texts(documents)

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

            logger.info(f"Added {len(chunks)} chunks for document {document_id}")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def search(self, query: str, top_k: int = 4, document_ids: Optional[List[str]] = None) -> List[Dict]:
        """Search for similar chunks"""
        try:
            query_embedding = embedding_service.embed_text(query)
            where_filter = {"document_id": {"$in": document_ids}} if document_ids else None

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )

            formatted_results = []
            if results and results.get("ids"):
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1 - results["distances"][0][i]
                    })

            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document"""
        try:
            results = self.collection.get(where={"document_id": document_id})
            if results and results.get("ids"):
                self.collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])
                logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
                return deleted_count
            return 0
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    def get_document_count(self) -> int:
        """Get total number of unique documents"""
        try:
            results = self.collection.get()
            if results and results.get("metadatas"):
                unique_docs = {meta["document_id"] for meta in results["metadatas"]}
                return len(unique_docs)
            return 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    def get_all_documents(self) -> List[Dict]:
        """Get metadata for all documents"""
        try:
            results = self.collection.get()
            if not results or not results.get("metadatas"):
                return []

            docs_dict = {}
            for metadata in results["metadatas"]:
                doc_id = metadata["document_id"]
                if doc_id not in docs_dict:
                    docs_dict[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "file_type": metadata.get("file_type", "Unknown"),
                        "pages": metadata.get("pages", 0),
                        "chunks": 0
                    }
                docs_dict[doc_id]["chunks"] += 1

            return list(docs_dict.values())

        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []


# Singleton instance
vector_store = VectorStore()

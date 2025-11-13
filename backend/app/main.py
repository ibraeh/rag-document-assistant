
"""
FastAPI main application
RAG Document Assistant - Complete Backend API
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
import shutil
import time
from typing import List, Optional
import logging
import os
import asyncio
from datetime import datetime

from app.config import settings
from app.models import (
    DocumentUploadResponse,
    QuestionRequest,
    QuestionResponse,
    Source,
    DocumentListResponse,
    DocumentInfo,
    HealthResponse,
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SystemStatistics,
    FeedbackRequest,
    FeedbackResponse
)
from app.services.document_processor import document_processor
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service
from app.services.embeddings import embedding_service

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    description="""
    RAG-based Document Question Answering System

    ## Features
    * Upload documents (PDF, DOCX, TXT)
    * Semantic search across documents
    * Question answering with source citations
    * Document management
    * Real-time statistics

    ## Endpoints
    * `/upload` - Upload and process documents
    * `/ask` - Ask questions about documents
    * `/search` - Semantic search
    * `/documents` - List and manage documents
    * `/health` - System health check
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log') if Path('logs').exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {duration:.2f}s"
        )

        return response

    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url.path} - Error: {str(e)}")
        raise

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 50)
    logger.info(f"Starting {settings.app_name}")
    logger.info("=" * 50)

    # Create necessary directories
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.vector_db_path).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Log configuration
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Vector DB path: {settings.vector_db_path}")
    logger.info(f"Max file size: {settings.max_file_size / (1024*1024):.1f}MB")
    logger.info(f"Allowed extensions: {settings.allowed_extensions}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Embedding Model: {settings.embedding_model}")

    # Check services
    doc_count = vector_store.get_document_count()
    logger.info(f"Documents indexed: {doc_count}")

    logger.info("Application started successfully!")
    logger.info("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")

    # Log final statistics
    stats = llm_service.get_usage_stats()
    logger.info(f"Total requests processed: {stats['total_requests']}")
    logger.info(f"Total tokens used: {stats['total_tokens_used']}")

    logger.info("Application shut down successfully")

# ============================================================================
# ROOT & INFO ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict, tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "description": "RAG-based Document Question Answering System",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "ask": "/ask",
            "search": "/search",
            "documents": "/documents",
            "statistics": "/statistics",
            "documentation": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns system health status and basic metrics
    """
    try:
        # Check vector store
        doc_count = vector_store.get_document_count()
        vector_status = "connected"

        # Check embedding service
        try:
            embedding_info = embedding_service.get_model_info()
            embedding_status = "loaded"
        except:
            embedding_status = "error"

        # Check LLM service
        llm_stats = llm_service.get_usage_stats()

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            vector_store_status=vector_status,
            documents_indexed=doc_count,
            uptime_seconds=time.time()  # Simplified - track actual uptime in production
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Service unhealthy"
        )

# ============================================================================
# DOCUMENT UPLOAD ENDPOINT
# ============================================================================

@app.post(
    "/upload", 
    response_model=DocumentUploadResponse,
    tags=["Documents"],
    summary="Upload and process a document"
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document

    - **file**: PDF, DOCX, or TXT file (max 10MB)

    Returns document metadata and processing status
    """
    upload_start = time.time()

    try:
        logger.info(f"Received file upload: {file.filename}")

        # Validate filename
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )

        # Read file content
        content = await file.read()
        file_size = len(content)
        file_type = Path(file.filename).suffix.lower()

        logger.info(f"File size: {file_size / (1024*1024):.2f}MB")

        # Validate file
        try:
            document_processor.validate_file(file.filename, file_size)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Save file temporarily
        file_path = Path(settings.upload_dir) / file.filename

        # Handle duplicate filenames
        if file_path.exists():
            base = file_path.stem
            ext = file_path.suffix
            counter = 1
            while file_path.exists():
                file_path = Path(settings.upload_dir) / f"{base}_{counter}{ext}"
                counter += 1

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File saved to: {file_path}")

        # Process document
        try:
            result = document_processor.process_document(
                str(file_path),
                file.filename,
                file_type
            )
        except Exception as e:
            # Clean up file if processing fails
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=422,
                detail=f"Document processing failed: {str(e)}"
            )

        # Add to vector store
        try:
            num_chunks = vector_store.add_documents(
                result["chunks"],
                result["document_id"]
            )
        except Exception as e:
            # Clean up file if vector store fails
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to index document: {str(e)}"
            )

        # Calculate total time
        total_time = time.time() - upload_start

        logger.info(
            f"Document processed successfully: {result['document_id']} - "
            f"Time: {total_time:.2f}s"
        )

        return DocumentUploadResponse(
            document_id=result["document_id"],
            filename=file.filename,
            pages=result["pages"],
            chunks=num_chunks,
            status="success",
            message=f"Document uploaded and indexed successfully in {total_time:.2f}s",
            file_size=file_size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

# ============================================================================
# QUESTION ANSWERING ENDPOINT
# ============================================================================

@app.post(
    "/ask", 
    response_model=QuestionResponse,
    tags=["Question Answering"],
    summary="Ask a question about documents"
)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about uploaded documents

    - **question**: Your question (3-500 characters)
    - **document_ids**: Optional list of specific documents to search
    - **top_k**: Number of relevant chunks to retrieve (1-10)
    - **include_sources**: Whether to include source citations
    - **temperature**: LLM temperature (0.0-2.0)

    Returns an answer with source citations
    """
    try:
        start_time = time.time()
        logger.info(f"Received question: {request.question[:100]}...")

        # Check if any documents exist
        doc_count = vector_store.get_document_count()
        if doc_count == 0:
            return QuestionResponse(
                answer="No documents have been uploaded yet. Please upload some documents first before asking questions.",
                sources=[],
                question=request.question,
                model_used="none",
                processing_time=time.time() - start_time,
                tokens_used=0
            )

        # Search for relevant chunks
        try:
            search_results = vector_store.search(
                query=request.question,
                top_k=request.top_k,
                document_ids=request.document_ids
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )

        if not search_results:
            return QuestionResponse(
                answer="I couldn't find any relevant information to answer your question. Please try rephrasing or check if the relevant documents are uploaded.",
                sources=[],
                question=request.question,
                model_used="none",
                processing_time=time.time() - start_time,
                tokens_used=0
            )

        logger.info(f"Found {len(search_results)} relevant chunks")

        # Generate answer
        try:
            answer = llm_service.generate_answer(
                request.question,
                search_results,
                temperature=request.temperature,
                use_cache=True
            )
            model_used = settings.llm_model
            tokens_used = None  # Would track from LLM response
        except Exception as e:
            logger.warning(f"LLM generation failed, using fallback: {e}")
            answer = llm_service.generate_fallback_answer(
                request.question,
                search_results
            )
            model_used = "fallback"
            tokens_used = 0

        # Format sources
        sources = []
        if request.include_sources:
            for result in search_results:
                metadata = result['metadata']
                source = Source(
                    document_id=metadata['document_id'],
                    document_name=metadata.get('filename', 'Unknown'),
                    page_number=metadata.get('page_number'),
                    chunk_text=result['text'][:300] + "..." if len(result['text']) > 300 else result['text'],
                    relevance_score=round(result['score'], 3),
                    chunk_id=metadata.get('chunk_id')
                )
                sources.append(source)

        processing_time = time.time() - start_time
        logger.info(f"Question answered in {processing_time:.2f}s")

        return QuestionResponse(
            answer=answer,
            sources=sources,
            question=request.question,
            model_used=model_used,
            processing_time=processing_time,
            tokens_used=tokens_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )

# ============================================================================
# SEMANTIC SEARCH ENDPOINT
# ============================================================================

@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Semantic search across documents"
)
async def search_documents(request: SearchRequest):
    """
    Perform semantic search across documents without generating an answer

    - **query**: Search query
    - **document_ids**: Optional document filter
    - **top_k**: Number of results (1-50)
    - **similarity_threshold**: Minimum similarity score

    Returns relevant document chunks
    """
    try:
        start_time = time.time()
        logger.info(f"Search query: {request.query}")

        # Perform search
        search_results = vector_store.search(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids
        )

        # Filter by similarity threshold if specified
        if request.similarity_threshold:
            search_results = [
                r for r in search_results 
                if r['score'] >= request.similarity_threshold
            ]

        # Format results
        results = []
        for result in search_results:
            metadata = result['metadata']
            search_result = SearchResult(
                document_id=metadata['document_id'],
                document_name=metadata.get('filename', 'Unknown'),
                chunk_text=result['text'],
                page_number=metadata.get('page_number'),
                similarity_score=round(result['score'], 4),
                metadata=metadata
            )
            results.append(search_result)

        processing_time = time.time() - start_time

        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.get(
    "/documents", 
    response_model=DocumentListResponse,
    tags=["Documents"],
    summary="List all documents"
)
async def list_documents():
    """
    List all uploaded documents

    Returns metadata for all indexed documents
    """
    try:
        docs = vector_store.get_all_documents()

        document_infos = []
        for doc in docs:
            doc_info = DocumentInfo(
                document_id=doc['document_id'],
                filename=doc['filename'],
                upload_date=None,  # Could add to metadata
                pages=doc['pages'],
                chunks=doc['chunks'],
                file_size=0,  # Could add to metadata
                file_type=doc['file_type']
            )
            document_infos.append(doc_info)

        return DocumentListResponse(
            documents=document_infos,
            total_count=len(document_infos)
        )

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )

@app.get(
    "/documents/{document_id}",
    response_model=DocumentInfo,
    tags=["Documents"],
    summary="Get document details"
)
async def get_document(document_id: str):
    """Get detailed information about a specific document"""
    try:
        docs = vector_store.get_all_documents()
        doc = next((d for d in docs if d['document_id'] == document_id), None)

        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )

        return DocumentInfo(
            document_id=doc['document_id'],
            filename=doc['filename'],
            upload_date=None,
            pages=doc['pages'],
            chunks=doc['chunks'],
            file_size=0,
            file_type=doc['file_type']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(
    "/documents/{document_id}",
    tags=["Documents"],
    summary="Delete a document"
)
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks

    - **document_id**: ID of document to delete

    Returns deletion confirmation
    """
    try:
        logger.info(f"Deleting document: {document_id}")

        deleted_count = vector_store.delete_document(document_id)

        if deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )

        # Try to delete the physical file
        try:
            for file_path in Path(settings.upload_dir).glob(f"*{document_id}*"):
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete physical file: {e}")

        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully",
            "chunks_deleted": deleted_count,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

# ============================================================================
# STATISTICS ENDPOINT
# ============================================================================

@app.get(
    "/statistics",
    response_model=SystemStatistics,
    tags=["Statistics"],
    summary="Get system statistics"
)
async def get_statistics():
    """
    Get overall system statistics

    Returns metrics about documents, queries, and performance
    """
    try:
        # Get document stats
        all_docs = vector_store.get_all_documents()
        total_docs = len(all_docs)
        total_chunks = sum(doc['chunks'] for doc in all_docs)

        # Get LLM stats
        llm_stats = llm_service.get_usage_stats()

        # Get embedding stats
        embedding_info = embedding_service.get_model_info()

        return SystemStatistics(
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_queries=llm_stats['total_requests'],
            average_query_time=None,  # Would calculate from logs
            average_answer_length=None,
            most_queried_documents=None
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FEEDBACK ENDPOINT
# ============================================================================

@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    summary="Submit feedback on an answer"
)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback on a question/answer

    Helps improve the system over time
    """
    try:
        # In production, store this in a database
        logger.info(
            f"Feedback received - Rating: {feedback.rating}, "
            f"Helpful: {feedback.helpful}, Accurate: {feedback.accurate}"
        )

        feedback_id = f"fb_{int(time.time())}"

        return FeedbackResponse(
            message="Thank you for your feedback!",
            feedback_id=feedback_id
        )

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.post(
    "/clear-cache",
    tags=["Utilities"],
    summary="Clear all caches"
)
async def clear_caches():
    """Clear LLM and embedding caches"""
    try:
        llm_service.clear_cache()
        embedding_service.clear_cache()

        return {
            "success": True,
            "message": "All caches cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/system-info",
    tags=["Info"],
    summary="Get system information"
)
async def get_system_info():
    """Get detailed system information"""
    try:
        return {
            "app_name": settings.app_name,
            "version": "1.0.0",
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": embedding_service.get_embedding_dimension(),
            "max_file_size_mb": settings.max_file_size / (1024 * 1024),
            "supported_formats": list(settings.allowed_extensions),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "vector_store": "ChromaDB",
            "documents_indexed": vector_store.get_document_count(),
            "llm_stats": llm_service.get_usage_stats(),
            "cache_stats": embedding_service.cache.get_cache_stats()
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=settings.debug
    )

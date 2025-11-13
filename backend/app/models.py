
"""
Pydantic models for request/response validation
RAG Document Assistant - Complete Models File
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, Enum):
    """Supported file types"""
    PDF = ".pdf"
    DOCX = ".docx"
    TXT = ".txt"


# ============================================================================
# DOCUMENT MODELS
# ============================================================================

class DocumentUploadResponse(BaseModel):
    """Response after uploading a document"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    pages: int = Field(..., ge=0, description="Number of pages")
    chunks: int = Field(..., ge=0, description="Number of text chunks created")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.now)
    file_size: Optional[int] = Field(None, description="File size in bytes")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "research_paper_a1b2c3d4",
                "filename": "research_paper.pdf",
                "pages": 15,
                "chunks": 32,
                "status": "success",
                "message": "Document uploaded and indexed successfully",
                "file_size": 2048576
            }
        }


class DocumentInfo(BaseModel):
    """Document metadata information"""
    document_id: str
    filename: str
    upload_date: Optional[datetime] = None
    pages: int = Field(..., ge=0)
    chunks: int = Field(..., ge=0)
    file_size: int = Field(..., ge=0)
    file_type: str
    status: Optional[str] = "completed"

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "research_paper_a1b2c3d4",
                "filename": "research_paper.pdf",
                "upload_date": "2025-01-15T10:30:00",
                "pages": 15,
                "chunks": 32,
                "file_size": 2048576,
                "file_type": ".pdf",
                "status": "completed"
            }
        }


class DocumentListResponse(BaseModel):
    """Response for listing documents"""
    documents: List[DocumentInfo]
    total_count: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "document_id": "doc1",
                        "filename": "file1.pdf",
                        "pages": 10,
                        "chunks": 25,
                        "file_size": 1024000,
                        "file_type": ".pdf"
                    }
                ],
                "total_count": 1
            }
        }


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document"""
    document_id: str
    message: str
    chunks_deleted: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# QUESTION ANSWERING MODELS
# ============================================================================

class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(
        ..., 
        min_length=3, 
        max_length=500,
        description="Question to ask about the documents"
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of document IDs to search. If None, searches all documents"
    )
    top_k: int = Field(
        default=4, 
        ge=1, 
        le=10,
        description="Number of relevant chunks to retrieve"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source citations in response"
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation"
    )

    @validator('question')
    def question_not_empty(cls, v):
        """Validate question is not empty or whitespace"""
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main findings of this research?",
                "document_ids": ["research_paper_a1b2c3d4"],
                "top_k": 4,
                "include_sources": True,
                "temperature": 0.7
            }
        }


class Source(BaseModel):
    """Source citation information"""
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document filename")
    page_number: Optional[int] = Field(None, ge=1, description="Page number in document")
    chunk_text: str = Field(..., description="Relevant text excerpt")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    chunk_id: Optional[int] = Field(None, description="Chunk identifier within document")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "research_paper_a1b2c3d4",
                "document_name": "research_paper.pdf",
                "page_number": 5,
                "chunk_text": "The study found that...",
                "relevance_score": 0.87,
                "chunk_id": 12
            }
        }


class QuestionResponse(BaseModel):
    """Response model for question answering"""
    answer: str = Field(..., description="Generated answer to the question")
    sources: List[Source] = Field(default_factory=list, description="Source citations")
    question: str = Field(..., description="Original question asked")
    model_used: str = Field(..., description="Model used for generation")
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used (if available)")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the research, the main findings are...",
                "sources": [
                    {
                        "document_id": "doc1",
                        "document_name": "research.pdf",
                        "page_number": 5,
                        "chunk_text": "The study found...",
                        "relevance_score": 0.87
                    }
                ],
                "question": "What are the main findings?",
                "model_used": "gpt-3.5-turbo",
                "processing_time": 2.34,
                "tokens_used": 450
            }
        }


# ============================================================================
# CONVERSATION MODELS (Optional - for chat history)
# ============================================================================

class ConversationMessage(BaseModel):
    """Single message in a conversation"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[Source]] = Field(None, description="Sources for assistant messages")

    @validator('role')
    def validate_role(cls, v):
        """Validate role is either user or assistant"""
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be "user", "assistant", or "system"')
        return v


class ConversationHistory(BaseModel):
    """Conversation history"""
    session_id: str = Field(..., description="Unique session identifier")
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    document_ids: Optional[List[str]] = Field(None, description="Documents in this conversation")


# ============================================================================
# SYSTEM MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="System status: 'healthy' or 'unhealthy'")
    version: str = Field(..., description="Application version")
    vector_store_status: str = Field(..., description="Vector store connection status")
    documents_indexed: int = Field(..., ge=0, description="Number of documents in index")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "vector_store_status": "connected",
                "documents_indexed": 42,
                "uptime_seconds": 3600.5
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type or category")
    detail: Optional[str] = Field(None, description="Detailed error message")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Question must be at least 3 characters long",
                "timestamp": "2025-01-15T10:30:00",
                "request_id": "req_123abc"
            }
        }


# ============================================================================
# SEARCH & RETRIEVAL MODELS
# ============================================================================

class SearchRequest(BaseModel):
    """Request for semantic search without answer generation"""
    query: str = Field(..., min_length=3, max_length=500)
    document_ids: Optional[List[str]] = None
    top_k: int = Field(default=10, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Single search result"""
    document_id: str
    document_name: str
    chunk_text: str
    page_number: Optional[int] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response for search requests"""
    results: List[SearchResult]
    query: str
    total_results: int
    processing_time: float


# ============================================================================
# STATISTICS & ANALYTICS MODELS
# ============================================================================

class DocumentStatistics(BaseModel):
    """Statistics for a single document"""
    document_id: str
    filename: str
    total_queries: int = Field(default=0, ge=0)
    average_relevance_score: Optional[float] = None
    last_accessed: Optional[datetime] = None


class SystemStatistics(BaseModel):
    """Overall system statistics"""
    total_documents: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    total_queries: int = Field(..., ge=0)
    average_query_time: Optional[float] = None
    average_answer_length: Optional[int] = None
    most_queried_documents: Optional[List[str]] = None


# ============================================================================
# BATCH OPERATIONS MODELS
# ============================================================================

class BatchQuestionRequest(BaseModel):
    """Request for batch question processing"""
    questions: List[str] = Field(..., min_items=1, max_items=50)
    document_ids: Optional[List[str]] = None
    top_k: int = Field(default=4, ge=1, le=10)


class BatchQuestionResponse(BaseModel):
    """Response for batch questions"""
    results: List[QuestionResponse]
    total_questions: int
    successful: int
    failed: int
    total_processing_time: float


# ============================================================================
# FEEDBACK MODELS
# ============================================================================

class FeedbackRequest(BaseModel):
    """User feedback on an answer"""
    question_id: Optional[str] = Field(None, description="Identifier of the question")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional feedback comment")
    helpful: Optional[bool] = Field(None, description="Was the answer helpful?")
    accurate: Optional[bool] = Field(None, description="Was the answer accurate?")

    class Config:
        json_schema_extra = {
            "example": {
                "question_id": "q_123abc",
                "rating": 5,
                "comment": "Very helpful and accurate answer with good sources",
                "helpful": True,
                "accurate": True
            }
        }


class FeedbackResponse(BaseModel):
    """Response after submitting feedback"""
    message: str
    feedback_id: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# EXPORT MODELS
# ============================================================================

class ExportFormat(str, Enum):
    """Supported export formats"""
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    TXT = "txt"


class ExportRequest(BaseModel):
    """Request to export conversation or results"""
    session_id: Optional[str] = None
    format: ExportFormat = Field(default=ExportFormat.PDF)
    include_sources: bool = Field(default=True)
    include_metadata: bool = Field(default=False)


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class ChunkingConfig(BaseModel):
    """Configuration for document chunking"""
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    separators: List[str] = Field(default=["", "", ". ", " ", ""])


class RetrievalConfig(BaseModel):
    """Configuration for retrieval"""
    top_k: int = Field(default=4, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    reranking_enabled: bool = Field(default=False)


# ============================================================================
# UTILITY MODELS
# ============================================================================

class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


"""
LLM service for question answering
RAG Document Assistant - Complete LLM Service
"""
from typing import List, Dict, Optional, Iterator
from openai import OpenAI
from app.config import settings
import logging
import time
import hashlib

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Collection of prompt templates for different tasks"""

    SYSTEM_QA = """You are a helpful AI assistant that answers questions based on provided documents.

Instructions:
1. Answer using ONLY the provided sources
2. If not found, say "I cannot find this information in the provided documents"
3. Cite sources (e.g., "According to Source 1...")
4. Be concise but complete
5. Synthesize multiple sources if needed
6. Do not infer or fabricate
7. Maintain a professional tone"""

    SYSTEM_SUMMARIZATION = """You are a helpful AI assistant that creates concise summaries of documents.

Instructions:
1. Summarize the provided text
2. Focus on key points
3. Be objective and accurate
4. Limit to 3–5 sentences
5. Use bullet points if appropriate"""

    SYSTEM_COMPARISON = """You are a helpful AI assistant that compares information across multiple documents.

Instructions:
1. Identify similarities and differences
2. Reference each source clearly
3. Highlight agreements and disagreements
4. Be objective and organized"""

    @staticmethod
    def create_qa_prompt(question: str, context_chunks: List[Dict]) -> tuple:
        context = "".join(f"[Source {i+1}: {chunk.get('metadata', {}).get('filename', 'Unknown')}, Page {chunk.get('metadata', {}).get('page_number', '?')}]{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        )
        user_prompt = f"""Context from documents:
{context}

Question: {question}

Answer the question based on the context provided above. Remember to cite your sources."""
        return PromptTemplate.SYSTEM_QA, user_prompt

    @staticmethod
    def create_summarization_prompt(text: str, max_sentences: int = 5) -> tuple:
        user_prompt = f"Please summarize the following text in {max_sentences} sentences or less:{text}Summary:"
        return PromptTemplate.SYSTEM_SUMMARIZATION, user_prompt

    @staticmethod
    def create_comparison_prompt(question: str, sources_by_doc: Dict[str, List[Dict]]) -> tuple:
        context = ""
        for doc_id, chunks in sources_by_doc.items():
            if chunks:
                doc_name = chunks[0].get('metadata', {}).get('filename', doc_id)
                context += f"=== From {doc_name} ===" + "".join(chunk['text'] for chunk in chunks)
        user_prompt = f"""Compare the information from different documents regarding this question:Question: {question}Documents:{context}

Provide a comparative analysis:"""
        return PromptTemplate.SYSTEM_COMPARISON, user_prompt



class ResponseCache:
    """Simple in-memory cache for LLM responses"""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _generate_key(self, question: str, context_ids: List[str], model: str) -> str:
        content = f"{model}:{question}:{','.join(sorted(context_ids))}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, question: str, context_chunks: List[Dict], model: str) -> Optional[str]:
        try:
            context_ids = [chunk.get('id', str(i)) for i, chunk in enumerate(context_chunks)]
            key = self._generate_key(question, context_ids, model)
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            return None

    def set(self, question: str, context_chunks: List[Dict], model: str, response: str):
        try:
            context_ids = [chunk.get('id', str(i)) for i, chunk in enumerate(context_chunks)]
            key = self._generate_key(question, context_ids, model)
            if len(self.cache) >= self.max_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = response
        except Exception as e:
            logger.warning(f"Cache store error: {e}")

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "cached_items": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2)
        }

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0



class LLMService:
    """Handles LLM-based question answering"""

    def __init__(self):
        self.client = None
        self.cache = ResponseCache()
        self.total_tokens_used = 0
        self.total_requests = 0
        self._initialize_client()

    def _initialize_client(self):
        if settings.openai_api_key:
            try:
                self.client = OpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"OpenAI init error: {e}")
                self.client = None
        else:
            logger.warning("OpenAI API key missing")

    def generate_answer(self, question: str, context_chunks: List[Dict], model=None, temperature=None, max_tokens=None, use_cache=True) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not context_chunks:
            return "No relevant information available. Please upload documents first."

        model = model or settings.llm_model
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.max_tokens

        if use_cache:
            cached = self.cache.get(question, context_chunks, model)
            if cached:
                return cached

        if not self.client:
            return self.generate_fallback_answer(question, context_chunks)

        system_prompt, user_prompt = PromptTemplate.create_qa_prompt(question, context_chunks)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # temperature=temperature,
                # temperature=1,
                # max_completion_tokens=max_tokens
            )
            # print(response)
            answer = response.choices[0].message.content.strip()
            # print(answer)
            self.total_requests += 1
            if hasattr(response, "usage"):
                self.total_tokens_used += response.usage.total_tokens
            if use_cache:
                self.cache.set(question, context_chunks, model, answer)
            return answer
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self.generate_fallback_answer(question, context_chunks)

    def generate_fallback_answer(self, question: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "I couldn't find any relevant information to answer your question."
        answer_parts = ["Based on the documents, here's what I found:"]
        for i, chunk in enumerate(context_chunks[:3], 1):
            meta = chunk.get("metadata", {})
            doc = meta.get("filename", "Unknown")
            page = meta.get("page_number", "?")
            text = chunk["text"][:400]
            answer_parts.append(f"**From {doc} (Page {page}):**{text}...")
        answer_parts.append("*Note: This is a fallback response. For better answers, configure an OpenAI API key.*")
        return "".join(answer_parts)

    def summarize_document(self, text: str, max_sentences: int = 5, model: Optional[str] = None) -> str:
        if not self.client:
            return "Summary generation requires OpenAI API key."
        model = model or settings.llm_model
        system_prompt, user_prompt = PromptTemplate.create_summarization_prompt(text, max_sentences)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # temperature=0.5,
                # max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summary error: {e}")
            return f"Error generating summary: {str(e)}"


    def compare_documents(self, question: str, sources_by_doc: Dict[str, List[Dict]], model: Optional[str] = None) -> str:
        """
        Generate comparative analysis across multiple documents.
        """
        if not self.client:
            return "Comparison requires OpenAI API key."

        model = model or settings.llm_model
        system_prompt, user_prompt = PromptTemplate.create_comparison_prompt(question, sources_by_doc)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # temperature=0.7,
                # max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return f"Error generating comparison: {str(e)}"

    def get_usage_stats(self) -> dict:
        """Get usage statistics."""
        cache_stats = self.cache.get_stats()
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "average_tokens_per_request": round(self.total_tokens_used / self.total_requests, 2) if self.total_requests > 0 else 0,
            "cache_stats": cache_stats,
            "client_initialized": self.client is not None
        }

    def estimate_cost(self, model: Optional[str] = None) -> dict:
        """Estimate API cost based on usage."""
        model = model or settings.llm_model
        pricing = {
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        }

        if model not in pricing:
            return {"error": f"Pricing not available for model: {model}"}

        avg_price_per_1m = (pricing[model]["input"] + pricing[model]["output"]) / 2
        estimated_cost = (self.total_tokens_used / 1_000_000) * avg_price_per_1m

        return {
            "model": model,
            "total_tokens": self.total_tokens_used,
            "estimated_cost_usd": round(estimated_cost, 4),
            "pricing_per_1m_tokens": pricing[model]
        }

    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()

    def reset_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_requests = 0
        logger.info("Usage statistics reset")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

llm_service = LLMService()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ask_question(question: str, context_chunks: List[Dict], **kwargs) -> str:
    """
    Convenience function to ask a question.
    """
    return llm_service.generate_answer(question, context_chunks, **kwargs)


def get_llm_stats() -> dict:
    """Get LLM service statistics."""
    return llm_service.get_usage_stats()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # example_chunks = [
    #     {
    #         "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    #         "metadata": {"filename": "ml_intro.pdf", "page_number": 1, "document_id": "doc1"}
    #     },
    #     {
    #         "text": "Deep learning uses neural networks with multiple layers to process complex patterns.",
    #         "metadata": {"filename": "ml_intro.pdf", "page_number": 2, "document_id": "doc1"}
    #     }
    # ]

    # print("=== Testing Question Answering ===")
    # question = "What is machine learning?"
    # answer = llm_service.generate_answer(question, example_chunks)
    # print(f"Question: {question}")
    # print(f"Answer: {answer}")

    # print("=== Testing Cache ===")
    # answer2 = llm_service.generate_answer(question, example_chunks)
    # print("Second request (should be cached):")
    # print(answer2)


    # print("=== Usage Statistics ===")
    # stats = llm_service.get_usage_stats()
    # for key, value in stats.items():
    #     print(f"{key}: {value}")

    # print("=== Cost Estimate ===")
    # cost = llm_service.estimate_cost()
    # for key, value in cost.items():
    #     print(f"{key}: {value}")

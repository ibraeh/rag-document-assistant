"""
Helper utilities for RAG Document Assistant
Common functions used across the application
"""
import re
import hashlib
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


# ============================================================================
# TEXT PROCESSING HELPERS
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\[\]"-]', '', text)
    
    # Trim
    return text.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract top keywords from text (simple implementation)
    
    Args:
        text: Input text
        top_n: Number of keywords to return
    
    Returns:
        List of keywords
    """
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
    }
    
    # Tokenize and filter
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    filtered_words = [w for w in words if w not in stop_words]
    
    # Count frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def highlight_text(text: str, query: str, highlight_tag: str = "**") -> str:
    """
    Highlight query terms in text
    
    Args:
        text: Text to highlight
        query: Search query
        highlight_tag: Tag to wrap matches (default: markdown bold)
    
    Returns:
        Text with highlights
    """
    if not query or not text:
        return text
    
    # Split query into words
    query_words = query.lower().split()
    
    # Highlight each word
    result = text
    for word in query_words:
        pattern = re.compile(f'({re.escape(word)})', re.IGNORECASE)
        result = pattern.sub(f'{highlight_tag}\\1{highlight_tag}', result)
    
    return result


# ============================================================================
# DATA VALIDATION HELPERS
# ============================================================================

def validate_email(email: str) -> bool:
    """
    Validate email address format
    
    Args:
        email: Email address
    
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url: URL string
    
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^https?://[^\s]+$'
    return bool(re.match(pattern, url))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove dangerous characters
    
    Args:
        filename: Original filename
    
    Returns:
        Safe filename
    """
    # Remove path separators and dangerous characters
    safe = re.sub(r'[^\w\s.-]', '', filename)
    safe = safe.replace(' ', '_')
    
    # Limit length
    if len(safe) > 255:
        name = Path(safe).stem[:250]
        ext = Path(safe).suffix
        safe = name + ext
    
    return safe


# ============================================================================
# HASHING & ENCODING HELPERS
# ============================================================================

def generate_hash(content: Union[str, bytes], algorithm: str = 'md5') -> str:
    """
    Generate hash of content
    
    Args:
        content: Content to hash
        algorithm: Hash algorithm (md5, sha1, sha256)
    
    Returns:
        Hex digest of hash
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    if algorithm == 'md5':
        return hashlib.md5(content).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(content).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(content).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate unique ID
    
    Args:
        prefix: Optional prefix
        length: Length of random part
    
    Returns:
        Unique ID string
    """
    import secrets
    random_part = secrets.token_hex(length // 2)
    
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


# ============================================================================
# DATE & TIME HELPERS
# ============================================================================

def format_timestamp(dt: Optional[datetime] = None, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime object
    
    Args:
        dt: Datetime object (default: now)
        format: Format string
    
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format)


def parse_timestamp(timestamp_str: str, format: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """
    Parse timestamp string to datetime
    
    Args:
        timestamp_str: Timestamp string
        format: Format string
    
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        return datetime.strptime(timestamp_str, format)
    except (ValueError, TypeError):
        return None


def time_ago(dt: datetime) -> str:
    """
    Get human-readable time difference
    
    Args:
        dt: Datetime object
    
    Returns:
        Human-readable string (e.g., "2 hours ago")
    """
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif seconds < 2592000:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    else:
        months = int(seconds / 2592000)
        return f"{months} month{'s' if months != 1 else ''} ago"


# ============================================================================
# FILE & PATH HELPERS
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_human(size_bytes: int) -> str:
    """
    Convert file size to human-readable format
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Human-readable size (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_file_extension(filename: str) -> str:
    """
    Get file extension in lowercase
    
    Args:
        filename: Filename
    
    Returns:
        Extension (e.g., ".pdf")
    """
    return Path(filename).suffix.lower()


def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List files in directory matching pattern
    
    Args:
        directory: Directory path
        pattern: Glob pattern
    
    Returns:
        List of Path objects
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    return sorted(directory.glob(pattern))


# ============================================================================
# JSON & SERIALIZATION HELPERS
# ============================================================================

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string
    
    Args:
        json_str: JSON string
        default: Default value if parsing fails
    
    Returns:
        Parsed object or default
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: Any = None) -> Optional[str]:
    """
    Safely serialize object to JSON
    
    Args:
        obj: Object to serialize
        default: Default value if serialization fails
    
    Returns:
        JSON string or None
    """
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return default


def pretty_json(obj: Any) -> str:
    """
    Format object as pretty JSON
    
    Args:
        obj: Object to format
    
    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(obj, indent=2, default=str)


# ============================================================================
# PERFORMANCE & TIMING HELPERS
# ============================================================================

def timer(func):
    """
    Decorator to measure function execution time
    
    Usage:
        @timer
        def my_function():
            # code here
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Delay between retries in seconds
    
    Usage:
        @retry(max_attempts=3, delay=2.0)
        def my_function():
            # code that might fail
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}), retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator


class Timer:
    """
    Context manager for timing code blocks
    
    Usage:
        with Timer("operation"):
            # code to time
    """
    
    def __init__(self, name: str = "operation", log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        logger.log(self.log_level, f"{self.name} took {self.elapsed:.2f}s")


# ============================================================================
# STATISTICS HELPERS
# ============================================================================

def calculate_statistics(values: List[Union[int, float]]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with statistics
    """
    if not values:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0
        }
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    return {
        'count': n,
        'min': sorted_values[0],
        'max': sorted_values[-1],
        'mean': sum(values) / n,
        'median': sorted_values[n // 2] if n % 2 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    }


def calculate_percentage(part: Union[int, float], total: Union[int, float], decimals: int = 2) -> float:
    """
    Calculate percentage
    
    Args:
        part: Part value
        total: Total value
        decimals: Decimal places
    
    Returns:
        Percentage value
    """
    if total == 0:
        return 0.0
    return round((part / total) * 100, decimals)


# ============================================================================
# LOGGING HELPERS
# ============================================================================

def log_function_call(func):
    """
    Decorator to log function calls with arguments
    
    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            # code here
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_str = ', '.join(repr(a) for a in args)
        kwargs_str = ', '.join(f"{k}={v!r}" for k, v in kwargs.items())
        all_args = ', '.join(filter(None, [args_str, kwargs_str]))
        
        logger.debug(f"Calling {func.__name__}({all_args})")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result!r}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with specified configuration
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# PAGINATION HELPERS
# ============================================================================

def paginate(items: List[Any], page: int = 1, page_size: int = 10) -> Dict[str, Any]:
    """
    Paginate a list of items
    
    Args:
        items: List of items
        page: Page number (1-indexed)
        page_size: Items per page
    
    Returns:
        Dictionary with paginated results
    """
    total_items = len(items)
    total_pages = (total_items + page_size - 1) // page_size
    
    # Validate page
    page = max(1, min(page, total_pages or 1))
    
    # Calculate slice indices
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    return {
        'items': items[start_idx:end_idx],
        'page': page,
        'page_size': page_size,
        'total_items': total_items,
        'total_pages': total_pages,
        'has_next': page < total_pages,
        'has_prev': page > 1
    }


# ============================================================================
# CACHING HELPERS
# ============================================================================

class SimpleCache:
    """
    Simple in-memory cache with expiration
    
    Usage:
        cache = SimpleCache(ttl=300)  # 5 minutes
        cache.set('key', 'value')
        value = cache.get('key')
    """
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize cache
        
        Args:
            ttl: Time to live in seconds
        """
        self.cache = {}
        self.ttl = ttl
    
    def set(self, key: str, value: Any):
        """Set cache value"""
        self.cache[key] = {
            'value': value,
            'expires': time.time() + self.ttl
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cache value"""
        if key not in self.cache:
            return default
        
        item = self.cache[key]
        
        # Check expiration
        if time.time() > item['expires']:
            del self.cache[key]
            return default
        
        return item['value']
    
    def delete(self, key: str):
        """Delete cache value"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
    
    def cleanup(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            k for k, v in self.cache.items()
            if current_time > v['expires']
        ]
        for key in expired_keys:
            del self.cache[key]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Text Processing ===")
    text = "  This is   a test   with extra   spaces!  "
    print(f"Original: '{text}'")
    print(f"Cleaned: '{clean_text(text)}'")
    print(f"Truncated: '{truncate_text(text, 20)}'")
    
    print("\n=== Hashing ===")
    content = "Hello, World!"
    print(f"MD5: {generate_hash(content, 'md5')}")
    print(f"SHA256: {generate_hash(content, 'sha256')}")
    print(f"Unique ID: {generate_id('doc', 16)}")
    
    print("\n=== File Sizes ===")
    sizes = [1024, 1048576, 1073741824]
    for size in sizes:
        print(f"{size} bytes = {get_file_size_human(size)}")
    
    print("\n=== Timing ===")
    @timer
    def slow_function():
        time.sleep(0.1)
        return "done"
    
    result = slow_function()
    
    print("\n=== Statistics ===")
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats = calculate_statistics(values)
    print(f"Stats: {pretty_json(stats)}")
    
    print("\n=== Pagination ===")
    items = list(range(1, 101))
    page1 = paginate(items, page=1, page_size=10)
    print(f"Page 1: {page1['items']}")
    print(f"Total pages: {page1['total_pages']}")
    
    print("\n=== Cache ===")
    cache = SimpleCache(ttl=5)
    cache.set('test_key', 'test_value')
    print(f"Retrieved: {cache.get('test_key')}")
    
    print("\n=== Complete ===")
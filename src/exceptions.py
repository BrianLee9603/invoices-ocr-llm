"""
Custom exception classes for the Document AI pipeline.
Allows separating transient (retryable) errors from persistent (non-retryable) errors.
"""

class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class TransientError(PipelineError):
    """
    Raised for temporary failures that should be retried.
    Examples: LLM rate limit (HTTP 429), network timeouts, API disconnects.
    """
    pass


class PersistentError(PipelineError):
    """
    Raised for permanent failures that should fail immediately.
    Examples: corrupted PDF files, layout errors, validation structure mismatches.
    """
    pass

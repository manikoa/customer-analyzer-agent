"""
Core Exceptions
Custom exception classes for the Customer Feedback Analyzer Agent
"""

from typing import Optional, Dict, Any


# ============================================================================
# BASE EXCEPTIONS
# ============================================================================

class FeedbackAnalyzerError(Exception):
    """Base exception for all Feedback Analyzer errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional additional details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(FeedbackAnalyzerError):
    """Raised when there's a configuration error."""
    pass


class InvalidProviderError(ConfigurationError):
    """Raised when an invalid LLM provider is specified."""
    pass


class MissingAPIKeyError(ConfigurationError):
    """Raised when a required API key is missing."""
    pass


# ============================================================================
# DATA EXCEPTIONS
# ============================================================================

class DataError(FeedbackAnalyzerError):
    """Base exception for data-related errors."""
    pass


class DataLoadError(DataError):
    """Raised when data loading fails."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class EmptyDatasetError(DataError):
    """Raised when dataset is empty."""
    pass


# ============================================================================
# LLM EXCEPTIONS
# ============================================================================

class LLMError(FeedbackAnalyzerError):
    """Base exception for LLM-related errors."""
    pass


class LLMProviderError(LLMError):
    """Raised when LLM provider encounters an error."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or malformed."""
    pass


# ============================================================================
# WORKFLOW EXCEPTIONS
# ============================================================================

class WorkflowError(FeedbackAnalyzerError):
    """Base exception for workflow-related errors."""
    pass


class NodeExecutionError(WorkflowError):
    """Raised when a workflow node fails to execute."""
    
    def __init__(self, node_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize with node name.
        
        Args:
            node_name: Name of the failed node
            message: Error message
            details: Optional additional details
        """
        self.node_name = node_name
        full_message = f"Node '{node_name}' failed: {message}"
        super().__init__(full_message, details)


class StateValidationError(WorkflowError):
    """Raised when workflow state validation fails."""
    pass


class TriageError(WorkflowError):
    """Raised when triage logic encounters an error."""
    pass


# ============================================================================
# CHAIN EXCEPTIONS
# ============================================================================

class ChainError(FeedbackAnalyzerError):
    """Base exception for LangChain LCEL chain errors."""
    pass


class SentimentChainError(ChainError):
    """Raised when sentiment analysis chain fails."""
    pass


class CategoryChainError(ChainError):
    """Raised when category classification chain fails."""
    pass


class ReportChainError(ChainError):
    """Raised when report generation chain fails."""
    pass


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(FeedbackAnalyzerError):
    """Base exception for validation errors."""
    pass


class ModelValidationError(ValidationError):
    """Raised when Pydantic model validation fails."""
    pass


class InputValidationError(ValidationError):
    """Raised when input validation fails."""
    pass


class OutputValidationError(ValidationError):
    """Raised when output validation fails."""
    pass


# ============================================================================
# TOOL EXCEPTIONS
# ============================================================================

class ToolError(FeedbackAnalyzerError):
    """Base exception for tool-related errors."""
    pass


class FileToolError(ToolError):
    """Raised when file tool operation fails."""
    pass


class DatabaseToolError(ToolError):
    """Raised when database tool operation fails."""
    pass


class ExportToolError(ToolError):
    """Raised when export tool operation fails."""
    pass


class NotificationToolError(ToolError):
    """Raised when notification tool operation fails."""
    pass


# ============================================================================
# EXECUTION EXCEPTIONS
# ============================================================================

class ExecutionError(FeedbackAnalyzerError):
    """Base exception for execution-related errors."""
    pass


class TimeoutError(ExecutionError):
    """Raised when execution times out."""
    pass


class RetryExhaustedError(ExecutionError):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, attempts: int, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize with attempt count.
        
        Args:
            attempts: Number of attempts made
            message: Error message
            details: Optional additional details
        """
        self.attempts = attempts
        full_message = f"Failed after {attempts} attempts: {message}"
        super().__init__(full_message, details)


# ============================================================================
# AGENT EXCEPTIONS
# ============================================================================

class AgentError(FeedbackAnalyzerError):
    """Base exception for agent-related errors."""
    pass


class OrchestrationError(AgentError):
    """Raised when workflow orchestration fails."""
    pass


class PostProcessingError(AgentError):
    """Raised when post-processing fails."""
    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def wrap_exception(
    original_exception: Exception,
    custom_exception_class: type,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> FeedbackAnalyzerError:
    """
    Wrap an original exception in a custom exception class.
    
    Args:
        original_exception: The original exception
        custom_exception_class: The custom exception class to wrap with
        message: Custom error message
        details: Optional additional details
    
    Returns:
        Custom exception instance
    
    Example:
        >>> try:
        ...     # some operation
        ... except ValueError as e:
        ...     raise wrap_exception(e, DataValidationError, "Invalid data format")
    """
    error_details = details or {}
    error_details["original_error"] = str(original_exception)
    error_details["original_type"] = type(original_exception).__name__
    
    return custom_exception_class(message, error_details)


# ============================================================================
# ERROR HELPERS
# ============================================================================

def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error: The exception to check
    
    Returns:
        True if the error is retryable, False otherwise
    
    Example:
        >>> if is_retryable_error(error):
        ...     # retry the operation
    """
    retryable_types = (
        LLMTimeoutError,
        LLMRateLimitError,
        TimeoutError,
        # Network-related errors
    )
    
    return isinstance(error, retryable_types)


def get_error_severity(error: Exception) -> str:
    """
    Get the severity level of an error.
    
    Args:
        error: The exception to evaluate
    
    Returns:
        Severity level: 'low', 'medium', 'high', or 'critical'
    
    Example:
        >>> severity = get_error_severity(error)
        >>> if severity == 'critical':
        ...     # alert immediately
    """
    critical_errors = (
        DataLoadError,
        MissingAPIKeyError,
        StateValidationError
    )
    
    high_errors = (
        LLMError,
        WorkflowError,
        NodeExecutionError
    )
    
    medium_errors = (
        ValidationError,
        ToolError
    )
    
    if isinstance(error, critical_errors):
        return "critical"
    elif isinstance(error, high_errors):
        return "high"
    elif isinstance(error, medium_errors):
        return "medium"
    else:
        return "low"


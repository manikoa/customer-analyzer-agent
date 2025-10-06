"""
Core Constants
Centralized constants used throughout the application
"""

from typing import List

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Supported LLM providers
SUPPORTED_PROVIDERS: List[str] = ["gemini", "openai", "anthropic"]

# Default models for each provider
DEFAULT_MODELS = {
    "gemini": "gemini-1.5-flash",
    "openai": "gpt-4",
    "anthropic": "claude-3-5-sonnet-20241022"
}

# Temperature ranges
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0
DEFAULT_TEMPERATURE = 0.0

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

# Sentiment classifications
SENTIMENT_POSITIVE = "positive"
SENTIMENT_NEUTRAL = "neutral"
SENTIMENT_NEGATIVE = "negative"

SENTIMENTS: List[str] = [SENTIMENT_POSITIVE, SENTIMENT_NEUTRAL, SENTIMENT_NEGATIVE]

# Sentiment score ranges
SENTIMENT_SCORE_MIN = 1
SENTIMENT_SCORE_MAX = 10

# Sentiment thresholds (score ranges)
NEGATIVE_SCORE_MAX = 3
NEUTRAL_SCORE_MIN = 4
NEUTRAL_SCORE_MAX = 7
POSITIVE_SCORE_MIN = 8

# ============================================================================
# CATEGORIES
# ============================================================================

# Feedback categories
CATEGORY_BUG_REPORT = "Bug Report"
CATEGORY_FEATURE_REQUEST = "Feature Request"
CATEGORY_PERFORMANCE = "Performance Issue"
CATEGORY_UI_UX = "UI/UX Feedback"
CATEGORY_PRICING = "Pricing/Billing"
CATEGORY_DOCUMENTATION = "Documentation"
CATEGORY_INTEGRATION = "Integration/API"
CATEGORY_SECURITY = "Security/Privacy"
CATEGORY_SUPPORT = "Customer Support"
CATEGORY_GENERAL = "General Feedback"

CATEGORIES: List[str] = [
    CATEGORY_BUG_REPORT,
    CATEGORY_FEATURE_REQUEST,
    CATEGORY_PERFORMANCE,
    CATEGORY_UI_UX,
    CATEGORY_PRICING,
    CATEGORY_DOCUMENTATION,
    CATEGORY_INTEGRATION,
    CATEGORY_SECURITY,
    CATEGORY_SUPPORT,
    CATEGORY_GENERAL
]

# ============================================================================
# URGENCY AND PRIORITY
# ============================================================================

# Urgency levels
URGENCY_LOW = "low"
URGENCY_MEDIUM = "medium"
URGENCY_HIGH = "high"
URGENCY_CRITICAL = "critical"

URGENCY_LEVELS: List[str] = [
    URGENCY_LOW,
    URGENCY_MEDIUM,
    URGENCY_HIGH,
    URGENCY_CRITICAL
]

# Priority levels (same as urgency for consistency)
PRIORITY_LOW = "low"
PRIORITY_MEDIUM = "medium"
PRIORITY_HIGH = "high"
PRIORITY_CRITICAL = "critical"

PRIORITY_LEVELS: List[str] = [
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_HIGH,
    PRIORITY_CRITICAL
]

# ============================================================================
# WORKFLOW
# ============================================================================

# Workflow statuses
STATUS_INITIALIZED = "initialized"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_PARTIAL = "partial_completion"

WORKFLOW_STATUSES: List[str] = [
    STATUS_INITIALIZED,
    STATUS_PROCESSING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL
]

# Node names
NODE_INITIALIZE = "initialize"
NODE_SENTIMENT = "sentiment_analysis"
NODE_CATEGORY = "category_classification"
NODE_ENRICH = "enrich_feedback"
NODE_ESCALATE = "escalate_handler"
NODE_GROUP = "group_handler"
NODE_REPORT = "report_generation"
NODE_STATISTICS = "compute_statistics"
NODE_VALIDATE = "validate_results"
NODE_ERROR = "error_handler"
NODE_FINALIZE = "finalize"

ALL_NODES: List[str] = [
    NODE_INITIALIZE,
    NODE_SENTIMENT,
    NODE_CATEGORY,
    NODE_ENRICH,
    NODE_ESCALATE,
    NODE_GROUP,
    NODE_REPORT,
    NODE_STATISTICS,
    NODE_VALIDATE,
    NODE_ERROR,
    NODE_FINALIZE
]

# Routing decisions
ROUTE_ESCALATE = "escalate"
ROUTE_GROUP = "group"
ROUTE_REPORT = "report"
ROUTE_ERROR = "error"
ROUTE_CONTINUE = "continue"
ROUTE_END = "end"

# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================

# Processing defaults
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 300

# Feature flags
DEFAULT_ENABLE_DATABASE = True
DEFAULT_ENABLE_EXPORT = True
DEFAULT_ENABLE_NOTIFICATIONS = True

# Paths
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_DB_PATH = "feedback_analysis.db"

# Thresholds
DEFAULT_CRITICAL_THRESHOLD = 0.2  # 20% critical issues
DEFAULT_NEGATIVE_THRESHOLD = 0.5  # 50% negative sentiment

# ============================================================================
# DATA SOURCES
# ============================================================================

# Feedback sources
SOURCE_APP_STORE = "App Store"
SOURCE_GOOGLE_PLAY = "Google Play"
SOURCE_SURVEY = "Survey"
SOURCE_FEATURE_REQUEST = "Feature Request"
SOURCE_SUPPORT_TICKET = "Support Ticket"
SOURCE_EMAIL = "Email"
SOURCE_SOCIAL_MEDIA = "Social Media"
SOURCE_WEBSITE = "Website"

FEEDBACK_SOURCES: List[str] = [
    SOURCE_APP_STORE,
    SOURCE_GOOGLE_PLAY,
    SOURCE_SURVEY,
    SOURCE_FEATURE_REQUEST,
    SOURCE_SUPPORT_TICKET,
    SOURCE_EMAIL,
    SOURCE_SOCIAL_MEDIA,
    SOURCE_WEBSITE
]

# ============================================================================
# EXPORT FORMATS
# ============================================================================

FORMAT_JSON = "json"
FORMAT_CSV = "csv"
FORMAT_MARKDOWN = "markdown"
FORMAT_PDF = "pdf"

EXPORT_FORMATS: List[str] = [
    FORMAT_JSON,
    FORMAT_CSV,
    FORMAT_MARKDOWN,
    FORMAT_PDF
]

# ============================================================================
# VALIDATION
# ============================================================================

# Text constraints
MIN_FEEDBACK_LENGTH = 10
MAX_FEEDBACK_LENGTH = 5000

# List constraints
MIN_KEY_PHRASES = 2
MAX_KEY_PHRASES = 5

MIN_CATEGORIES = 0
MAX_CATEGORIES = 2

MIN_SUCCESS_METRICS = 2
MAX_SUCCESS_METRICS = 4

MIN_KEY_QUOTES = 2
MAX_KEY_QUOTES = 3

# ============================================================================
# ERROR TYPES
# ============================================================================

ERROR_VALIDATION = "validation_error"
ERROR_LLM = "llm_error"
ERROR_PROCESSING = "processing_error"
ERROR_TIMEOUT = "timeout_error"
ERROR_NETWORK = "network_error"
ERROR_UNKNOWN = "unknown_error"

ERROR_TYPES: List[str] = [
    ERROR_VALIDATION,
    ERROR_LLM,
    ERROR_PROCESSING,
    ERROR_TIMEOUT,
    ERROR_NETWORK,
    ERROR_UNKNOWN
]

# ============================================================================
# DISPLAY/FORMATTING
# ============================================================================

# Console formatting
SEPARATOR_LENGTH = 80
SEPARATOR_CHAR = "="
SUBSEPARATOR_CHAR = "-"

# Progress indicators
PROGRESS_SYMBOLS = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "processing": "🔄",
    "thinking": "🤔",
    "complete": "🎉"
}


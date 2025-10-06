"""
Core Package
Contains the core business logic and data structures
"""

# Constants and exceptions
from .constants import *
from .exceptions import *

# Data models
from .models import (
    SentimentResult,
    CategoryResult,
    ReportItem,
    ActionPlanReport,
    SentimentType,
    FeedbackCategory,
    UrgencyLevel,
    PriorityLevel,
    ImpactArea,
    TimeFrame
)

# State management  
from .state import (
    FeedbackAnalysisState,
    WorkflowStatus,
    ErrorType,
    create_initial_state,
    update_workflow_status,
    add_error,
    validate_state,
    get_state_summary,
    serialize_state_for_export
)

# Workflow nodes
from .nodes import (
    initialize_node,
    sentiment_analysis_node,
    category_classification_node,
    enrich_feedback_node,
    report_generation_node,
    compute_statistics_node,
    validate_results_node,
    error_handler_node,
    finalize_node,
    should_continue,
    get_node_function
)

# Workflow orchestration
from .workflow import (
    triage_feedback,
    create_workflow,
    run_workflow
)

__all__ = [
    # Constants (all exported via *)
    # Exceptions (all exported via *)
    
    # Models
    'SentimentResult',
    'CategoryResult',
    'ReportItem',
    'ActionPlanReport',
    'SentimentType',
    'FeedbackCategory',
    'UrgencyLevel',
    'PriorityLevel',
    'ImpactArea',
    'TimeFrame',
    
    # State
    'FeedbackAnalysisState',
    'WorkflowStatus',
    'ErrorType',
    'create_initial_state',
    'update_workflow_status',
    'add_error',
    'validate_state',
    'get_state_summary',
    'serialize_state_for_export',
    
    # Nodes
    'initialize_node',
    'sentiment_analysis_node',
    'category_classification_node',
    'enrich_feedback_node',
    'report_generation_node',
    'compute_statistics_node',
    'validate_results_node',
    'error_handler_node',
    'finalize_node',
    'should_continue',
    'get_node_function',
    
    # Workflow
    'triage_feedback',
    'create_workflow',
    'run_workflow',
]


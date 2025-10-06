"""
LangGraph State Management
Defines the central shared state object for the feedback analysis workflow
"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated
from datetime import datetime
from enum import Enum

# Import our Pydantic models for type hints
from core.models import SentimentResult, CategoryResult, ReportItem, ActionPlanReport


# ============================================================================
# ENUMS FOR STATE MANAGEMENT
# ============================================================================

class WorkflowStatus(str, Enum):
    """Status of the workflow execution."""
    INITIALIZED = "initialized"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CATEGORY_CLASSIFICATION = "category_classification"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class ErrorType(str, Enum):
    """Types of errors that can occur."""
    VALIDATION_ERROR = "validation_error"
    LLM_ERROR = "llm_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


# ============================================================================
# REDUCER FUNCTIONS FOR LIST FIELDS
# ============================================================================

def add_to_list(existing: List[Any], new: List[Any]) -> List[Any]:
    """
    Reducer function to append new items to existing list.
    
    This is used with Annotated types in LangGraph to handle
    updates to list fields in the state.
    """
    return existing + new


def merge_errors(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Merge error lists, avoiding duplicates."""
    seen = {(e['node'], e['timestamp']) for e in existing}
    merged = existing.copy()
    for error in new:
        key = (error['node'], error['timestamp'])
        if key not in seen:
            merged.append(error)
            seen.add(key)
    return merged


# ============================================================================
# MAIN STATE DEFINITION
# ============================================================================

class FeedbackAnalysisState(TypedDict):
    """
    Central state object for the feedback analysis workflow.
    
    This TypedDict defines all transient data that flows through the
    LangGraph workflow. Each node in the graph can read from and write
    to this state object.
    
    State Flow:
    1. Initialize with raw_feedback_items
    2. Sentiment Analysis → populates sentiment_results
    3. Category Classification → populates category_results
    4. Report Generation → populates action_plan_report
    
    Attributes:
        raw_feedback_items: Input list of raw feedback text strings
        
        sentiment_results: List of SentimentResult objects (one per feedback item)
        category_results: List of CategoryResult objects (one per feedback item)
        
        enriched_feedback: Combined data (text + sentiment + category)
        
        action_plan_report: Final ActionPlanReport with action items
        
        workflow_status: Current stage of the workflow
        current_node: Name of the currently executing node
        
        llm_provider: Which LLM provider to use (gemini, openai, anthropic)
        llm_model: Optional specific model name
        temperature: Temperature setting for LLM calls
        
        metadata: Additional metadata about the execution
        errors: List of errors encountered during execution
        
        start_time: When the workflow started
        end_time: When the workflow completed
        total_items: Total number of feedback items being processed
        processed_items: Number of items successfully processed so far
    """
    
    # ========================================================================
    # INPUT DATA
    # ========================================================================
    
    raw_feedback_items: List[str]
    """Raw feedback text strings to analyze"""
    
    # ========================================================================
    # ANALYSIS RESULTS (populated by nodes)
    # ========================================================================
    
    sentiment_results: Annotated[List[Optional[SentimentResult]], add_to_list]
    """Sentiment analysis results for each feedback item"""
    
    category_results: Annotated[List[Optional[CategoryResult]], add_to_list]
    """Category classification results for each feedback item"""
    
    enriched_feedback: Annotated[List[Dict[str, Any]], add_to_list]
    """Combined feedback with all analysis metadata"""
    
    action_plan_report: Optional[ActionPlanReport]
    """Final action plan report for Product Managers"""
    
    # ========================================================================
    # WORKFLOW CONTROL
    # ========================================================================
    
    workflow_status: WorkflowStatus
    """Current status of the workflow"""
    
    current_node: str
    """Name of the currently executing node"""
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    llm_provider: str
    """LLM provider: 'gemini', 'openai', or 'anthropic'"""
    
    llm_model: Optional[str]
    """Optional specific model name (uses default if not set)"""
    
    temperature: float
    """Temperature for LLM calls (0.0 = deterministic, higher = creative)"""
    
    batch_size: int
    """Number of items to process in parallel (default: 5)"""
    
    # ========================================================================
    # METADATA & TRACKING
    # ========================================================================
    
    metadata: Dict[str, Any]
    """Additional metadata about the workflow execution"""
    
    errors: Annotated[List[Dict[str, Any]], merge_errors]
    """List of errors encountered during execution"""
    
    start_time: Optional[str]
    """ISO timestamp when workflow started"""
    
    end_time: Optional[str]
    """ISO timestamp when workflow completed"""
    
    total_items: int
    """Total number of feedback items to process"""
    
    processed_items: int
    """Number of items successfully processed"""
    
    # ========================================================================
    # STATISTICS (computed during execution)
    # ========================================================================
    
    summary_stats: Optional[Dict[str, Any]]
    """Summary statistics computed from results"""


# ============================================================================
# INDIVIDUAL ITEM STATE (for parallel processing)
# ============================================================================

class FeedbackItemState(TypedDict):
    """
    State for processing a single feedback item.
    
    This is used when processing feedback items in parallel batches.
    Each item maintains its own processing state.
    """
    
    index: int
    """Index of this item in the original list"""
    
    raw_text: str
    """Raw feedback text"""
    
    sentiment_result: Optional[SentimentResult]
    """Sentiment analysis result"""
    
    category_result: Optional[CategoryResult]
    """Category classification result"""
    
    status: str
    """Processing status: pending, processing, completed, failed"""
    
    error: Optional[str]
    """Error message if processing failed"""
    
    retry_count: int
    """Number of retry attempts"""


# ============================================================================
# STATE INITIALIZATION HELPERS
# ============================================================================

def create_initial_state(
    feedback_items: List[str],
    llm_provider: str = "gemini",
    llm_model: Optional[str] = None,
    temperature: float = 0.0,
    batch_size: int = 5,
    metadata: Optional[Dict[str, Any]] = None
) -> FeedbackAnalysisState:
    """
    Create an initial state object for the workflow.
    
    Args:
        feedback_items: List of raw feedback text strings
        llm_provider: LLM provider to use
        llm_model: Optional specific model name
        temperature: Temperature for LLM calls
        batch_size: Number of items to process in parallel
        metadata: Additional metadata
    
    Returns:
        Initialized FeedbackAnalysisState
    
    Example:
        >>> state = create_initial_state(
        ...     feedback_items=["Great app!", "Crashes too much"],
        ...     llm_provider="gemini"
        ... )
    """
    return FeedbackAnalysisState(
        # Input
        raw_feedback_items=feedback_items,
        
        # Results (empty initially)
        sentiment_results=[],
        category_results=[],
        enriched_feedback=[],
        action_plan_report=None,
        
        # Workflow control
        workflow_status=WorkflowStatus.INITIALIZED,
        current_node="start",
        
        # Configuration
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        batch_size=batch_size,
        
        # Metadata
        metadata=metadata or {},
        errors=[],
        start_time=datetime.now().isoformat(),
        end_time=None,
        total_items=len(feedback_items),
        processed_items=0,
        
        # Statistics
        summary_stats=None
    )


def create_item_state(index: int, raw_text: str) -> FeedbackItemState:
    """
    Create a state object for a single feedback item.
    
    Args:
        index: Index in the feedback list
        raw_text: Raw feedback text
    
    Returns:
        Initialized FeedbackItemState
    """
    return FeedbackItemState(
        index=index,
        raw_text=raw_text,
        sentiment_result=None,
        category_result=None,
        status="pending",
        error=None,
        retry_count=0
    )


# ============================================================================
# STATE UPDATE HELPERS
# ============================================================================

def update_workflow_status(
    state: FeedbackAnalysisState,
    status: WorkflowStatus,
    node: str
) -> FeedbackAnalysisState:
    """
    Update workflow status and current node.
    
    Args:
        state: Current state
        status: New workflow status
        node: Current node name
    
    Returns:
        Updated state
    """
    state["workflow_status"] = status
    state["current_node"] = node
    return state


def add_error(
    state: FeedbackAnalysisState,
    node: str,
    error_type: ErrorType,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> FeedbackAnalysisState:
    """
    Add an error to the state.
    
    Args:
        state: Current state
        node: Node where error occurred
        error_type: Type of error
        message: Error message
        details: Additional error details
    
    Returns:
        Updated state
    """
    error_entry = {
        "node": node,
        "type": error_type.value,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "details": details or {}
    }
    
    # Use the merge_errors reducer
    state["errors"] = merge_errors(state["errors"], [error_entry])
    
    return state


def mark_completed(state: FeedbackAnalysisState) -> FeedbackAnalysisState:
    """
    Mark workflow as completed.
    
    Args:
        state: Current state
    
    Returns:
        Updated state
    """
    state["workflow_status"] = WorkflowStatus.COMPLETED
    state["end_time"] = datetime.now().isoformat()
    return state


def increment_processed(
    state: FeedbackAnalysisState,
    count: int = 1
) -> FeedbackAnalysisState:
    """
    Increment the count of processed items.
    
    Args:
        state: Current state
        count: Number of items to increment by
    
    Returns:
        Updated state
    """
    state["processed_items"] = min(
        state["processed_items"] + count,
        state["total_items"]
    )
    return state


# ============================================================================
# STATE VALIDATION
# ============================================================================

def validate_state(state: FeedbackAnalysisState) -> bool:
    """
    Validate that the state is well-formed.
    
    Args:
        state: State to validate
    
    Returns:
        True if valid, raises exception if invalid
    
    Raises:
        ValueError: If state is invalid
    """
    # Check required fields
    if not state.get("raw_feedback_items"):
        raise ValueError("State must have raw_feedback_items")
    
    if not state.get("llm_provider"):
        raise ValueError("State must have llm_provider")
    
    # Check list lengths match
    if state.get("sentiment_results") and state.get("category_results"):
        if len(state["sentiment_results"]) != len(state["category_results"]):
            raise ValueError("Sentiment and category results must have same length")
    
    # Check processed count
    if state["processed_items"] > state["total_items"]:
        raise ValueError(f"Processed items ({state['processed_items']}) cannot exceed total ({state['total_items']})")
    
    return True


def get_state_summary(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Get a summary of the current state.
    
    Args:
        state: Current state
    
    Returns:
        Dictionary with state summary
    """
    return {
        "status": state["workflow_status"],
        "current_node": state["current_node"],
        "total_items": state["total_items"],
        "processed_items": state["processed_items"],
        "progress_percentage": (state["processed_items"] / state["total_items"] * 100) if state["total_items"] > 0 else 0,
        "sentiment_results_count": len(state.get("sentiment_results", [])),
        "category_results_count": len(state.get("category_results", [])),
        "errors_count": len(state.get("errors", [])),
        "has_report": state.get("action_plan_report") is not None,
        "llm_provider": state["llm_provider"],
        "started_at": state.get("start_time"),
        "ended_at": state.get("end_time")
    }


# ============================================================================
# STATE EXPORT/SERIALIZATION
# ============================================================================

def serialize_state_for_export(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Serialize state for export (e.g., to JSON).
    
    Converts Pydantic models to dictionaries for serialization.
    
    Args:
        state: State to serialize
    
    Returns:
        Serializable dictionary
    """
    return {
        "raw_feedback_items": state["raw_feedback_items"],
        "sentiment_results": [
            r.model_dump() if r else None 
            for r in state.get("sentiment_results", [])
        ],
        "category_results": [
            r.model_dump() if r else None 
            for r in state.get("category_results", [])
        ],
        "enriched_feedback": state.get("enriched_feedback", []),
        "action_plan_report": (
            state["action_plan_report"].model_dump() 
            if state.get("action_plan_report") else None
        ),
        "workflow_status": state["workflow_status"],
        "current_node": state["current_node"],
        "llm_provider": state["llm_provider"],
        "llm_model": state.get("llm_model"),
        "temperature": state["temperature"],
        "metadata": state.get("metadata", {}),
        "errors": state.get("errors", []),
        "start_time": state.get("start_time"),
        "end_time": state.get("end_time"),
        "total_items": state["total_items"],
        "processed_items": state["processed_items"],
        "summary_stats": state.get("summary_stats")
    }


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the state management."""
    print("\n" + "=" * 80)
    print("LANGGRAPH STATE MANAGEMENT DEMO")
    print("=" * 80)
    
    # Create initial state
    print("\n1️⃣  Creating initial state...")
    state = create_initial_state(
        feedback_items=[
            "The app crashes frequently",
            "Love the new interface!",
            "Support is too slow"
        ],
        llm_provider="gemini",
        temperature=0.0,
        metadata={"source": "app_store", "version": "2.0"}
    )
    
    print(f"   ✅ State initialized with {state['total_items']} feedback items")
    print(f"   Status: {state['workflow_status']}")
    print(f"   Provider: {state['llm_provider']}")
    
    # Validate state
    print("\n2️⃣  Validating state...")
    try:
        validate_state(state)
        print("   ✅ State is valid")
    except ValueError as e:
        print(f"   ❌ State validation failed: {e}")
    
    # Update status
    print("\n3️⃣  Simulating workflow progression...")
    state = update_workflow_status(
        state,
        WorkflowStatus.SENTIMENT_ANALYSIS,
        "sentiment_node"
    )
    print(f"   Status: {state['workflow_status']}")
    print(f"   Node: {state['current_node']}")
    
    # Simulate processing
    state = increment_processed(state, 3)
    print(f"   Processed: {state['processed_items']}/{state['total_items']}")
    
    # Add an error
    print("\n4️⃣  Adding error example...")
    state = add_error(
        state,
        node="sentiment_node",
        error_type=ErrorType.LLM_ERROR,
        message="Rate limit exceeded",
        details={"retry_after": 60}
    )
    print(f"   Errors: {len(state['errors'])}")
    
    # Get summary
    print("\n5️⃣  State summary:")
    summary = get_state_summary(state)
    for key, value in summary.items():
        print(f"   {key:25s}: {value}")
    
    # Serialize for export
    print("\n6️⃣  Serializing state...")
    serialized = serialize_state_for_export(state)
    print(f"   ✅ Serialized {len(serialized)} fields")
    print(f"   Keys: {list(serialized.keys())[:5]}...")
    
    print("\n" + "=" * 80)
    print("✅ STATE MANAGEMENT DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


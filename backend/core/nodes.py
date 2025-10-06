"""
LangGraph Node Functions
Individual Python functions that represent each step in the workflow
and update the shared state object.
"""

from typing import Dict, Any, List
import time
from datetime import datetime

# Import chains
from chains.sentiment import create_sentiment_chain
from chains.category import create_category_chain
from chains.report import generate_action_plan

# Import state management
from core.state import (
    FeedbackAnalysisState,
    WorkflowStatus,
    ErrorType,
    update_workflow_status,
    add_error,
    increment_processed,
    mark_completed
)


# ============================================================================
# INITIALIZATION NODE
# ============================================================================

def initialize_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Initialize the workflow and validate input.
    
    This is the entry point of the workflow graph. It validates
    the input data and prepares the state for processing.
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state dictionary
    """
    print(f"\n{'='*80}")
    print(f"🚀 INITIALIZING WORKFLOW")
    print(f"{'='*80}")
    
    # Validate input
    if not state.get("raw_feedback_items"):
        return add_error(
            state,
            node="initialize_node",
            error_type=ErrorType.VALIDATION_ERROR,
            message="No feedback items provided"
        )
    
    print(f"✅ Initialized with {state['total_items']} feedback items")
    print(f"   Provider: {state['llm_provider']}")
    print(f"   Temperature: {state['temperature']}")
    
    # Update status
    return update_workflow_status(
        state,
        WorkflowStatus.SENTIMENT_ANALYSIS,
        "initialize_node"
    )


# ============================================================================
# SENTIMENT ANALYSIS NODE
# ============================================================================

def sentiment_analysis_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Perform sentiment analysis on all feedback items.
    
    This node:
    1. Creates a sentiment analysis chain
    2. Processes each feedback item
    3. Stores results in sentiment_results
    4. Updates processed count
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with sentiment_results populated
    """
    print(f"\n{'='*80}")
    print(f"💭 SENTIMENT ANALYSIS")
    print(f"{'='*80}")
    
    try:
        # Create chain
        chain = create_sentiment_chain(
            provider=state["llm_provider"],
            temperature=state["temperature"],
            model=state.get("llm_model")
        )
        
        feedback_items = state["raw_feedback_items"]
        sentiment_results = []
        errors_count = 0
        
        print(f"Analyzing {len(feedback_items)} items...")
        
        # Process each item
        for i, feedback in enumerate(feedback_items, 1):
            print(f"  [{i}/{len(feedback_items)}] Analyzing...", end="\r")
            
            try:
                result = chain.invoke(feedback)
                sentiment_results.append(result)
                
            except Exception as e:
                print(f"\n  ⚠️  Error on item {i}: {str(e)[:50]}...")
                sentiment_results.append(None)
                errors_count += 1
                
                # Add error to state
                state = add_error(
                    state,
                    node="sentiment_analysis_node",
                    error_type=ErrorType.LLM_ERROR,
                    message=f"Failed to analyze item {i}",
                    details={"error": str(e), "feedback": feedback[:100]}
                )
        
        print(f"\n✅ Sentiment analysis complete: {len(sentiment_results) - errors_count}/{len(feedback_items)} successful")
        
        # Update state
        state["sentiment_results"] = sentiment_results
        state = increment_processed(state, len(sentiment_results) - errors_count)
        
        # Move to next stage
        return update_workflow_status(
            state,
            WorkflowStatus.CATEGORY_CLASSIFICATION,
            "sentiment_analysis_node"
        )
        
    except Exception as e:
        print(f"\n❌ Critical error in sentiment analysis: {str(e)}")
        state = add_error(
            state,
            node="sentiment_analysis_node",
            error_type=ErrorType.LLM_ERROR,
            message=f"Critical sentiment analysis error: {str(e)}"
        )
        state["workflow_status"] = WorkflowStatus.FAILED
        return state


# ============================================================================
# CATEGORY CLASSIFICATION NODE
# ============================================================================

def category_classification_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Classify each feedback item into business categories.
    
    This node:
    1. Creates a category classification chain
    2. Processes each feedback item
    3. Stores results in category_results
    4. Updates processed count
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with category_results populated
    """
    print(f"\n{'='*80}")
    print(f"📋 CATEGORY CLASSIFICATION")
    print(f"{'='*80}")
    
    try:
        # Create chain
        chain = create_category_chain(
            provider=state["llm_provider"],
            temperature=state["temperature"],
            model=state.get("llm_model")
        )
        
        feedback_items = state["raw_feedback_items"]
        category_results = []
        errors_count = 0
        
        print(f"Categorizing {len(feedback_items)} items...")
        
        # Process each item
        for i, feedback in enumerate(feedback_items, 1):
            print(f"  [{i}/{len(feedback_items)}] Categorizing...", end="\r")
            
            try:
                result = chain.invoke(feedback)
                category_results.append(result)
                
            except Exception as e:
                print(f"\n  ⚠️  Error on item {i}: {str(e)[:50]}...")
                category_results.append(None)
                errors_count += 1
                
                # Add error to state
                state = add_error(
                    state,
                    node="category_classification_node",
                    error_type=ErrorType.LLM_ERROR,
                    message=f"Failed to categorize item {i}",
                    details={"error": str(e), "feedback": feedback[:100]}
                )
        
        print(f"\n✅ Categorization complete: {len(category_results) - errors_count}/{len(feedback_items)} successful")
        
        # Update state
        state["category_results"] = category_results
        
        # Move to next stage
        return update_workflow_status(
            state,
            WorkflowStatus.REPORT_GENERATION,
            "category_classification_node"
        )
        
    except Exception as e:
        print(f"\n❌ Critical error in category classification: {str(e)}")
        state = add_error(
            state,
            node="category_classification_node",
            error_type=ErrorType.LLM_ERROR,
            message=f"Critical categorization error: {str(e)}"
        )
        state["workflow_status"] = WorkflowStatus.FAILED
        return state


# ============================================================================
# DATA ENRICHMENT NODE
# ============================================================================

def enrich_feedback_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Combine sentiment and category results with raw feedback.
    
    This node merges all analysis results into enriched_feedback list
    for easier consumption by the report generation node.
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with enriched_feedback populated
    """
    print(f"\n{'='*80}")
    print(f"🔗 ENRICHING FEEDBACK DATA")
    print(f"{'='*80}")
    
    try:
        feedback_items = state["raw_feedback_items"]
        sentiment_results = state.get("sentiment_results", [])
        category_results = state.get("category_results", [])
        
        enriched_feedback = []
        
        for i, feedback in enumerate(feedback_items):
            sentiment = sentiment_results[i] if i < len(sentiment_results) else None
            category = category_results[i] if i < len(category_results) else None
            
            # Skip items with no valid results
            if not sentiment or not category:
                continue
            
            enriched_item = {
                'text': feedback,
                'category': category.primary_category,
                'urgency': category.urgency,
                'sentiment': sentiment.sentiment,
                'sentiment_score': sentiment.score,
                'confidence': (sentiment.confidence + category.confidence) / 2,
                'key_phrases': category.key_phrases,
                'reasoning': {
                    'sentiment': sentiment.reasoning,
                    'category': category.reasoning
                }
            }
            
            enriched_feedback.append(enriched_item)
        
        print(f"✅ Enriched {len(enriched_feedback)} feedback items")
        
        state["enriched_feedback"] = enriched_feedback
        
        return state
        
    except Exception as e:
        print(f"\n❌ Error enriching feedback: {str(e)}")
        state = add_error(
            state,
            node="enrich_feedback_node",
            error_type=ErrorType.UNKNOWN_ERROR,
            message=f"Failed to enrich feedback: {str(e)}"
        )
        return state


# ============================================================================
# REPORT GENERATION NODE
# ============================================================================

def report_generation_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Generate comprehensive action plan report for Product Managers.
    
    This node:
    1. Uses enriched feedback data
    2. Calls report generation chain
    3. Stores ActionPlanReport in state
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with action_plan_report populated
    """
    print(f"\n{'='*80}")
    print(f"📝 GENERATING ACTION PLAN REPORT")
    print(f"{'='*80}")
    
    try:
        enriched_feedback = state.get("enriched_feedback", [])
        
        if not enriched_feedback:
            print("⚠️  No enriched feedback available, using raw data...")
            # Fallback to raw feedback with minimal metadata
            feedback_items = state["raw_feedback_items"]
            enriched_feedback = [
                {'text': fb, 'category': 'General Feedback', 'urgency': 'medium'}
                for fb in feedback_items
            ]
        
        print(f"Generating report from {len(enriched_feedback)} items...")
        
        # Generate report
        report = generate_action_plan(
            enriched_feedback,
            provider=state["llm_provider"],
            model=state.get("llm_model"),
            verbose=False
        )
        
        print(f"✅ Report generated with {len(report.action_items)} action items")
        print(f"   Quick Wins: {len(report.quick_wins)}")
        print(f"   Strategic Initiatives: {len(report.strategic_initiatives)}")
        
        # Update state
        state["action_plan_report"] = report
        
        # Mark as completed
        return mark_completed(state)
        
    except Exception as e:
        print(f"\n❌ Error generating report: {str(e)}")
        state = add_error(
            state,
            node="report_generation_node",
            error_type=ErrorType.LLM_ERROR,
            message=f"Failed to generate report: {str(e)}"
        )
        state["workflow_status"] = WorkflowStatus.FAILED
        return state


# ============================================================================
# SUMMARY STATISTICS NODE
# ============================================================================

def compute_statistics_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Compute summary statistics from analysis results.
    
    This node calculates various metrics and distributions that
    can be used for reporting and analytics.
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with summary_stats populated
    """
    print(f"\n{'='*80}")
    print(f"📊 COMPUTING STATISTICS")
    print(f"{'='*80}")
    
    try:
        from collections import Counter
        
        sentiment_results = [r for r in state.get("sentiment_results", []) if r]
        category_results = [r for r in state.get("category_results", []) if r]
        
        # Sentiment statistics
        sentiment_distribution = Counter([s.sentiment for s in sentiment_results])
        avg_sentiment_score = sum(s.score for s in sentiment_results) / len(sentiment_results) if sentiment_results else 0
        avg_sentiment_confidence = sum(s.confidence for s in sentiment_results) / len(sentiment_results) if sentiment_results else 0
        
        # Category statistics
        category_distribution = Counter([c.primary_category for c in category_results])
        urgency_distribution = Counter([c.urgency for c in category_results])
        avg_category_confidence = sum(c.confidence for c in category_results) / len(category_results) if category_results else 0
        
        # Critical items
        critical_items = [c for c in category_results if c.urgency in ['critical', 'high']]
        
        summary_stats = {
            'total_items': state['total_items'],
            'processed_items': state['processed_items'],
            'success_rate': (state['processed_items'] / state['total_items'] * 100) if state['total_items'] > 0 else 0,
            'sentiment_distribution': dict(sentiment_distribution),
            'category_distribution': dict(category_distribution),
            'urgency_distribution': dict(urgency_distribution),
            'avg_sentiment_score': round(avg_sentiment_score, 2),
            'avg_sentiment_confidence': round(avg_sentiment_confidence, 3),
            'avg_category_confidence': round(avg_category_confidence, 3),
            'critical_items_count': len(critical_items),
            'critical_items_percentage': round(len(critical_items) / len(category_results) * 100, 1) if category_results else 0,
            'errors_count': len(state.get('errors', []))
        }
        
        state["summary_stats"] = summary_stats
        
        print(f"✅ Statistics computed")
        print(f"   Success Rate: {summary_stats['success_rate']:.1f}%")
        print(f"   Avg Sentiment: {summary_stats['avg_sentiment_score']:.1f}/10")
        print(f"   Critical Items: {summary_stats['critical_items_count']}")
        
        return state
        
    except Exception as e:
        print(f"\n⚠️  Error computing statistics: {str(e)}")
        state = add_error(
            state,
            node="compute_statistics_node",
            error_type=ErrorType.UNKNOWN_ERROR,
            message=f"Failed to compute statistics: {str(e)}"
        )
        return state


# ============================================================================
# VALIDATION NODE
# ============================================================================

def validate_results_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Validate that results meet quality thresholds.
    
    This node checks:
    - Minimum success rate
    - Result completeness
    - Data consistency
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state (possibly with validation errors)
    """
    print(f"\n{'='*80}")
    print(f"✅ VALIDATING RESULTS")
    print(f"{'='*80}")
    
    try:
        sentiment_results = state.get("sentiment_results", [])
        category_results = state.get("category_results", [])
        
        # Check if we have results
        if not sentiment_results or not category_results:
            state = add_error(
                state,
                node="validate_results_node",
                error_type=ErrorType.VALIDATION_ERROR,
                message="No analysis results found"
            )
            return state
        
        # Check success rate
        successful_sentiment = len([r for r in sentiment_results if r])
        successful_category = len([r for r in category_results if r])
        
        success_rate = min(successful_sentiment, successful_category) / len(sentiment_results) * 100
        
        print(f"   Sentiment Success: {successful_sentiment}/{len(sentiment_results)} ({successful_sentiment/len(sentiment_results)*100:.1f}%)")
        print(f"   Category Success: {successful_category}/{len(category_results)} ({successful_category/len(category_results)*100:.1f}%)")
        print(f"   Overall Success: {success_rate:.1f}%")
        
        # Warn if success rate is low
        if success_rate < 80:
            print(f"   ⚠️  Warning: Success rate below 80%")
            state = add_error(
                state,
                node="validate_results_node",
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Low success rate: {success_rate:.1f}%",
                details={"threshold": 80, "actual": success_rate}
            )
        
        # Check length consistency
        if len(sentiment_results) != len(category_results):
            state = add_error(
                state,
                node="validate_results_node",
                error_type=ErrorType.VALIDATION_ERROR,
                message="Result count mismatch between sentiment and category",
                details={
                    "sentiment_count": len(sentiment_results),
                    "category_count": len(category_results)
                }
            )
        
        print("✅ Validation complete")
        
        return state
        
    except Exception as e:
        print(f"\n⚠️  Error during validation: {str(e)}")
        state = add_error(
            state,
            node="validate_results_node",
            error_type=ErrorType.VALIDATION_ERROR,
            message=f"Validation failed: {str(e)}"
        )
        return state


# ============================================================================
# ERROR HANDLING NODE
# ============================================================================

def error_handler_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Handle errors and determine if workflow should continue or fail.
    
    This node analyzes errors and decides whether to:
    - Continue with partial results
    - Retry failed operations
    - Fail the workflow
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with error handling decisions
    """
    print(f"\n{'='*80}")
    print(f"⚠️  ERROR HANDLER")
    print(f"{'='*80}")
    
    errors = state.get("errors", [])
    
    if not errors:
        print("✅ No errors detected")
        return state
    
    print(f"Found {len(errors)} error(s):")
    
    critical_errors = 0
    
    for i, error in enumerate(errors, 1):
        print(f"  {i}. [{error['type']}] {error['node']}: {error['message']}")
        
        # Count critical errors
        if error['type'] in [ErrorType.CONFIGURATION_ERROR.value, ErrorType.LLM_ERROR.value]:
            critical_errors += 1
    
    # Decide on action
    if critical_errors > len(errors) * 0.5:
        print(f"\n❌ Too many critical errors ({critical_errors}), marking workflow as failed")
        state["workflow_status"] = WorkflowStatus.FAILED
    else:
        print(f"\n⚠️  Non-critical errors present, continuing with partial results")
    
    return state


# ============================================================================
# FINALIZATION NODE
# ============================================================================

def finalize_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Finalize the workflow and prepare output.
    
    This node:
    - Sets end_time
    - Computes final statistics
    - Prepares summary
    
    Args:
        state: Current workflow state
    
    Returns:
        Final state with all processing complete
    """
    print(f"\n{'='*80}")
    print(f"🏁 FINALIZING WORKFLOW")
    print(f"{'='*80}")
    
    # Set end time if not already set
    if not state.get("end_time"):
        state["end_time"] = datetime.now().isoformat()
    
    # Compute execution time
    if state.get("start_time") and state.get("end_time"):
        start = datetime.fromisoformat(state["start_time"])
        end = datetime.fromisoformat(state["end_time"])
        duration = (end - start).total_seconds()
        
        state["metadata"]["execution_time_seconds"] = duration
        print(f"⏱️  Execution time: {duration:.2f} seconds")
    
    # Print summary
    report = state.get("action_plan_report")
    if report:
        print(f"📋 Report Summary:")
        print(f"   Title: {report.report_title}")
        print(f"   Action Items: {len(report.action_items)}")
        print(f"   Overall Sentiment: {report.overall_sentiment}")
    
    stats = state.get("summary_stats")
    if stats:
        print(f"\n📊 Final Statistics:")
        print(f"   Processed: {stats['processed_items']}/{stats['total_items']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        print(f"   Avg Sentiment: {stats['avg_sentiment_score']:.1f}/10")
    
    errors = state.get("errors", [])
    if errors:
        print(f"\n⚠️  Errors: {len(errors)}")
    
    print(f"\n✅ Workflow Status: {state['workflow_status']}")
    print(f"{'='*80}\n")
    
    return state


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def should_continue(state: FeedbackAnalysisState) -> str:
    """
    Decision function for conditional edges in LangGraph.
    
    Determines the next node based on workflow status.
    
    Args:
        state: Current workflow state
    
    Returns:
        Name of the next node to execute
    """
    status = state.get("workflow_status")
    
    if status == WorkflowStatus.FAILED:
        return "error_handler"
    elif status == WorkflowStatus.COMPLETED:
        return "finalize"
    else:
        return "continue"


def get_node_function(node_name: str):
    """
    Get node function by name.
    
    Args:
        node_name: Name of the node
    
    Returns:
        Node function
    
    Example:
        >>> func = get_node_function("sentiment_analysis_node")
        >>> updated_state = func(current_state)
    """
    node_map = {
        "initialize": initialize_node,
        "sentiment_analysis": sentiment_analysis_node,
        "category_classification": category_classification_node,
        "enrich_feedback": enrich_feedback_node,
        "report_generation": report_generation_node,
        "compute_statistics": compute_statistics_node,
        "validate_results": validate_results_node,
        "error_handler": error_handler_node,
        "finalize": finalize_node
    }
    
    return node_map.get(node_name)


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the node functions."""
    from core.state import create_initial_state
    
    print("\n" + "=" * 80)
    print("NODE FUNCTIONS DEMO")
    print("=" * 80)
    
    # Create initial state
    feedback_items = [
        "The app crashes every time I export data. Completely broken!",
        "Love the new design! So much cleaner and easier to use.",
        "Customer support is terrible. Been waiting 5 days for a response."
    ]
    
    state = create_initial_state(
        feedback_items=feedback_items,
        llm_provider="gemini",
        temperature=0.0
    )
    
    print(f"\n📋 Testing node functions with {len(feedback_items)} feedback items")
    print(f"   (Note: This demo shows node structure without actual LLM calls)")
    
    # Test each node (without actually calling LLMs)
    print(f"\n{'─'*80}")
    print("NODE EXECUTION FLOW:")
    print(f"{'─'*80}")
    
    nodes = [
        ("initialize", initialize_node),
        ("sentiment_analysis", "sentiment_analysis_node"),
        ("category_classification", "category_classification_node"),
        ("enrich_feedback", enrich_feedback_node),
        ("report_generation", "report_generation_node"),
        ("compute_statistics", compute_statistics_node),
        ("validate_results", validate_results_node),
        ("finalize", finalize_node)
    ]
    
    for i, (name, node) in enumerate(nodes, 1):
        print(f"\n{i}. {name.upper()}")
        print(f"   Function: {node if isinstance(node, str) else node.__name__}")
        if callable(node):
            print(f"   Purpose: {node.__doc__.split('.')[0] if node.__doc__ else 'N/A'}")
    
    print(f"\n{'─'*80}")
    print("✅ All node functions defined and ready")
    print(f"{'─'*80}\n")


if __name__ == "__main__":
    main()


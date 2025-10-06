"""
LangGraph Workflow - The Brain of the Customer Feedback Analyzer Agent
Implements dynamic routing and triage logic with conditional edges
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

# Import state and nodes
from core.state import (
    FeedbackAnalysisState,
    WorkflowStatus,
    create_initial_state,
    get_state_summary
)

from core.nodes import (
    initialize_node,
    sentiment_analysis_node,
    category_classification_node,
    enrich_feedback_node,
    report_generation_node,
    compute_statistics_node,
    validate_results_node,
    error_handler_node,
    finalize_node
)


# ============================================================================
# TRIAGE AND ROUTING LOGIC
# ============================================================================

def triage_feedback(state: FeedbackAnalysisState) -> str:
    """
    Core triage logic - The Brain of the Agent.
    
    Examines priority derived from sentiment and category to determine
    the next workflow path:
    - ESCALATE: High-priority issues requiring immediate attention
    - GROUP: Medium-priority issues to be grouped for analysis
    - END: Low-priority or completed processing
    
    Args:
        state: Current workflow state
    
    Returns:
        Next node name: "escalate", "group", or "end"
    """
    print(f"\n{'='*80}")
    print(f"🧠 TRIAGE LOGIC - ANALYZING PRIORITY")
    print(f"{'='*80}")
    
    # Check if we have results to triage
    sentiment_results = state.get("sentiment_results", [])
    category_results = state.get("category_results", [])
    
    if not sentiment_results or not category_results:
        print("⚠️  No results to triage - proceeding to end")
        return "end"
    
    # Calculate priority metrics
    critical_count = 0
    high_priority_count = 0
    negative_sentiment_count = 0
    avg_sentiment = 0
    
    valid_sentiments = [s for s in sentiment_results if s]
    valid_categories = [c for c in category_results if c]
    
    if valid_sentiments:
        avg_sentiment = sum(s.score for s in valid_sentiments) / len(valid_sentiments)
        negative_sentiment_count = len([s for s in valid_sentiments if s.sentiment == "negative"])
    
    if valid_categories:
        critical_count = len([c for c in valid_categories if c.urgency == "critical"])
        high_priority_count = len([c for c in valid_categories if c.urgency in ["critical", "high"]])
    
    total_items = len(sentiment_results)
    
    print(f"\n📊 Triage Metrics:")
    print(f"   Total Items: {total_items}")
    print(f"   Critical Issues: {critical_count}")
    print(f"   High Priority: {high_priority_count}")
    print(f"   Negative Sentiment: {negative_sentiment_count}")
    print(f"   Avg Sentiment Score: {avg_sentiment:.1f}/10")
    
    # Decision Logic
    critical_threshold = 0.2  # 20% critical issues
    high_priority_threshold = 0.4  # 40% high priority
    negative_threshold = 0.5  # 50% negative sentiment
    
    critical_ratio = critical_count / total_items if total_items > 0 else 0
    high_priority_ratio = high_priority_count / total_items if total_items > 0 else 0
    negative_ratio = negative_sentiment_count / total_items if total_items > 0 else 0
    
    # ESCALATE: Critical issues or very negative feedback
    if critical_count > 0 or critical_ratio >= critical_threshold:
        print(f"\n🔴 DECISION: ESCALATE")
        print(f"   Reason: {critical_count} critical issue(s) detected")
        print(f"   Action: Immediate attention required")
        return "escalate"
    
    # ESCALATE: High proportion of high-priority or negative feedback
    if (high_priority_ratio >= high_priority_threshold or 
        negative_ratio >= negative_threshold or 
        avg_sentiment < 4.0):
        print(f"\n🟠 DECISION: ESCALATE")
        print(f"   Reason: High priority ratio ({high_priority_ratio:.1%}) or negative sentiment")
        print(f"   Action: Requires prioritized handling")
        return "escalate"
    
    # GROUP: Medium priority - analyze patterns and group similar issues
    if high_priority_count > 0 or negative_sentiment_count > 0:
        print(f"\n🟡 DECISION: GROUP")
        print(f"   Reason: Medium priority issues present")
        print(f"   Action: Group and analyze patterns")
        return "group"
    
    # END: Mostly positive feedback, proceed to completion
    print(f"\n🟢 DECISION: PROCEED TO END")
    print(f"   Reason: Low priority, mostly positive feedback")
    print(f"   Action: Standard processing")
    return "end"


def route_after_triage(state: FeedbackAnalysisState) -> Literal["escalate_handler", "group_handler", "report_generation"]:
    """
    Route feedback after triage analysis.
    
    This is the conditional edge function used by LangGraph.
    
    Args:
        state: Current workflow state
    
    Returns:
        Next node name for conditional routing
    """
    decision = triage_feedback(state)
    
    if decision == "escalate":
        return "escalate_handler"
    elif decision == "group":
        return "group_handler"
    else:
        return "report_generation"


def should_validate(state: FeedbackAnalysisState) -> Literal["validate_results", "finalize"]:
    """
    Determine if validation is needed.
    
    Args:
        state: Current workflow state
    
    Returns:
        Next node for validation routing
    """
    # Always validate if we have results
    if state.get("action_plan_report"):
        return "validate_results"
    return "finalize"


def check_errors(state: FeedbackAnalysisState) -> Literal["error_handler", "compute_statistics"]:
    """
    Check if there are errors that need handling.
    
    Args:
        state: Current workflow state
    
    Returns:
        Next node based on error state
    """
    errors = state.get("errors", [])
    
    # If there are critical errors, handle them
    if len(errors) > 5 or state.get("workflow_status") == WorkflowStatus.FAILED:
        return "error_handler"
    
    return "compute_statistics"


# ============================================================================
# SPECIALIZED HANDLER NODES
# ============================================================================

def escalate_handler_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Handle escalated high-priority feedback.
    
    This node processes critical issues that require immediate attention:
    - Flags urgent items
    - Prepares priority notifications
    - Ensures critical issues are at the top of the report
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with escalation metadata
    """
    print(f"\n{'='*80}")
    print(f"🚨 ESCALATION HANDLER")
    print(f"{'='*80}")
    
    category_results = state.get("category_results", [])
    sentiment_results = state.get("sentiment_results", [])
    raw_feedback = state.get("raw_feedback_items", [])
    
    # Identify critical items
    critical_items = []
    for i, (cat, sent) in enumerate(zip(category_results, sentiment_results)):
        # Bounds check
        if i >= len(raw_feedback):
            break
        if cat and sent:
            if cat.urgency in ["critical", "high"] or sent.score <= 3:
                critical_items.append({
                    "index": i,
                    "feedback": raw_feedback[i],
                    "urgency": cat.urgency,
                    "category": cat.primary_category,
                    "sentiment_score": sent.score
                })
    
    print(f"🔴 Critical Items Identified: {len(critical_items)}")
    
    # Add escalation metadata
    if "metadata" not in state:
        state["metadata"] = {}
    
    state["metadata"]["escalated"] = True
    state["metadata"]["critical_items"] = critical_items
    state["metadata"]["escalation_count"] = len(critical_items)
    state["metadata"]["requires_immediate_attention"] = len([
        item for item in critical_items if item["urgency"] == "critical"
    ])
    
    print(f"   Total Critical: {len(critical_items)}")
    print(f"   Requires Immediate Attention: {state['metadata']['requires_immediate_attention']}")
    
    # Print top critical issues
    if critical_items:
        print(f"\n🔥 Top Critical Issues:")
        for i, item in enumerate(critical_items[:3], 1):
            print(f"   {i}. [{item['urgency'].upper()}] {item['category']}")
            print(f"      Sentiment: {item['sentiment_score']}/10")
            print(f"      Feedback: {item['feedback'][:80]}...")
    
    print(f"\n{'='*80}\n")
    
    return state


def group_handler_node(state: FeedbackAnalysisState) -> Dict[str, Any]:
    """
    Group and analyze similar feedback patterns.
    
    This node identifies patterns in medium-priority feedback:
    - Groups by category
    - Identifies common themes
    - Prepares grouped analysis
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with grouping metadata
    """
    print(f"\n{'='*80}")
    print(f"📦 GROUPING HANDLER")
    print(f"{'='*80}")
    
    from collections import defaultdict
    
    category_results = state.get("category_results", [])
    
    # Group by category
    groups = defaultdict(list)
    for i, cat in enumerate(category_results):
        if cat:
            groups[cat.primary_category].append({
                "index": i,
                "feedback": state["raw_feedback_items"][i],
                "urgency": cat.urgency,
                "key_phrases": cat.key_phrases
            })
    
    print(f"📊 Grouped into {len(groups)} categories:")
    
    grouped_analysis = {}
    for category, items in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        grouped_analysis[category] = {
            "count": len(items),
            "urgency_distribution": {},
            "sample_feedback": [item["feedback"] for item in items[:3]]
        }
        
        # Calculate urgency distribution
        urgency_counts = defaultdict(int)
        for item in items:
            urgency_counts[item["urgency"]] += 1
        grouped_analysis[category]["urgency_distribution"] = dict(urgency_counts)
        
        print(f"   • {category}: {len(items)} items")
        print(f"     Urgency: {dict(urgency_counts)}")
    
    # Add grouping metadata
    if "metadata" not in state:
        state["metadata"] = {}
    
    state["metadata"]["grouped"] = True
    state["metadata"]["group_count"] = len(groups)
    state["metadata"]["grouped_analysis"] = grouped_analysis
    
    print(f"\n✅ Grouping complete")
    print(f"{'='*80}\n")
    
    return state


# ============================================================================
# BUILD THE WORKFLOW GRAPH
# ============================================================================

def create_workflow() -> StateGraph:
    """
    Create the complete LangGraph workflow with dynamic routing.
    
    Workflow Structure:
    1. Initialize → Validate input
    2. Sentiment Analysis → Analyze sentiment
    3. Category Classification → Classify categories
    4. Enrich Feedback → Combine results
    5. **TRIAGE** → Route based on priority:
       - ESCALATE → escalate_handler → Report
       - GROUP → group_handler → Report
       - END → Report directly
    6. Report Generation → Create action plan
    7. Compute Statistics → Calculate metrics
    8. Validate Results → Quality checks
    9. Finalize → Complete workflow
    
    Returns:
        Configured StateGraph ready for compilation
    """
    # Create the graph
    workflow = StateGraph(FeedbackAnalysisState)
    
    # Add all nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("sentiment_analysis", sentiment_analysis_node)
    workflow.add_node("category_classification", category_classification_node)
    workflow.add_node("enrich_feedback", enrich_feedback_node)
    workflow.add_node("escalate_handler", escalate_handler_node)
    workflow.add_node("group_handler", group_handler_node)
    workflow.add_node("report_generation", report_generation_node)
    workflow.add_node("compute_statistics", compute_statistics_node)
    workflow.add_node("validate_results", validate_results_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add sequential edges
    workflow.add_edge("initialize", "sentiment_analysis")
    workflow.add_edge("sentiment_analysis", "category_classification")
    workflow.add_edge("category_classification", "enrich_feedback")
    
    # CONDITIONAL EDGE: Triage routing (The Brain!)
    workflow.add_conditional_edges(
        "enrich_feedback",
        route_after_triage,
        {
            "escalate_handler": "escalate_handler",
            "group_handler": "group_handler",
            "report_generation": "report_generation"
        }
    )
    
    # Both handlers lead to report generation
    workflow.add_edge("escalate_handler", "report_generation")
    workflow.add_edge("group_handler", "report_generation")
    
    # CONDITIONAL EDGE: Error checking
    workflow.add_conditional_edges(
        "report_generation",
        check_errors,
        {
            "error_handler": "error_handler",
            "compute_statistics": "compute_statistics"
        }
    )
    
    # Error handler can retry or continue
    workflow.add_edge("error_handler", "compute_statistics")
    
    # Continue to validation
    workflow.add_edge("compute_statistics", "validate_results")
    
    # CONDITIONAL EDGE: Validation routing
    workflow.add_conditional_edges(
        "validate_results",
        should_validate,
        {
            "validate_results": "finalize",  # Always finalize after validation
            "finalize": "finalize"
        }
    )
    
    # Set finish point
    workflow.add_edge("finalize", END)
    
    return workflow


# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

def run_workflow(
    feedback_items: list[str],
    llm_provider: str = "gemini",
    llm_model: str = None,
    temperature: float = 0.0,
    verbose: bool = True
) -> FeedbackAnalysisState:
    """
    Execute the complete workflow on feedback items.
    
    Args:
        feedback_items: List of raw feedback text strings
        llm_provider: LLM provider (gemini, openai, anthropic)
        llm_model: Optional specific model name
        temperature: LLM temperature
        verbose: Print progress
    
    Returns:
        Final workflow state with all results
    
    Example:
        >>> feedback = ["App crashes", "Love it!", "Too slow"]
        >>> final_state = run_workflow(feedback, provider="gemini")
        >>> report = final_state['action_plan_report']
    """
    if verbose:
        print("\n" + "=" * 80)
        print("🚀 STARTING CUSTOMER FEEDBACK ANALYZER WORKFLOW")
        print("=" * 80)
        print(f"Feedback Items: {len(feedback_items)}")
        print(f"Provider: {llm_provider}")
    
    # Create initial state
    initial_state = create_initial_state(
        feedback_items=feedback_items,
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature
    )
    
    # Build and compile workflow
    workflow_graph = create_workflow()
    app = workflow_graph.compile()
    
    if verbose:
        print("\n✅ Workflow compiled successfully")
        print("🔄 Executing workflow...\n")
    
    # Execute workflow
    final_state = app.invoke(initial_state)
    
    if verbose:
        print("\n" + "=" * 80)
        print("✅ WORKFLOW EXECUTION COMPLETE")
        print("=" * 80)
        
        # Print summary
        summary = get_state_summary(final_state)
        print(f"\n📊 Final Summary:")
        print(f"   Status: {summary['status']}")
        print(f"   Processed: {summary['processed_items']}/{summary['total_items']}")
        print(f"   Success Rate: {summary['progress_percentage']:.1f}%")
        print(f"   Errors: {summary['errors_count']}")
        print(f"   Report Generated: {'Yes' if summary['has_report'] else 'No'}")
        
        if final_state.get("action_plan_report"):
            report = final_state["action_plan_report"]
            print(f"\n📋 Action Plan:")
            print(f"   Title: {report.report_title}")
            print(f"   Action Items: {len(report.action_items)}")
            print(f"   Quick Wins: {len(report.quick_wins)}")
            print(f"   Overall Sentiment: {report.overall_sentiment}")
        
        # Triage results
        if final_state["metadata"].get("escalated"):
            print(f"\n🚨 Escalation Metadata:")
            print(f"   Critical Items: {final_state['metadata']['escalation_count']}")
            print(f"   Immediate Attention: {final_state['metadata']['requires_immediate_attention']}")
        
        if final_state["metadata"].get("grouped"):
            print(f"\n📦 Grouping Metadata:")
            print(f"   Groups: {final_state['metadata']['group_count']}")
        
        print("\n" + "=" * 80 + "\n")
    
    return final_state


def visualize_workflow():
    """
    Generate a visual representation of the workflow graph.
    
    Returns:
        Graph visualization (requires graphviz)
    """
    try:
        workflow = create_workflow()
        app = workflow.compile()
        
        # Try to get the graph
        print("\n📊 Workflow Graph Structure:")
        print("="*80)
        print("\nNodes:")
        print("  • initialize")
        print("  • sentiment_analysis")
        print("  • category_classification")
        print("  • enrich_feedback")
        print("  • escalate_handler (conditional)")
        print("  • group_handler (conditional)")
        print("  • report_generation")
        print("  • compute_statistics")
        print("  • validate_results")
        print("  • error_handler (conditional)")
        print("  • finalize")
        
        print("\nConditional Routing Points:")
        print("  🧠 enrich_feedback → [TRIAGE]")
        print("     ├─ ESCALATE → escalate_handler")
        print("     ├─ GROUP → group_handler")
        print("     └─ END → report_generation")
        print("\n  ⚠️  report_generation → [ERROR CHECK]")
        print("     ├─ Has Errors → error_handler")
        print("     └─ No Errors → compute_statistics")
        
        print("="*80 + "\n")
        
        return app
        
    except Exception as e:
        print(f"⚠️  Could not visualize: {str(e)}")
        return None


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the complete workflow."""
    print("\n" + "=" * 80)
    print("LANGGRAPH WORKFLOW DEMO")
    print("=" * 80)
    
    # Check for API keys
    from utils.llm import get_available_providers, print_provider_status
    
    print("\n🔍 Checking LLM Provider Status...")
    print_provider_status()
    
    available = get_available_providers()
    if not available:
        print("\n⚠️  No LLM providers available!")
        print("\n📝 Workflow structure created successfully.")
        print("   Set an API key to run the full workflow:")
        print("     export GOOGLE_API_KEY='your-key'")
        print("     export OPENAI_API_KEY='your-key'")
        print("\n🎯 Visualizing workflow structure...\n")
        visualize_workflow()
        return
    
    provider = available[0]
    print(f"\n✅ Using provider: {provider.upper()}\n")
    
    # Sample feedback with diverse priorities
    sample_feedback = [
        # Critical issues
        "CRITICAL: The system is down and we're losing money every minute! This is a complete disaster!",
        "App crashes and deletes all my data. I've lost everything. Absolutely furious!",
        
        # High priority
        "The app is extremely slow and frequently freezes. Very frustrating to use daily.",
        "Customer support hasn't responded in a week. This is unacceptable service.",
        
        # Medium priority
        "Would really like to see dark mode added. My eyes hurt after long sessions.",
        "The navigation could be more intuitive. Sometimes I get lost finding features.",
        
        # Low priority / Positive
        "Love the new design! So much cleaner and easier to use.",
        "Great app overall, just minor suggestions for improvement.",
        "The team is doing an excellent job. Keep up the good work!"
    ]
    
    print(f"📋 Sample Feedback ({len(sample_feedback)} items):")
    for i, fb in enumerate(sample_feedback, 1):
        priority_marker = "🔴" if i <= 2 else "🟠" if i <= 4 else "🟡" if i <= 6 else "🟢"
        print(f"   {priority_marker} {i}. {fb[:60]}...")
    
    print("\n" + "─" * 80)
    print("🚀 Running Complete Workflow with Triage Logic...")
    print("─" * 80)
    
    # Run workflow
    final_state = run_workflow(
        feedback_items=sample_feedback,
        llm_provider=provider,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("✅ WORKFLOW DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


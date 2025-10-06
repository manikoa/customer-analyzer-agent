"""
Report Generation Chain using LangChain LCEL
Generates actionable product management reports from categorized feedback
"""

from typing import List, Dict, Any
from langchain_core.output_parsers import PydanticOutputParser

# Import models from centralized location
from core.models import ReportItem, ActionPlanReport, PriorityLevel, ImpactArea, TimeFrame

# Import utilities
from utils.llm import get_llm
from utils.prompts import create_report_prompt


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_feedback_summary(feedback_items: List[Dict[str, Any]]) -> str:
    """
    Prepare a summary of feedback items for the LLM.
    
    Args:
        feedback_items: List of feedback dicts
    
    Returns:
        Formatted string summary
    """
    summary_parts = []
    for i, item in enumerate(feedback_items, 1):
        text = item.get('text', item.get('Raw_Text', 'No text'))
        category = item.get('category', 'Unknown')
        urgency = item.get('urgency', 'medium')
        
        summary_parts.append(f"{i}. [{category}] [{urgency}] {text}")
    
    return "\n".join(summary_parts)


def create_report_chain(provider: str = "gemini", model: str = None):
    """
    Create a chain for generating action plan reports.
    
    Args:
        provider: LLM provider
        model: Optional model name
    
    Returns:
        Runnable chain
    """
    from utils.prompts import create_report_prompt
    llm = get_llm(provider=provider, model=model, temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=ActionPlanReport)
    prompt = create_report_prompt(format_instructions=parser.get_format_instructions())
    
    return prompt | llm | parser


def create_single_item_chain(provider: str = "gemini", model: str = None):
    """
    Create a chain for generating single action items.
    
    Args:
        provider: LLM provider
        model: Optional model name
    
    Returns:
        Runnable chain
    """
    llm = get_llm(provider=provider, model=model, temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=ReportItem)
    
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(
        """Generate a Product Manager action item for this feedback category.

Category: {category}
Urgency: {urgency}
Feedback Count: {count}

Sample Feedback:
{sample_feedback}

{format_instructions}

Provide a detailed action item with:
- Clear title
- Priority level
- Problem statement
- Recommended action
- Success metrics"""
    )
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    
    return prompt | llm | parser


# ============================================================================
# REPORT GENERATION FUNCTIONS
# ============================================================================

def generate_action_plan(
    feedback_items: List[Dict[str, Any]],
    provider: str = "gemini",
    model: str = None,
    verbose: bool = False
) -> ActionPlanReport:
    """
    Generate a complete action plan from categorized feedback.
    
    Args:
        feedback_items: List of feedback dicts with 'text', 'category', 'urgency', etc.
        provider: LLM provider to use
        model: Optional specific model name
        verbose: If True, print progress
    
    Returns:
        ActionPlanReport with prioritized action items
    
    Example:
        >>> feedback = load_categorized_feedback()
        >>> report = generate_action_plan(feedback, provider="gemini")
        >>> print(f"Generated {len(report.action_items)} action items")
    """
    if verbose:
        print(f"📊 Analyzing {len(feedback_items)} feedback items...")
    
    # Prepare summary
    feedback_summary = prepare_feedback_summary(feedback_items)
    
    if verbose:
        print("🤖 Generating action plan with LLM...")
    
    # Create chain and generate report
    chain = create_report_chain(provider=provider, model=model)
    report = chain.invoke(feedback_summary)
    
    if verbose:
        print(f"✅ Generated {len(report.action_items)} prioritized action items")
    
    return report


def generate_single_action_item(
    category: str,
    sample_feedback: List[str],
    urgency: str = "medium",
    provider: str = "gemini",
    model: str = None
) -> ReportItem:
    """
    Generate a single action item for a specific feedback category.
    
    Args:
        category: Feedback category
        sample_feedback: List of sample feedback texts
        urgency: Overall urgency level
        provider: LLM provider
        model: Optional model name
    
    Returns:
        Single ReportItem
    
    Example:
        >>> item = generate_single_action_item(
        ...     category="Bug Report",
        ...     sample_feedback=["App crashes on export", "Can't save my work"],
        ...     urgency="high"
        ... )
    """
    chain = create_single_item_chain(provider=provider, model=model)
    
    input_data = {
        "category": category,
        "count": len(sample_feedback),
        "urgency": urgency,
        "sample_feedback": "\n".join([f"- {fb}" for fb in sample_feedback])
    }
    
    result = chain.invoke(input_data)
    return result


def format_report_markdown(report: ActionPlanReport) -> str:
    """
    Format ActionPlanReport as readable Markdown.
    
    Args:
        report: ActionPlanReport object
    
    Returns:
        Formatted Markdown string
    """
    lines = []
    
    # Header
    lines.append(f"# {report.report_title}")
    lines.append(f"\n**Generated:** {report.generated_at}")
    lines.append(f"**Feedback Analyzed:** {report.total_feedback_analyzed} items")
    lines.append(f"**Overall Sentiment:** {report.overall_sentiment}")
    lines.append("\n---\n")
    
    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append(report.summary)
    lines.append("\n")
    
    # Top Categories
    lines.append("### Top Categories\n")
    for cat in report.top_categories:
        lines.append(f"- {cat}")
    lines.append("\n")
    
    # Quick Wins
    lines.append("### Quick Wins 🎯\n")
    for i, win in enumerate(report.quick_wins, 1):
        lines.append(f"{i}. {win}")
    lines.append("\n")
    
    # Strategic Initiatives
    lines.append("### Strategic Initiatives 🚀\n")
    for i, initiative in enumerate(report.strategic_initiatives, 1):
        lines.append(f"{i}. {initiative}")
    lines.append("\n---\n")
    
    # Action Items
    lines.append("## Prioritized Action Items\n")
    
    # Group by priority
    priority_order = [PriorityLevel.CRITICAL, PriorityLevel.HIGH, PriorityLevel.MEDIUM, PriorityLevel.LOW]
    
    for priority in priority_order:
        items = [item for item in report.action_items if item.priority == priority.value]
        if not items:
            continue
        
        priority_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }
        
        lines.append(f"\n### {priority_emoji[priority.value]} {priority.value.upper()} Priority\n")
        
        for item in items:
            lines.append(f"\n#### {item.title}\n")
            lines.append(f"**Category:** {item.category} | **Effort:** {item.estimated_effort} | **Timeframe:** {item.timeframe}")
            lines.append(f"\n**Problem:**\n{item.problem_statement}\n")
            lines.append(f"**Affected Users:** {item.affected_users} | **Pain Level:** {item.user_pain_level}/10 | **Supporting Feedback:** {item.supporting_feedback_count} items\n")
            lines.append(f"**Recommended Action:**\n{item.recommended_action}\n")
            lines.append(f"**Expected Outcome:**\n{item.expected_outcome}\n")
            lines.append(f"**Impact Areas:** {', '.join(item.impact_areas)}\n")
            lines.append(f"**Success Metrics:**")
            for metric in item.success_metrics:
                lines.append(f"- {metric}")
            lines.append(f"\n**Key Customer Quotes:**")
            for quote in item.key_quotes:
                lines.append(f'> "{quote}"')
            lines.append(f"\n**Risk if Ignored:**\n{item.risk_if_ignored}\n")
            if item.dependencies:
                lines.append(f"**Dependencies:** {', '.join(item.dependencies)}\n")
            lines.append("\n---\n")
    
    return "\n".join(lines)


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the report generation chain."""
    print("\n" + "=" * 80)
    print("PRODUCT MANAGEMENT ACTION PLAN GENERATOR")
    print("=" * 80)
    
    # Sample categorized feedback
    sample_feedback = [
        # Bug Reports
        {"text": "The app crashes every time I try to export my data to CSV. Lost all my work!", 
         "category": "Bug Report", "urgency": "critical"},
        {"text": "Can't log in on mobile. Authentication keeps failing.", 
         "category": "Bug Report", "urgency": "high"},
        {"text": "Dashboard doesn't refresh automatically, have to reload page.", 
         "category": "Bug Report", "urgency": "medium"},
        {"text": "App crashes when I upload large files over 100MB", 
         "category": "Bug Report", "urgency": "high"},
        
        # Feature Requests
        {"text": "Please add dark mode! My eyes hurt after long sessions.", 
         "category": "Feature Request", "urgency": "medium"},
        {"text": "Would love to see bulk editing capabilities for multiple items.", 
         "category": "Feature Request", "urgency": "medium"},
        {"text": "Add keyboard shortcuts for power users.", 
         "category": "Feature Request", "urgency": "low"},
        {"text": "Need integration with Slack for notifications.", 
         "category": "Feature Request", "urgency": "medium"},
        
        # Performance
        {"text": "Dashboard takes 30+ seconds to load. Unacceptably slow.", 
         "category": "Performance", "urgency": "high"},
        {"text": "Search is really laggy with large datasets.", 
         "category": "Performance", "urgency": "medium"},
        
        # User Experience
        {"text": "Navigation is confusing. Can never find the settings.", 
         "category": "User Experience", "urgency": "medium"},
        {"text": "The onboarding process is too long and complicated.", 
         "category": "User Experience", "urgency": "medium"},
        
        # Customer Support
        {"text": "Been waiting 5 days for support response. This is unacceptable!", 
         "category": "Customer Support", "urgency": "high"},
        
        # Pricing
        {"text": "Pricing is too high compared to competitors. Hard to justify the cost.", 
         "category": "Pricing", "urgency": "medium"},
    ]
    
    print(f"\n📊 Analyzing {len(sample_feedback)} feedback items...")
    print("\nFeedback breakdown:")
    
    from collections import Counter
    categories = Counter([item['category'] for item in sample_feedback])
    for cat, count in categories.most_common():
        print(f"  • {cat}: {count} items")
    
    print("\n" + "─" * 80)
    print("🤖 Generating comprehensive action plan...")
    print("─" * 80)
    
    # Generate report
    report = generate_action_plan(sample_feedback, verbose=True)
    
    # Display summary
    print("\n" + "=" * 80)
    print("📋 ACTION PLAN REPORT")
    print("=" * 80)
    
    print(f"\n📊 {report.report_title}")
    print(f"Generated: {report.generated_at}")
    print(f"Feedback analyzed: {report.total_feedback_analyzed} items")
    print(f"Overall sentiment: {report.overall_sentiment.upper()}")
    
    print(f"\n📝 Executive Summary:")
    print(f"{report.summary}")
    
    print(f"\n🎯 Quick Wins:")
    for i, win in enumerate(report.quick_wins, 1):
        print(f"  {i}. {win}")
    
    print(f"\n🚀 Strategic Initiatives:")
    for i, init in enumerate(report.strategic_initiatives, 1):
        print(f"  {i}. {init}")
    
    print(f"\n📌 Top Categories:")
    for cat in report.top_categories:
        print(f"  • {cat}")
    
    print(f"\n" + "=" * 80)
    print(f"📋 PRIORITIZED ACTION ITEMS ({len(report.action_items)} items)")
    print("=" * 80)
    
    for i, item in enumerate(report.action_items, 1):
        priority_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }
        
        print(f"\n{priority_emoji[item.priority]} [{item.priority.upper()}] {i}. {item.title}")
        print(f"{'─' * 78}")
        print(f"Category: {item.category}")
        print(f"Affected: {item.affected_users} | Pain: {item.user_pain_level}/10 | Support: {item.supporting_feedback_count} items")
        print(f"Effort: {item.estimated_effort} | Timeframe: {item.timeframe}")
        print(f"\n💡 Problem:")
        print(f"   {item.problem_statement}")
        print(f"\n✅ Recommendation:")
        print(f"   {item.recommended_action}")
        print(f"\n🎯 Expected Outcome:")
        print(f"   {item.expected_outcome}")
        print(f"\n📈 Success Metrics:")
        for metric in item.success_metrics:
            print(f"   • {metric}")
        print(f"\n⚠️  Risk if Ignored:")
        print(f"   {item.risk_if_ignored}")
    
    # Save as markdown
    markdown = format_report_markdown(report)
    
    output_file = "action_plan_report.md"
    with open(output_file, 'w') as f:
        f.write(markdown)
    
    print(f"\n" + "=" * 80)
    print(f"✅ Report saved to: {output_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Check available providers
    from utils.llm import print_provider_status, get_available_providers
    
    print("\n🔍 Checking LLM Provider Status...")
    print_provider_status()
    
    available = get_available_providers()
    if not available:
        print("\n⚠️  No LLM providers available!")
        print("\nTo use this tool, set one of these API keys:")
        print("  export GOOGLE_API_KEY='your-key'      # For Gemini")
        print("  export OPENAI_API_KEY='your-key'      # For OpenAI")
        print("  export ANTHROPIC_API_KEY='your-key'   # For Claude")
        exit(1)
    
    main()


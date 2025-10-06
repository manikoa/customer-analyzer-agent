"""
Category Classification Chain using LangChain LCEL
Classifies customer feedback into predefined business categories
"""

from typing import List, Dict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import models from centralized location
from core.models import CategoryResult, FeedbackCategory

# Import utilities
from utils.llm import get_llm, get_available_providers
from utils.prompts import create_category_prompt


# ============================================================================
# CATEGORY DESCRIPTIONS FOR LLM
# ============================================================================

# Category descriptions for the LLM
CATEGORY_DESCRIPTIONS = {
    FeedbackCategory.BUG_REPORT: "Technical issues, crashes, errors, broken functionality, unexpected behavior",
    FeedbackCategory.FEATURE_REQUEST: "Requests for new functionality, capabilities, or enhancements",
    FeedbackCategory.USER_EXPERIENCE: "UI/UX feedback, navigation issues, design concerns, usability problems",
    FeedbackCategory.PERFORMANCE: "Speed issues, loading times, responsiveness, lag, optimization",
    FeedbackCategory.CUSTOMER_SUPPORT: "Service quality, response time, help desk, support team interactions",
    FeedbackCategory.PRICING: "Cost concerns, value for money, billing issues, pricing model feedback",
    FeedbackCategory.DOCUMENTATION: "Help content, tutorials, guides, API docs, learning resources",
    FeedbackCategory.SECURITY: "Privacy concerns, data protection, authentication, authorization issues",
    FeedbackCategory.INTEGRATION: "Third-party integrations, API issues, compatibility, interoperability",
    FeedbackCategory.GENERAL_FEEDBACK: "General praise, complaints, or suggestions that don't fit other categories"
}


# ============================================================================
# CHAIN BUILDER
# ============================================================================

def create_category_chain(provider: str = "gemini", temperature: float = 0.0, model: str = None):
    """
    Create the category classification chain using LangChain LCEL.
    
    This chain follows the pattern: Prompt | Model | Parser
    
    Args:
        provider: LLM provider ("openai", "gemini", or "anthropic")
        temperature: Model temperature (0.0 recommended for consistent categorization)
        model: Optional specific model name
    
    Returns:
        Configured LCEL chain that takes feedback_text and returns CategoryResult
    
    Example:
        >>> chain = create_category_chain(provider="gemini")
        >>> result = chain.invoke("The app crashes when I try to export data")
        >>> print(result.primary_category)
        'Bug Report'
    """
    # Initialize components
    llm = get_llm(provider=provider, temperature=temperature, model=model)
    parser = PydanticOutputParser(pydantic_object=CategoryResult)
    
    # Create prompt template
    prompt = create_category_prompt(format_instructions=parser.get_format_instructions())
    
    # Build LCEL chain: Prompt | Model | Parser
    chain = (
        {"feedback_text": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    
    return chain


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def categorize_feedback(
    feedback_text: str,
    provider: str = "gemini",
    model: str = None,
    verbose: bool = False
) -> CategoryResult:
    """
    Convenience function to categorize a single piece of feedback.
    
    Args:
        feedback_text: The customer feedback text to categorize
        provider: LLM provider to use
        model: Optional specific model name
        verbose: If True, print categorization details
    
    Returns:
        CategoryResult object with categorization details
    
    Example:
        >>> result = categorize_feedback("I'd love to see dark mode!")
        >>> print(result.primary_category)
        'Feature Request'
    """
    chain = create_category_chain(provider=provider, model=model)
    result = chain.invoke(feedback_text)
    
    if verbose:
        print(f"\n📋 Category: {result.primary_category}")
        if result.secondary_categories:
            print(f"   Secondary: {', '.join(result.secondary_categories)}")
        print(f"   Urgency: {result.urgency.upper()}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Reasoning: {result.reasoning}")
    
    return result


def batch_categorize(
    feedback_list: List[str],
    provider: str = "gemini",
    model: str = None,
    show_progress: bool = True
) -> List[CategoryResult]:
    """
    Categorize multiple feedback items in batch.
    
    Args:
        feedback_list: List of feedback texts
        provider: LLM provider to use
        model: Optional specific model name
        show_progress: If True, show progress indicator
    
    Returns:
        List of CategoryResult objects
    
    Example:
        >>> feedbacks = ["Bug: app crashes", "Add dark mode please"]
        >>> results = batch_categorize(feedbacks)
        >>> for result in results:
        ...     print(result.primary_category)
    """
    chain = create_category_chain(provider=provider, model=model)
    results = []
    
    for i, feedback in enumerate(feedback_list, 1):
        if show_progress:
            print(f"Processing {i}/{len(feedback_list)}...", end="\r")
        
        try:
            result = chain.invoke(feedback)
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error processing feedback {i}: {str(e)}")
            # Create a default result for failed items
            results.append(None)
    
    if show_progress:
        print(f"\n✅ Processed {len(feedback_list)} items")
    
    return results


def get_category_distribution(results: List[CategoryResult]) -> dict:
    """
    Analyze the distribution of categories in a batch of results.
    
    Args:
        results: List of CategoryResult objects
    
    Returns:
        Dictionary with category counts and percentages
    
    Example:
        >>> results = batch_categorize(feedbacks)
        >>> dist = get_category_distribution(results)
        >>> print(f"Bug Reports: {dist['Bug Report']['count']}")
    """
    from collections import Counter
    
    # Count primary categories
    primary_counts = Counter([r.primary_category for r in results if r])
    
    # Count all categories (including secondary)
    all_categories = []
    for r in results:
        if r:
            all_categories.append(r.primary_category)
            all_categories.extend(r.secondary_categories)
    
    all_counts = Counter(all_categories)
    total = len([r for r in results if r])
    
    distribution = {}
    for category in FeedbackCategory:
        cat_value = category.value
        primary_count = primary_counts.get(cat_value, 0)
        total_count = all_counts.get(cat_value, 0)
        
        distribution[cat_value] = {
            'primary_count': primary_count,
            'total_count': total_count,
            'primary_percentage': (primary_count / total * 100) if total > 0 else 0,
            'total_percentage': (total_count / total * 100) if total > 0 else 0
        }
    
    return distribution


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the category classification chain."""
    print("\n" + "=" * 80)
    print("CATEGORY CLASSIFICATION CHAIN DEMO")
    print("=" * 80)
    
    # Sample feedback covering different categories
    test_feedbacks = [
        # Bug Report
        "The application crashes every time I try to export my data to CSV. This is really frustrating!",
        
        # Feature Request
        "It would be amazing if you could add a dark mode option. My eyes hurt after using the app at night.",
        
        # User Experience
        "The navigation is confusing. I can never find the settings page, and the menu layout is not intuitive.",
        
        # Performance
        "The dashboard takes forever to load. Sometimes it takes 30+ seconds just to see my data.",
        
        # Customer Support
        "I've been waiting 3 days for a response to my support ticket. The customer service is terrible.",
        
        # Pricing
        "The pricing is way too high compared to competitors. I don't think it's worth $99/month for these features.",
        
        # Documentation
        "The API documentation is incomplete. There are no examples for authentication, and many endpoints are undocumented.",
        
        # Mixed (Bug + Feature)
        "The mobile app keeps logging me out randomly (annoying!). Also, please add fingerprint authentication for faster login.",
        
        # General Positive
        "Love this product! It has made my workflow so much more efficient. Keep up the great work!",
        
        # Security
        "I'm concerned about how my data is being stored. Are you using encryption? What about GDPR compliance?"
    ]
    
    print(f"\nAnalyzing {len(test_feedbacks)} feedback samples...\n")
    
    for i, feedback in enumerate(test_feedbacks, 1):
        print(f"\n{'═' * 80}")
        print(f"FEEDBACK {i}:")
        print(f"'{feedback}'")
        print(f"{'─' * 80}")
        
        try:
            # Categorize feedback
            result = categorize_feedback(feedback, verbose=False)
            
            # Display results
            print(f"\n📊 CATEGORIZATION RESULTS:")
            print(f"  Primary:      {result.primary_category}")
            if result.secondary_categories:
                print(f"  Secondary:    {', '.join(result.secondary_categories)}")
            print(f"  Urgency:      {result.urgency.upper()}")
            print(f"  Confidence:   {result.confidence:.1%}")
            print(f"  Reasoning:    {result.reasoning}")
            print(f"  Key Phrases:")
            for phrase in result.key_phrases:
                print(f"    • {phrase}")
        
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
    
    print("\n" + "=" * 80)
    
    # Show distribution
    print("\n📈 CATEGORY DISTRIBUTION:")
    print("─" * 80)
    
    results = [categorize_feedback(fb, verbose=False) for fb in test_feedbacks]
    distribution = get_category_distribution(results)
    
    # Sort by primary count
    sorted_dist = sorted(
        distribution.items(),
        key=lambda x: x[1]['primary_count'],
        reverse=True
    )
    
    for category, stats in sorted_dist:
        if stats['primary_count'] > 0:
            print(f"  {category:20s}  Primary: {stats['primary_count']:2d} ({stats['primary_percentage']:5.1f}%)  "
                  f"Total: {stats['total_count']:2d} ({stats['total_percentage']:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Check available providers before running
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


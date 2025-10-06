"""
Complete Feedback Analysis Pipeline
Integrates sentiment analysis, category classification, and report generation
"""

from typing import List, Dict, Any
from chains.sentiment import create_sentiment_chain, SentimentResult
from chains.category import create_category_chain, CategoryResult
from chains.report import generate_action_plan, ActionPlanReport, format_report_markdown
from utils.llm import get_available_providers, print_provider_status


def analyze_feedback_pipeline(
    raw_feedback: List[str],
    provider: str = "gemini",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete end-to-end feedback analysis pipeline.
    
    Pipeline stages:
    1. Sentiment Analysis - Score each feedback item
    2. Category Classification - Categorize each item
    3. Report Generation - Create actionable product management report
    
    Args:
        raw_feedback: List of raw feedback text strings
        provider: LLM provider to use
        verbose: If True, print progress
    
    Returns:
        Dictionary containing:
            - sentiment_results: List of SentimentResult objects
            - category_results: List of CategoryResult objects
            - report: ActionPlanReport object
            - summary_stats: Quick statistics
    
    Example:
        >>> feedbacks = ["App crashes constantly", "Love the new features!"]
        >>> results = analyze_feedback_pipeline(feedbacks, provider="gemini")
        >>> print(results['report'].report_title)
    """
    if verbose:
        print("\n" + "=" * 80)
        print("🚀 COMPLETE FEEDBACK ANALYSIS PIPELINE")
        print("=" * 80)
        print(f"\nAnalyzing {len(raw_feedback)} feedback items...")
        print(f"Provider: {provider.upper()}\n")
    
    # Stage 1: Sentiment Analysis
    if verbose:
        print("📊 STAGE 1: Sentiment Analysis")
        print("─" * 80)
    
    sentiment_chain = create_sentiment_chain(provider=provider)
    sentiment_results = []
    
    for i, feedback in enumerate(raw_feedback, 1):
        if verbose:
            print(f"  [{i}/{len(raw_feedback)}] Analyzing sentiment...", end="\r")
        
        try:
            result = sentiment_chain.invoke(feedback)
            sentiment_results.append(result)
        except Exception as e:
            if verbose:
                print(f"\n  ⚠️  Error analyzing feedback {i}: {str(e)}")
            sentiment_results.append(None)
    
    if verbose:
        successful = len([r for r in sentiment_results if r])
        print(f"  ✅ Completed: {successful}/{len(raw_feedback)} items analyzed\n")
    
    # Stage 2: Category Classification
    if verbose:
        print("📋 STAGE 2: Category Classification")
        print("─" * 80)
    
    category_chain = create_category_chain(provider=provider)
    category_results = []
    
    for i, feedback in enumerate(raw_feedback, 1):
        if verbose:
            print(f"  [{i}/{len(raw_feedback)}] Categorizing...", end="\r")
        
        try:
            result = category_chain.invoke(feedback)
            category_results.append(result)
        except Exception as e:
            if verbose:
                print(f"\n  ⚠️  Error categorizing feedback {i}: {str(e)}")
            category_results.append(None)
    
    if verbose:
        successful = len([r for r in category_results if r])
        print(f"  ✅ Completed: {successful}/{len(raw_feedback)} items categorized\n")
    
    # Stage 3: Report Generation
    if verbose:
        print("📝 STAGE 3: Action Plan Generation")
        print("─" * 80)
    
    # Prepare feedback items with all metadata
    enriched_feedback = []
    for i, feedback in enumerate(raw_feedback):
        sentiment = sentiment_results[i]
        category = category_results[i]
        
        if sentiment and category:
            enriched_feedback.append({
                'text': feedback,
                'category': category.primary_category,
                'urgency': category.urgency,
                'sentiment': sentiment.sentiment,
                'sentiment_score': sentiment.score,
                'confidence': (sentiment.confidence + category.confidence) / 2
            })
    
    if verbose:
        print(f"  Generating comprehensive action plan...")
    
    report = generate_action_plan(enriched_feedback, provider=provider, verbose=False)
    
    if verbose:
        print(f"  ✅ Generated report with {len(report.action_items)} action items\n")
    
    # Calculate summary statistics
    summary_stats = calculate_summary_stats(sentiment_results, category_results)
    
    if verbose:
        print_summary(summary_stats, report)
    
    return {
        'sentiment_results': sentiment_results,
        'category_results': category_results,
        'report': report,
        'summary_stats': summary_stats,
        'enriched_feedback': enriched_feedback
    }


def calculate_summary_stats(
    sentiment_results: List[SentimentResult],
    category_results: List[CategoryResult]
) -> Dict[str, Any]:
    """Calculate summary statistics from analysis results."""
    from collections import Counter
    
    # Filter out None results
    sentiments = [r for r in sentiment_results if r]
    categories = [r for r in category_results if r]
    
    # Sentiment statistics
    sentiment_distribution = Counter([s.sentiment for s in sentiments])
    avg_sentiment_score = sum(s.score for s in sentiments) / len(sentiments) if sentiments else 0
    avg_sentiment_confidence = sum(s.confidence for s in sentiments) / len(sentiments) if sentiments else 0
    
    # Category statistics
    category_distribution = Counter([c.primary_category for c in categories])
    urgency_distribution = Counter([c.urgency for c in categories])
    avg_category_confidence = sum(c.confidence for c in categories) / len(categories) if categories else 0
    
    # Critical items
    critical_items = [c for c in categories if c.urgency in ['critical', 'high']]
    
    return {
        'total_items': len(sentiment_results),
        'successful_analyses': len(sentiments),
        'sentiment_distribution': dict(sentiment_distribution),
        'category_distribution': dict(category_distribution),
        'urgency_distribution': dict(urgency_distribution),
        'avg_sentiment_score': avg_sentiment_score,
        'avg_sentiment_confidence': avg_sentiment_confidence,
        'avg_category_confidence': avg_category_confidence,
        'critical_items_count': len(critical_items),
        'critical_items_percentage': (len(critical_items) / len(categories) * 100) if categories else 0
    }


def print_summary(summary_stats: Dict[str, Any], report: ActionPlanReport):
    """Print analysis summary."""
    print("=" * 80)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Successfully analyzed: {summary_stats['successful_analyses']}/{summary_stats['total_items']} items")
    print(f"📈 Average sentiment score: {summary_stats['avg_sentiment_score']:.1f}/10")
    print(f"🎯 Average confidence: {summary_stats['avg_sentiment_confidence']:.1%}")
    
    print(f"\n💭 Sentiment Distribution:")
    for sentiment, count in sorted(summary_stats['sentiment_distribution'].items()):
        percentage = (count / summary_stats['successful_analyses'] * 100)
        bar = "█" * int(percentage / 5)
        print(f"  {sentiment:8s}: {bar} {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n📋 Category Distribution:")
    for category, count in sorted(summary_stats['category_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / summary_stats['successful_analyses'] * 100)
        bar = "█" * int(percentage / 5)
        print(f"  {category:20s}: {bar} {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n⚠️  Urgency Levels:")
    urgency_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
    for urgency in ['critical', 'high', 'medium', 'low']:
        count = summary_stats['urgency_distribution'].get(urgency, 0)
        if count > 0:
            percentage = (count / summary_stats['successful_analyses'] * 100)
            emoji = urgency_emoji.get(urgency, '⚪')
            print(f"  {emoji} {urgency:8s}: {count:3d} ({percentage:5.1f}%)")
    
    if summary_stats['critical_items_count'] > 0:
        print(f"\n🚨 {summary_stats['critical_items_count']} items require immediate attention!")
    
    print(f"\n📝 Report Generated:")
    print(f"  • Title: {report.report_title}")
    print(f"  • Action Items: {len(report.action_items)}")
    print(f"  • Quick Wins: {len(report.quick_wins)}")
    print(f"  • Strategic Initiatives: {len(report.strategic_initiatives)}")
    
    print("\n" + "=" * 80)


def save_complete_report(
    results: Dict[str, Any],
    output_dir: str = ".",
    base_filename: str = "feedback_analysis"
):
    """
    Save complete analysis results to files.
    
    Args:
        results: Results dictionary from analyze_feedback_pipeline
        output_dir: Directory to save files
        base_filename: Base name for output files
    """
    import json
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save markdown report
    markdown = format_report_markdown(results['report'])
    markdown_file = os.path.join(output_dir, f"{base_filename}_{timestamp}.md")
    with open(markdown_file, 'w') as f:
        f.write(markdown)
    print(f"✅ Markdown report saved: {markdown_file}")
    
    # Save JSON data
    json_data = {
        'generated_at': timestamp,
        'summary_stats': results['summary_stats'],
        'report': results['report'].model_dump(),
        'enriched_feedback': results['enriched_feedback']
    }
    json_file = os.path.join(output_dir, f"{base_filename}_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✅ JSON data saved: {json_file}")
    
    # Save detailed CSV
    import csv
    csv_file = os.path.join(output_dir, f"{base_filename}_{timestamp}_details.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'feedback', 'sentiment', 'sentiment_score', 'category', 'urgency', 'confidence'
        ])
        writer.writeheader()
        
        for item in results['enriched_feedback']:
            writer.writerow({
                'feedback': item['text'],
                'sentiment': item['sentiment'],
                'sentiment_score': item['sentiment_score'],
                'category': item['category'],
                'urgency': item['urgency'],
                'confidence': f"{item['confidence']:.2f}"
            })
    print(f"✅ Detailed CSV saved: {csv_file}")


# ============================================================================
# DEMO
# ============================================================================

def main():
    """Demo the complete pipeline."""
    # Sample diverse feedback
    sample_feedback = [
        "The app crashes every single time I try to export my data! This is completely unacceptable and I've lost hours of work.",
        "Love the new interface! It's so much cleaner and easier to navigate. Great job on the redesign!",
        "Your customer support is terrible. I've been waiting 5 days for a response to my critical issue.",
        "Would really appreciate a dark mode option for night work sessions. My eyes get strained.",
        "The dashboard loads extremely slowly, sometimes taking 30+ seconds. This makes the app almost unusable.",
        "Please add bulk editing! Having to edit items one by one is incredibly tedious and time-consuming.",
        "The mobile app keeps logging me out randomly. Very frustrating when I'm in the middle of something.",
        "Pricing is way too high compared to similar tools. Hard to justify the cost to my team.",
        "Documentation is severely lacking. Many features are not explained and I had to figure things out by trial and error.",
        "The search function is broken - it doesn't find obvious matches and returns irrelevant results.",
        "Absolutely love this product! It has transformed our workflow and saved us countless hours.",
        "Need better integration with Slack and Microsoft Teams for notifications.",
        "The onboarding process is too complicated. Took me an hour just to set up my account.",
        "Security is a concern - there's no two-factor authentication option available.",
        "API documentation is outdated and many endpoints don't work as described.",
        "App performance on large datasets is terrible. Everything grinds to a halt with more than 1000 items.",
        "Would love to see keyboard shortcuts for power users. Mouse navigation is slow.",
        "The export feature only supports CSV. Please add Excel and PDF options!",
        "UI is confusing and not intuitive. Had to watch multiple tutorials just to understand basic features.",
        "This tool has been a game-changer for our team. Highly recommend it!",
    ]
    
    # Run complete pipeline
    results = analyze_feedback_pipeline(
        raw_feedback=sample_feedback,
        provider="gemini",
        verbose=True
    )
    
    # Save reports
    print("\n💾 Saving complete analysis reports...")
    save_complete_report(results)
    
    # Print top action items
    print("\n" + "=" * 80)
    print("🎯 TOP 3 PRIORITY ACTION ITEMS")
    print("=" * 80)
    
    for i, item in enumerate(results['report'].action_items[:3], 1):
        priority_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }
        
        print(f"\n{i}. {priority_emoji[item.priority]} {item.title}")
        print(f"   Category: {item.category} | Priority: {item.priority.upper()}")
        print(f"   Timeframe: {item.timeframe} | Effort: {item.estimated_effort}")
        print(f"   Action: {item.recommended_action[:100]}...")
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Check providers
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


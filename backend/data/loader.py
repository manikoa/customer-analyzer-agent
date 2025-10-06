"""
Data Loader Utilities
Functions to read and parse the feedback dataset
"""

import csv
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


# ============================================================================
# CSV LOADING FUNCTIONS
# ============================================================================

def load_feedback_csv(
    filepath: str = "feedback_data.csv",
    encoding: str = "utf-8"
) -> List[Dict[str, str]]:
    """
    Load feedback data from CSV file using csv.DictReader.
    
    Args:
        filepath: Path to the CSV file
        encoding: File encoding (default: utf-8)
    
    Returns:
        List of dictionaries, each representing a feedback item
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        csv.Error: If CSV is malformed
    
    Example:
        >>> feedback = load_feedback_csv("feedback_data.csv")
        >>> print(f"Loaded {len(feedback)} items")
        >>> print(feedback[0]['Raw_Text'])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    feedback_items = []
    
    with open(filepath, 'r', encoding=encoding) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            feedback_items.append({
                'ID': row.get('ID', ''),
                'Raw_Text': row.get('Raw_Text', ''),
                'Source': row.get('Source', '')
            })
    
    return feedback_items


def load_feedback_pandas(
    filepath: str = "feedback_data.csv",
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Load feedback data from CSV file using pandas.
    
    Args:
        filepath: Path to the CSV file
        encoding: File encoding (default: utf-8)
    
    Returns:
        pandas DataFrame with feedback data
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    
    Example:
        >>> df = load_feedback_pandas("feedback_data.csv")
        >>> print(df.head())
        >>> print(f"Shape: {df.shape}")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    df = pd.read_csv(filepath, encoding=encoding)
    
    # Basic cleaning
    df['Raw_Text'] = df['Raw_Text'].fillna('').astype(str)
    df['Source'] = df['Source'].fillna('Unknown').astype(str)
    df['ID'] = df['ID'].fillna(0).astype(int)
    
    return df


# ============================================================================
# DATA EXTRACTION FUNCTIONS
# ============================================================================

def get_raw_feedback_texts(
    feedback_data: List[Dict[str, str]]
) -> List[str]:
    """
    Extract just the raw feedback text from loaded data.
    
    Args:
        feedback_data: List of feedback dictionaries
    
    Returns:
        List of raw feedback text strings
    
    Example:
        >>> feedback = load_feedback_csv()
        >>> texts = get_raw_feedback_texts(feedback)
        >>> print(texts[0])
    """
    return [item['Raw_Text'] for item in feedback_data if item.get('Raw_Text')]


def filter_by_source(
    feedback_data: List[Dict[str, str]],
    source: str
) -> List[Dict[str, str]]:
    """
    Filter feedback by source.
    
    Args:
        feedback_data: List of feedback dictionaries
        source: Source to filter by (e.g., "App Store", "Survey")
    
    Returns:
        Filtered list of feedback items
    
    Example:
        >>> feedback = load_feedback_csv()
        >>> app_store = filter_by_source(feedback, "App Store")
        >>> print(f"App Store feedback: {len(app_store)} items")
    """
    return [
        item for item in feedback_data 
        if item.get('Source', '').lower() == source.lower()
    ]


def get_sample_feedback(
    feedback_data: List[Dict[str, str]],
    n: int = 10,
    random: bool = False
) -> List[Dict[str, str]]:
    """
    Get a sample of feedback items.
    
    Args:
        feedback_data: List of feedback dictionaries
        n: Number of items to sample
        random: If True, randomly sample; if False, take first n items
    
    Returns:
        List of sampled feedback items
    
    Example:
        >>> feedback = load_feedback_csv()
        >>> sample = get_sample_feedback(feedback, n=5, random=True)
    """
    if random:
        import random as rnd
        return rnd.sample(feedback_data, min(n, len(feedback_data)))
    else:
        return feedback_data[:n]


# ============================================================================
# DATA STATISTICS
# ============================================================================

def get_dataset_stats(
    feedback_data: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Get statistics about the feedback dataset.
    
    Args:
        feedback_data: List of feedback dictionaries
    
    Returns:
        Dictionary with dataset statistics
    
    Example:
        >>> feedback = load_feedback_csv()
        >>> stats = get_dataset_stats(feedback)
        >>> print(f"Total items: {stats['total_items']}")
    """
    from collections import Counter
    
    total = len(feedback_data)
    sources = Counter(item.get('Source', 'Unknown') for item in feedback_data)
    
    text_lengths = [len(item.get('Raw_Text', '')) for item in feedback_data]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    
    return {
        'total_items': total,
        'sources': dict(sources),
        'avg_text_length': round(avg_length, 1),
        'min_text_length': min(text_lengths) if text_lengths else 0,
        'max_text_length': max(text_lengths) if text_lengths else 0
    }


def print_dataset_info(filepath: str = "feedback_data.csv"):
    """
    Print information about the feedback dataset.
    
    Args:
        filepath: Path to the CSV file
    
    Example:
        >>> print_dataset_info("feedback_data.csv")
    """
    feedback = load_feedback_csv(filepath)
    stats = get_dataset_stats(feedback)
    
    print("\n" + "=" * 80)
    print("FEEDBACK DATASET INFORMATION")
    print("=" * 80)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Items: {stats['total_items']}")
    print(f"   Avg Text Length: {stats['avg_text_length']} characters")
    print(f"   Min/Max Length: {stats['min_text_length']} / {stats['max_text_length']} characters")
    
    print(f"\n📍 Sources:")
    for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_items'] * 100)
        print(f"   • {source:20s}: {count:3d} items ({percentage:5.1f}%)")
    
    print(f"\n📝 Sample Feedback (first 3):")
    for i, item in enumerate(feedback[:3], 1):
        print(f"\n   {i}. [ID: {item['ID']}] [{item['Source']}]")
        print(f"      {item['Raw_Text'][:80]}...")
    
    print("\n" + "=" * 80 + "\n")


# ============================================================================
# WORKFLOW INTEGRATION
# ============================================================================

def load_feedback_for_workflow(
    filepath: str = "feedback_data.csv",
    limit: Optional[int] = None,
    sources: Optional[List[str]] = None
) -> List[str]:
    """
    Load feedback data ready for workflow execution.
    
    This function loads feedback and returns just the raw text
    strings ready to be fed into the analysis workflow.
    
    Args:
        filepath: Path to the CSV file
        limit: Optional limit on number of items to load
        sources: Optional list of sources to filter by
    
    Returns:
        List of raw feedback text strings
    
    Example:
        >>> feedback_texts = load_feedback_for_workflow(
        ...     filepath="feedback_data.csv",
        ...     limit=20,
        ...     sources=["App Store", "Google Play"]
        ... )
        >>> # Ready to use with workflow
        >>> from core.workflow import run_workflow
        >>> results = run_workflow(feedback_texts, provider="gemini")
    """
    # Load data
    feedback = load_feedback_csv(filepath)
    
    # Filter by sources if specified
    if sources:
        filtered = []
        for item in feedback:
            if item.get('Source') in sources:
                filtered.append(item)
        feedback = filtered
    
    # Apply limit if specified
    if limit:
        feedback = feedback[:limit]
    
    # Extract raw texts
    texts = get_raw_feedback_texts(feedback)
    
    return texts


def create_enriched_dataset(
    original_data: List[Dict[str, str]],
    sentiment_results: List[Any],
    category_results: List[Any],
    output_file: str = "enriched_feedback.csv"
) -> pd.DataFrame:
    """
    Create an enriched dataset by combining original data with analysis results.
    
    Args:
        original_data: Original feedback data
        sentiment_results: List of SentimentResult objects
        category_results: List of CategoryResult objects
        output_file: Output CSV file path
    
    Returns:
        pandas DataFrame with enriched data
    
    Example:
        >>> feedback = load_feedback_csv()
        >>> # ... run analysis ...
        >>> enriched = create_enriched_dataset(
        ...     feedback,
        ...     sentiment_results,
        ...     category_results,
        ...     "enriched_feedback.csv"
        ... )
    """
    enriched = []
    
    for i, item in enumerate(original_data):
        sentiment = sentiment_results[i] if i < len(sentiment_results) else None
        category = category_results[i] if i < len(category_results) else None
        
        enriched_item = {
            'ID': item.get('ID'),
            'Raw_Text': item.get('Raw_Text'),
            'Source': item.get('Source'),
            'Sentiment': sentiment.sentiment if sentiment else None,
            'Sentiment_Score': sentiment.score if sentiment else None,
            'Sentiment_Confidence': sentiment.confidence if sentiment else None,
            'Category': category.primary_category if category else None,
            'Urgency': category.urgency if category else None,
            'Category_Confidence': category.confidence if category else None
        }
        
        enriched.append(enriched_item)
    
    df = pd.DataFrame(enriched)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Enriched dataset saved to: {output_file}")
    
    return df


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the data loader functions."""
    print("\n" + "=" * 80)
    print("DATA LOADER DEMONSTRATION")
    print("=" * 80)
    
    # Load data
    print("\n1️⃣  Loading feedback data...")
    try:
        feedback = load_feedback_csv("feedback_data.csv")
        print(f"   ✅ Loaded {len(feedback)} feedback items")
    except FileNotFoundError as e:
        print(f"   ❌ Error: {e}")
        print("\n   Creating sample data...")
        print("   Run this script from the project root where feedback_data.csv exists")
        return
    
    # Get statistics
    print("\n2️⃣  Dataset statistics:")
    stats = get_dataset_stats(feedback)
    for key, value in stats.items():
        if key != 'sources':
            print(f"   {key}: {value}")
    
    print(f"\n   Sources breakdown:")
    for source, count in stats['sources'].items():
        print(f"     • {source}: {count} items")
    
    # Filter by source
    print("\n3️⃣  Filtering by source:")
    app_store = filter_by_source(feedback, "App Store")
    print(f"   App Store feedback: {len(app_store)} items")
    
    surveys = filter_by_source(feedback, "Survey")
    print(f"   Survey feedback: {len(surveys)} items")
    
    # Get sample
    print("\n4️⃣  Random sample (3 items):")
    sample = get_sample_feedback(feedback, n=3, random=True)
    for item in sample:
        print(f"\n   [ID: {item['ID']}] [{item['Source']}]")
        print(f"   {item['Raw_Text'][:80]}...")
    
    # Prepare for workflow
    print("\n5️⃣  Preparing for workflow:")
    texts = load_feedback_for_workflow(
        filepath="feedback_data.csv",
        limit=5,
        sources=["App Store"]
    )
    print(f"   ✅ Prepared {len(texts)} texts for analysis")
    
    # Full dataset info
    print("\n6️⃣  Full dataset information:")
    print_dataset_info("feedback_data.csv")
    
    print("\n" + "=" * 80)
    print("✅ DATA LOADER DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


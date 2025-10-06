"""
Sentiment Analysis Chain using LangChain LCEL
Classifies customer feedback sentiment with a numerical score (1-10)
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import models from centralized location
from core.models import SentimentResult

# Import utilities
from utils.llm import get_llm, get_available_providers
from utils.prompts import create_sentiment_prompt


# ============================================================================
# CHAIN BUILDER
# ============================================================================


def create_sentiment_chain(provider: str = "gemini", temperature: float = 0.0, model: str = None):
    """
    Create the sentiment analysis chain using LangChain LCEL.
    
    This chain follows the pattern: Prompt | Model | Parser
    
    Args:
        provider: LLM provider ("openai", "gemini", or "anthropic")
        temperature: Model temperature
        model: Optional specific model name
    
    Returns:
        Configured LCEL chain that takes feedback_text and returns SentimentResult
    """
    # Initialize components
    llm = get_llm(provider=provider, temperature=temperature, model=model)
    parser = PydanticOutputParser(pydantic_object=SentimentResult)
    
    # Create prompt template with format instructions from utils
    prompt = create_sentiment_prompt(
        format_instructions=parser.get_format_instructions()
    )
    
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

def analyze_sentiment(
    feedback_text: str,
    provider: str = "gemini",
    temperature: float = 0.0,
    model: str = None,
    verbose: bool = False
) -> SentimentResult:
    """
    Analyze sentiment of a single feedback text.
    
    Args:
        feedback_text: The customer feedback to analyze
        provider: LLM provider ("openai" or "gemini")
        temperature: Model temperature for generation
        model: Optional specific model name
        verbose: Print processing information
    
    Returns:
        SentimentResult object with classification details
    
    Example:
        >>> result = analyze_sentiment("This product is amazing! Best purchase ever!")
        >>> print(f"Sentiment: {result.sentiment}, Score: {result.score}")
        Sentiment: positive, Score: 10
    """
    if verbose:
        print(f"🔄 Analyzing sentiment with {provider}...")
    
    try:
        chain = create_sentiment_chain(
            provider=provider,
            temperature=temperature,
            model=model
        )
        result = chain.invoke(feedback_text)
        
        if verbose:
            print(f"✓ Analysis complete!")
            print(f"  Sentiment: {result.sentiment}")
            print(f"  Score: {result.score}/10")
            print(f"  Confidence: {result.confidence:.2f}")
        
        return result
    
    except Exception as e:
        if verbose:
            print(f"✗ Error during analysis: {str(e)}")
        raise


def batch_analyze_sentiments(
    feedback_texts: list[str],
    provider: str = "gemini",
    temperature: float = 0.0,
    model: str = None,
    verbose: bool = False
) -> list[SentimentResult]:
    """
    Analyze sentiment for multiple feedback texts.
    
    Args:
        feedback_texts: List of customer feedback texts
        provider: LLM provider ("openai" or "gemini")
        temperature: Model temperature
        model: Optional specific model name
        verbose: Print progress information
    
    Returns:
        List of SentimentResult objects
    """
    chain = create_sentiment_chain(
        provider=provider,
        temperature=temperature,
        model=model
    )
    
    results = []
    total = len(feedback_texts)
    
    for i, text in enumerate(feedback_texts, 1):
        if verbose:
            print(f"Processing {i}/{total}...")
        
        try:
            result = chain.invoke(text)
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  ✗ Error: {str(e)}")
            # Continue processing other texts
            continue
    
    return results


# ============================================================================
# MAIN / EXAMPLES
# ============================================================================

def main():
    """
    Example usage of the sentiment analysis chain.
    """
    print("=" * 80)
    print("SENTIMENT ANALYSIS CHAIN - DEMO")
    print("=" * 80)
    
    # Sample feedback texts
    sample_feedbacks = [
        "This product is absolutely amazing! Best purchase I've ever made. The quality exceeds all expectations.",
        "The app crashes constantly and customer support is unhelpful. Very disappointed.",
        "It's okay. Does what it's supposed to do. Nothing special.",
        "I love the new features, but the price increase is a bit steep.",
        "Terrible experience. Would not recommend to anyone. Complete waste of money."
    ]
    
    # Analyze each feedback
    for i, feedback in enumerate(sample_feedbacks, 1):
        print(f"\n{'─' * 80}")
        print(f"FEEDBACK {i}:")
        print(f"'{feedback}'")
        print(f"{'─' * 80}")
        
        try:
            # Analyze sentiment
            result = analyze_sentiment(feedback, verbose=False)
            
            # Display results
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"  Sentiment:    {result.sentiment.upper()}")
            print(f"  Score:        {result.score}/10")
            print(f"  Confidence:   {result.confidence:.1%}")
            print(f"  Reasoning:    {result.reasoning}")
            print(f"  Key Phrases:  {', '.join(result.key_phrases)}")
        
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
    
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


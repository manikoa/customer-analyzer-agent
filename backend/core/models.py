"""
Pydantic Models for Customer Feedback Analyzer
Defines structured outputs for all LLM chains with validation
"""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ============================================================================
# ENUMS FOR TYPE SAFETY
# ============================================================================

class SentimentType(str, Enum):
    """Sentiment classification types."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class FeedbackCategory(str, Enum):
    """Predefined business categories for customer feedback."""
    BUG_REPORT = "Bug Report"
    FEATURE_REQUEST = "Feature Request"
    USER_EXPERIENCE = "User Experience"
    PERFORMANCE = "Performance"
    CUSTOMER_SUPPORT = "Customer Support"
    PRICING = "Pricing"
    DOCUMENTATION = "Documentation"
    SECURITY = "Security"
    INTEGRATION = "Integration"
    GENERAL_FEEDBACK = "General Feedback"


class UrgencyLevel(str, Enum):
    """Urgency levels for feedback items."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PriorityLevel(str, Enum):
    """Priority levels for action items."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ImpactArea(str, Enum):
    """Business impact areas."""
    USER_RETENTION = "User Retention"
    USER_ACQUISITION = "User Acquisition"
    REVENUE = "Revenue"
    COST_REDUCTION = "Cost Reduction"
    CUSTOMER_SATISFACTION = "Customer Satisfaction"
    TECHNICAL_DEBT = "Technical Debt"
    COMPETITIVE_ADVANTAGE = "Competitive Advantage"


class TimeFrame(str, Enum):
    """Implementation timeframes."""
    IMMEDIATE = "Immediate (0-1 week)"
    SHORT_TERM = "Short-term (2-4 weeks)"
    MEDIUM_TERM = "Medium-term (1-3 months)"
    LONG_TERM = "Long-term (3+ months)"


# ============================================================================
# 1. SENTIMENT RESULT MODEL
# ============================================================================

class SentimentResult(BaseModel):
    """
    Structured output for sentiment analysis.
    
    This model enforces the structure of sentiment analysis results
    from the LLM, ensuring consistency and type safety.
    
    Attributes:
        sentiment: Classification (positive, neutral, or negative)
        score: Numerical intensity score from 1-10
        confidence: Confidence level (0.0 to 1.0)
        key_phrases: 2-5 phrases that influenced the sentiment
        reasoning: Brief explanation for the classification
    
    Scoring Guidelines:
        1-3: Strongly negative (angry, frustrated, very dissatisfied)
        4-5: Negative (disappointed, unhappy, critical)
        6: Neutral (balanced, factual, no strong emotion)
        7-8: Positive (satisfied, happy, complimentary)
        9-10: Strongly positive (delighted, enthusiastic, highly satisfied)
    """
    
    sentiment: SentimentType = Field(
        description="The overall sentiment classification"
    )
    
    score: int = Field(
        description="Numerical sentiment intensity from 1 (most negative) to 10 (most positive)",
        ge=1,
        le=10
    )
    
    confidence: float = Field(
        description="Confidence level in the classification (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    key_phrases: List[str] = Field(
        description="1-5 key phrases from the feedback that influenced the sentiment",
        min_items=1,
        max_items=5
    )
    
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) for the sentiment classification"
    )
    
    @validator('score')
    def validate_score_matches_sentiment(cls, v, values):
        """Ensure score aligns with sentiment classification."""
        if 'sentiment' not in values:
            return v
        
        sentiment = values['sentiment']
        
        if sentiment == SentimentType.NEGATIVE and v > 5:
            raise ValueError(f"Negative sentiment requires score 1-5, got {v}")
        elif sentiment == SentimentType.POSITIVE and v < 7:
            raise ValueError(f"Positive sentiment requires score 7-10, got {v}")
        elif sentiment == SentimentType.NEUTRAL and (v < 4 or v > 7):
            raise ValueError(f"Neutral sentiment requires score 4-7, got {v}")
        
        return v
    
    @validator('key_phrases')
    def clean_key_phrases(cls, v):
        """Clean and validate key phrases."""
        cleaned = [phrase.strip() for phrase in v if phrase.strip()]
        if len(cleaned) < 2:
            raise ValueError("At least 2 key phrases are required")
        return cleaned[:5]
    
    @validator('reasoning')
    def validate_reasoning_length(cls, v):
        """Ensure reasoning is not too long."""
        if len(v) > 500:
            return v[:497] + "..."
        return v.strip()
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "score": 9,
                "confidence": 0.92,
                "key_phrases": ["love the product", "excellent service", "highly recommend"],
                "reasoning": "Customer expresses strong satisfaction with multiple positive phrases and enthusiastic language."
            }
        }


# ============================================================================
# 2. CATEGORY RESULT MODEL
# ============================================================================

class CategoryResult(BaseModel):
    """
    Structured output for category classification.
    
    This model enforces the structure of category classification results,
    enabling consistent categorization of feedback into business themes.
    
    Attributes:
        primary_category: Main category (theme) that best fits the feedback
        secondary_categories: Up to 2 additional relevant categories (sub-themes)
        key_phrases: 2-5 supporting phrases from the feedback
        confidence: Confidence in the categorization (0.0 to 1.0)
        reasoning: Brief explanation for the categorization
        urgency: Priority level (low, medium, high, critical)
    """
    
    primary_category: FeedbackCategory = Field(
        description="The primary category (theme) that best fits this feedback"
    )
    
    secondary_categories: List[FeedbackCategory] = Field(
        default_factory=list,
        description="Additional relevant categories (sub-themes), max 2 items",
        max_items=2
    )
    
    key_phrases: List[str] = Field(
        description="1-5 specific phrases from the feedback that support the categorization",
        min_items=1,
        max_items=5
    )
    
    confidence: float = Field(
        description="Confidence level for this categorization (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    reasoning: str = Field(
        description="Brief explanation (1-3 sentences) for why this categorization was chosen"
    )
    
    urgency: UrgencyLevel = Field(
        description="Urgency level: low, medium, high, or critical"
    )
    
    @validator('secondary_categories')
    def no_duplicate_categories(cls, v, values):
        """Ensure secondary categories don't include the primary category."""
        if 'primary_category' in values:
            primary = values['primary_category']
            v = [cat for cat in v if cat != primary]
        
        # Remove duplicates
        seen = set()
        unique = []
        for cat in v:
            if cat not in seen:
                seen.add(cat)
                unique.append(cat)
        
        return unique[:2]  # Limit to 2
    
    @validator('key_phrases')
    def clean_key_phrases(cls, v):
        """Clean and validate key phrases."""
        cleaned = [phrase.strip() for phrase in v if phrase.strip()]
        if len(cleaned) < 2:
            raise ValueError("At least 2 key phrases are required")
        return cleaned[:5]
    
    @validator('reasoning')
    def validate_reasoning_length(cls, v):
        """Ensure reasoning is concise."""
        if len(v) > 500:
            return v[:497] + "..."
        return v.strip()
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "primary_category": "Bug Report",
                "secondary_categories": ["Performance"],
                "key_phrases": ["app crashes", "data loss", "export fails"],
                "confidence": 0.95,
                "reasoning": "Clear technical issue with specific functionality breakdown and data integrity concerns.",
                "urgency": "critical"
            }
        }


# ============================================================================
# 3. REPORT ITEM MODEL
# ============================================================================

class ReportItem(BaseModel):
    """
    Structured output for product management action items.
    
    This model defines a comprehensive action item for Product Managers,
    derived from analyzed customer feedback with clear recommendations.
    
    Attributes:
        title: Clear, actionable title (max 100 chars)
        category: Primary feedback category this addresses
        priority: Priority level (low, medium, high, critical)
        impact_areas: Business areas affected (1-3 items)
        problem_statement: Clear description of the problem (2-3 sentences)
        affected_users: Description of user segments affected
        user_pain_level: Pain level from 1 (minor) to 10 (critical blocker)
        supporting_feedback_count: Number of feedback items supporting this
        recommended_action: Specific, actionable recommendation (2-4 sentences)
        expected_outcome: Expected result (1-2 sentences)
        estimated_effort: Rough effort estimate (Low, Medium, or High)
        timeframe: Recommended implementation timeframe
        success_metrics: 2-4 measurable metrics to track success
        key_quotes: 2-3 representative customer quotes
        risk_if_ignored: Consequences of inaction (1-2 sentences)
        dependencies: Other items or systems this depends on (optional)
    """
    
    title: str = Field(
        description="Clear, actionable title for the action item",
        max_length=100
    )
    
    category: FeedbackCategory = Field(
        description="The primary feedback category this addresses"
    )
    
    priority: PriorityLevel = Field(
        description="Priority level: low, medium, high, or critical"
    )
    
    impact_areas: List[ImpactArea] = Field(
        description="Business areas this will impact (1-3 items)",
        min_items=1,
        max_items=3
    )
    
    problem_statement: str = Field(
        description="Clear description of the problem or opportunity (2-3 sentences)"
    )
    
    affected_users: str = Field(
        description="Description of which user segments are affected"
    )
    
    user_pain_level: int = Field(
        description="Pain level from 1 (minor inconvenience) to 10 (critical blocker)",
        ge=1,
        le=10
    )
    
    supporting_feedback_count: int = Field(
        description="Number of feedback items supporting this action",
        ge=1
    )
    
    recommended_action: str = Field(
        description="Specific, actionable recommendation for the product team (2-4 sentences)"
    )
    
    expected_outcome: str = Field(
        description="Expected result or improvement from taking this action (1-2 sentences)"
    )
    
    estimated_effort: str = Field(
        description="Rough effort estimate: Low, Medium, or High"
    )
    
    timeframe: TimeFrame = Field(
        description="Recommended implementation timeframe"
    )
    
    success_metrics: List[str] = Field(
        description="1-4 measurable metrics to track success",
        min_items=1,
        max_items=4
    )
    
    key_quotes: List[str] = Field(
        description="1-3 representative customer quotes that highlight the issue",
        min_items=1,
        max_items=3
    )
    
    risk_if_ignored: str = Field(
        description="Potential consequences of not addressing this issue (1-2 sentences)"
    )
    
    dependencies: Optional[List[str]] = Field(
        default_factory=list,
        description="Other items or systems this depends on (optional)"
    )
    
    @validator('title')
    def validate_title_length(cls, v):
        """Ensure title is concise."""
        v = v.strip()
        if len(v) > 100:
            return v[:97] + "..."
        return v
    
    @validator('estimated_effort')
    def validate_effort(cls, v):
        """Validate effort level."""
        v = v.strip().capitalize()
        allowed = ['Low', 'Medium', 'High']
        if v not in allowed:
            raise ValueError(f"Effort must be one of: {', '.join(allowed)}")
        return v
    
    @validator('key_quotes')
    def clean_quotes(cls, v):
        """Clean and validate quotes."""
        cleaned = [q.strip().strip('"\'') for q in v if q.strip()]
        if len(cleaned) < 2:
            raise ValueError("At least 2 key quotes are required")
        return cleaned[:3]
    
    @validator('success_metrics')
    def clean_metrics(cls, v):
        """Clean and validate success metrics."""
        cleaned = [m.strip() for m in v if m.strip()]
        if len(cleaned) < 2:
            raise ValueError("At least 2 success metrics are required")
        return cleaned[:4]
    
    @validator('problem_statement', 'recommended_action', 'expected_outcome', 'risk_if_ignored')
    def validate_text_length(cls, v):
        """Ensure text fields are reasonable length."""
        v = v.strip()
        if len(v) > 1000:
            return v[:997] + "..."
        return v
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "title": "Fix Critical Data Export Crash",
                "category": "Bug Report",
                "priority": "critical",
                "impact_areas": ["User Retention", "Customer Satisfaction"],
                "problem_statement": "Users experience crashes when exporting data to CSV, resulting in lost work and data. This affects approximately 60% of active users who regularly use export functionality.",
                "affected_users": "All users who export data (estimated 60% of active users)",
                "user_pain_level": 9,
                "supporting_feedback_count": 15,
                "recommended_action": "Implement robust error handling in the export module, add data validation before export processing, and create automated tests for various data sizes and formats.",
                "expected_outcome": "Eliminate export crashes, improve user confidence in the export feature, and reduce support tickets by 40%.",
                "estimated_effort": "Medium",
                "timeframe": "Immediate (0-1 week)",
                "success_metrics": [
                    "Zero export-related crashes",
                    "95% successful export rate",
                    "40% reduction in support tickets",
                    "User satisfaction score increase"
                ],
                "key_quotes": [
                    "The app crashes every time I try to export",
                    "Lost all my work again!",
                    "Export feature is completely broken"
                ],
                "risk_if_ignored": "Continued user frustration, increased churn rate, negative reviews, and potential revenue impact from losing power users.",
                "dependencies": ["Database optimization", "File handling improvements"]
            }
        }


# ============================================================================
# SUPPORTING MODELS
# ============================================================================

class ActionPlanReport(BaseModel):
    """
    Complete action plan with multiple prioritized items.
    
    This is the top-level report generated for Product Managers.
    """
    
    report_title: str = Field(
        description="Title for this action plan report"
    )
    
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp when report was generated"
    )
    
    summary: str = Field(
        description="Executive summary of key findings (3-5 sentences)"
    )
    
    total_feedback_analyzed: int = Field(
        description="Total number of feedback items analyzed",
        ge=1
    )
    
    action_items: List[ReportItem] = Field(
        description="Prioritized list of action items (typically 3-10 items)",
        min_items=1
    )
    
    overall_sentiment: SentimentType = Field(
        description="Overall sentiment trend: positive, neutral, or negative"
    )
    
    top_categories: List[FeedbackCategory] = Field(
        description="Top 3-5 most frequent feedback categories",
        max_items=5
    )
    
    quick_wins: List[str] = Field(
        description="1-3 easy wins that can be implemented quickly",
        min_items=1,
        max_items=3
    )
    
    strategic_initiatives: List[str] = Field(
        description="1-3 longer-term strategic recommendations",
        min_items=1,
        max_items=3
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


# ============================================================================
# MODEL VALIDATION HELPERS
# ============================================================================

def validate_sentiment_result(data: dict) -> SentimentResult:
    """
    Validate and parse sentiment result data.
    
    Args:
        data: Dictionary containing sentiment data
    
    Returns:
        Validated SentimentResult object
    
    Raises:
        ValidationError: If data doesn't match model requirements
    """
    return SentimentResult(**data)


def validate_category_result(data: dict) -> CategoryResult:
    """
    Validate and parse category result data.
    
    Args:
        data: Dictionary containing category data
    
    Returns:
        Validated CategoryResult object
    
    Raises:
        ValidationError: If data doesn't match model requirements
    """
    return CategoryResult(**data)


def validate_report_item(data: dict) -> ReportItem:
    """
    Validate and parse report item data.
    
    Args:
        data: Dictionary containing report item data
    
    Returns:
        Validated ReportItem object
    
    Raises:
        ValidationError: If data doesn't match model requirements
    """
    return ReportItem(**data)


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the Pydantic models."""
    import json
    
    print("\n" + "=" * 80)
    print("PYDANTIC MODELS DEMONSTRATION")
    print("=" * 80)
    
    # 1. SentimentResult
    print("\n1️⃣  SENTIMENT RESULT MODEL")
    print("─" * 80)
    
    sentiment_data = {
        "sentiment": "positive",
        "score": 9,
        "confidence": 0.92,
        "key_phrases": ["love the product", "excellent service", "highly recommend"],
        "reasoning": "Customer expresses strong satisfaction with multiple positive phrases."
    }
    
    sentiment = SentimentResult(**sentiment_data)
    print("✅ SentimentResult created and validated")
    print(json.dumps(sentiment.model_dump(), indent=2))
    
    # 2. CategoryResult
    print("\n2️⃣  CATEGORY RESULT MODEL")
    print("─" * 80)
    
    category_data = {
        "primary_category": "Bug Report",
        "secondary_categories": ["Performance"],
        "key_phrases": ["app crashes", "data loss", "export fails"],
        "confidence": 0.95,
        "reasoning": "Clear technical issue with specific functionality breakdown.",
        "urgency": "critical"
    }
    
    category = CategoryResult(**category_data)
    print("✅ CategoryResult created and validated")
    print(json.dumps(category.model_dump(), indent=2))
    
    # 3. ReportItem
    print("\n3️⃣  REPORT ITEM MODEL")
    print("─" * 80)
    
    report_item_data = {
        "title": "Fix Critical Data Export Crash",
        "category": "Bug Report",
        "priority": "critical",
        "impact_areas": ["User Retention", "Customer Satisfaction"],
        "problem_statement": "Users experience crashes when exporting data.",
        "affected_users": "All users who export data (60%)",
        "user_pain_level": 9,
        "supporting_feedback_count": 15,
        "recommended_action": "Implement robust error handling in the export module.",
        "expected_outcome": "Eliminate export crashes and improve user confidence.",
        "estimated_effort": "Medium",
        "timeframe": "Immediate (0-1 week)",
        "success_metrics": ["Zero crashes", "95% success rate", "40% fewer tickets"],
        "key_quotes": ["App crashes on export", "Lost all my work!"],
        "risk_if_ignored": "Continued churn and negative reviews.",
        "dependencies": []
    }
    
    report_item = ReportItem(**report_item_data)
    print("✅ ReportItem created and validated")
    print(json.dumps(report_item.model_dump(), indent=2)[:500] + "...")
    
    # Test validation
    print("\n4️⃣  VALIDATION TESTING")
    print("─" * 80)
    
    try:
        # This should fail - negative sentiment with high score
        invalid = SentimentResult(
            sentiment="negative",
            score=9,  # Invalid!
            confidence=0.8,
            key_phrases=["bad", "terrible"],
            reasoning="Test"
        )
    except Exception as e:
        print(f"✅ Validation correctly caught error:")
        print(f"   {type(e).__name__}: {str(e)[:100]}")
    
    print("\n" + "=" * 80)
    print("✅ ALL MODELS VALIDATED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


"""
Prompt Templates
Contains all prompt templates used for LLM interactions
"""

from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# SENTIMENT ANALYSIS PROMPTS
# ============================================================================

SENTIMENT_ANALYSIS_PROMPT = """You are an expert sentiment analysis system for customer feedback.

Your task is to analyze the following customer feedback text and classify its sentiment.

**Scoring Guidelines:**
- 1-3: Strongly negative (angry, frustrated, very dissatisfied)
- 4-5: Negative (disappointed, unhappy, critical)
- 6: Neutral (balanced, factual, no strong emotion)
- 7-8: Positive (satisfied, happy, complimentary)
- 9-10: Strongly positive (delighted, enthusiastic, highly satisfied)

**Customer Feedback:**
{feedback_text}

**Instructions:**
1. Determine the overall sentiment (positive, neutral, or negative)
2. Assign a numerical score from 1-10 based on the intensity
3. Identify 2-5 key phrases that influenced your decision
4. Provide a brief reasoning (1-2 sentences)
5. Estimate your confidence level (0.0-1.0)

{format_instructions}

Analyze the feedback and respond with the structured sentiment result:"""


CATEGORY_ANALYSIS_PROMPT = """You are an expert at categorizing customer feedback into themes.

Your task is to analyze the following customer feedback and assign it to relevant categories.

**Available Categories:**
- Bug Report: Technical issues, crashes, errors
- Feature Request: Requests for new functionality
- User Experience: UI/UX feedback, navigation issues
- Performance: Speed, loading times, responsiveness
- Customer Support: Service quality, response time
- Pricing: Cost concerns, value for money
- Documentation: Help content, tutorials, guides
- General Feedback: Praise, complaints, suggestions

**Customer Feedback:**
{feedback_text}

**Instructions:**
1. Identify the primary category that best fits this feedback
2. Optionally identify 1-2 secondary categories if applicable
3. Extract key phrases that relate to each category
4. Provide brief reasoning for your categorization

{format_instructions}

Analyze the feedback and respond with the structured categorization result:"""


FEATURE_SUGGESTION_PROMPT = """You are a product manager analyzing customer feedback to generate actionable feature improvements.

Based on the following customer pain points, generate prioritized feature suggestions.

**Pain Points:**
{pain_points}

**Instructions:**
1. Generate 3-5 specific, actionable feature suggestions
2. Prioritize by potential impact on customer satisfaction
3. Consider feasibility and alignment with common pain points
4. Provide clear action items for the product team

{format_instructions}

Generate the feature improvement suggestions:"""


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def create_sentiment_prompt(format_instructions: str = "") -> ChatPromptTemplate:
    """
    Create a ChatPromptTemplate for sentiment analysis.
    
    Args:
        format_instructions: Optional format instructions for structured output
    
    Returns:
        Configured ChatPromptTemplate
    """
    return ChatPromptTemplate.from_template(
        template=SENTIMENT_ANALYSIS_PROMPT,
        partial_variables={"format_instructions": format_instructions}
    )


def create_category_prompt(format_instructions: str = "") -> ChatPromptTemplate:
    """
    Create a ChatPromptTemplate for category analysis.
    
    Args:
        format_instructions: Optional format instructions for structured output
    
    Returns:
        Configured ChatPromptTemplate
    """
    return ChatPromptTemplate.from_template(
        template=CATEGORY_ANALYSIS_PROMPT,
        partial_variables={"format_instructions": format_instructions}
    )


def create_feature_suggestion_prompt(format_instructions: str = "") -> ChatPromptTemplate:
    """
    Create a ChatPromptTemplate for feature suggestions.
    
    Args:
        format_instructions: Optional format instructions for structured output
    
    Returns:
        Configured ChatPromptTemplate
    """
    return ChatPromptTemplate.from_template(
        template=FEATURE_SUGGESTION_PROMPT,
        partial_variables={"format_instructions": format_instructions}
    )


def create_report_prompt(format_instructions: str = "") -> ChatPromptTemplate:
    """
    Create a ChatPromptTemplate for action plan report generation.
    
    Args:
        format_instructions: Optional format instructions for structured output
    
    Returns:
        Configured ChatPromptTemplate
    """
    template = """You are a product manager creating actionable reports from customer feedback.

**Your Task:**
Analyze the provided feedback items and generate a comprehensive action plan with prioritized recommendations.

**Feedback Data:**
{feedback_data}

**Instructions:**
1. Group related feedback items into actionable themes
2. Prioritize based on:
   - User pain level
   - Number of affected users
   - Business impact
   - Implementation feasibility
3. For each action item, provide:
   - Clear problem statement
   - Affected user segment
   - Recommended action
   - Expected outcome
   - Success metrics
4. Identify quick wins (high impact, low effort)

{format_instructions}

Generate the action plan report:"""
    
    return ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": format_instructions}
    )


# ============================================================================
# CUSTOM PROMPT TEMPLATES
# ============================================================================

def create_custom_analysis_prompt(
    task_description: str,
    guidelines: dict,
    input_variables: list[str],
    format_instructions: str = ""
) -> ChatPromptTemplate:
    """
    Create a custom analysis prompt template.
    
    Args:
        task_description: Description of the analysis task
        guidelines: Dictionary of guideline categories and their content
        input_variables: List of variable names to include in the template
        format_instructions: Optional format instructions
    
    Returns:
        Configured ChatPromptTemplate
    
    Example:
        >>> prompt = create_custom_analysis_prompt(
        ...     task_description="Analyze customer urgency",
        ...     guidelines={"High": "Immediate action needed", "Low": "Can wait"},
        ...     input_variables=["feedback_text"]
        ... )
    """
    guidelines_text = "\n".join([
        f"- {key}: {value}" for key, value in guidelines.items()
    ])
    
    variables_template = "\n".join([
        f"**{var.replace('_', ' ').title()}:**\n{{{var}}}\n"
        for var in input_variables
    ])
    
    template = f"""You are an expert analyst for customer feedback.

{task_description}

**Guidelines:**
{guidelines_text}

{variables_template}

{{format_instructions}}

Provide your analysis:"""
    
    return ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": format_instructions}
    )


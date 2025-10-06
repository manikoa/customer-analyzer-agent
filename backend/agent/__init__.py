"""
Customer Feedback Analyzer Agent
The core brain and orchestration logic for the agent system
"""

from .agent import FeedbackAnalyzerAgent, AnalysisResult, analyze
from .orchestrator import WorkflowOrchestrator
from .executor import WorkflowExecutor

__all__ = [
    'FeedbackAnalyzerAgent',
    'AnalysisResult',
    'analyze',
    'WorkflowOrchestrator',
    'WorkflowExecutor'
]

__version__ = "1.0.0"


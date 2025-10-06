"""
FeedbackAnalyzerAgent - The Brain of the System
Main agent class that coordinates all components
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .config import AgentConfig
    from .orchestrator import WorkflowOrchestrator
except ImportError:
    from config.settings import AgentConfig
    from agent.orchestrator import WorkflowOrchestrator

from core.state import FeedbackAnalysisState, create_initial_state, get_state_summary
from data.loader import load_feedback_for_workflow
from tools.database_tools import log_to_db_tool, get_database_stats
from tools.export_tools import export_analysis_results
from tools.notification_tools import create_alert_tool, send_email_notification_tool


class FeedbackAnalyzerAgent:
    """
    The Brain of the Customer Feedback Analyzer Agent.
    
    This class coordinates all components of the system:
    - Data loading and validation
    - Workflow execution with LangGraph
    - Intelligent triage and routing
    - Results persistence and export
    - Notifications and alerts
    
    Attributes:
        config: Agent configuration
        orchestrator: Workflow orchestrator
        state: Current workflow state
        history: Execution history
    
    Example:
        >>> agent = FeedbackAnalyzerAgent(
        ...     llm_provider="gemini",
        ...     temperature=0.0
        ... )
        >>> results = agent.analyze_feedback(
        ...     feedback_items=["Great app!", "Too slow"],
        ...     enable_notifications=True
        ... )
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        llm_provider: str = "gemini",
        llm_model: Optional[str] = None,
        temperature: float = 0.0,
        batch_size: int = 5,
        enable_database: bool = True,
        enable_export: bool = True,
        enable_notifications: bool = True,
        output_dir: str = "output",
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize the Feedback Analyzer Agent.
        
        Args:
            llm_provider: LLM provider (gemini, openai, anthropic)
            llm_model: Optional specific model name
            temperature: LLM temperature (0.0 = deterministic)
            batch_size: Batch size for processing
            enable_database: Enable database logging
            enable_export: Enable automatic export
            enable_notifications: Enable notifications/alerts
            output_dir: Output directory for exports
            config: Optional pre-configured AgentConfig
        """
        # Configuration
        self.config = config or AgentConfig(
            llm_provider=llm_provider,
            llm_model=llm_model,
            temperature=temperature,
            batch_size=batch_size,
            enable_database=enable_database,
            enable_export=enable_export,
            enable_notifications=enable_notifications,
            output_dir=output_dir
        )
        
        # Orchestrator
        self.orchestrator = WorkflowOrchestrator(self.config)
        
        # State
        self.state: Optional[FeedbackAnalysisState] = None
        self.history: List[Dict[str, Any]] = []
        
        # Initialization time
        self.initialized_at = datetime.now()
        
        print(f"✅ FeedbackAnalyzerAgent initialized")
        print(f"   Provider: {self.config.llm_provider}")
        print(f"   Temperature: {self.config.temperature}")
        print(f"   Features: Database={enable_database}, "
              f"Export={enable_export}, Notifications={enable_notifications}")
    
    def analyze_feedback(
        self,
        feedback_items: List[str] = None,
        feedback_file: str = None,
        limit: Optional[int] = None,
        sources: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> 'AnalysisResult':
        """
        Analyze customer feedback (main entry point).
        
        Args:
            feedback_items: List of feedback text strings
            feedback_file: Path to CSV file with feedback
            limit: Optional limit on items to process
            sources: Optional filter by sources
            metadata: Additional metadata
            verbose: Print progress information
        
        Returns:
            AnalysisResult object with all results
        
        Example:
            >>> agent = FeedbackAnalyzerAgent()
            >>> result = agent.analyze_feedback(
            ...     feedback_file="feedback_data.csv",
            ...     limit=20,
            ...     verbose=True
            ... )
        """
        if verbose:
            print("\n" + "="*80)
            print("🧠 FEEDBACK ANALYZER AGENT - STARTING ANALYSIS")
            print("="*80)
        
        # Step 1: Load feedback data
        if feedback_items is None:
            if feedback_file is None:
                raise ValueError("Either feedback_items or feedback_file must be provided")
            
            if verbose:
                print(f"\n📊 Loading feedback from: {feedback_file}")
            
            feedback_items = load_feedback_for_workflow(
                filepath=feedback_file,
                limit=limit,
                sources=sources
            )
        
        if verbose:
            print(f"✅ Loaded {len(feedback_items)} feedback items")
        
        # Step 2: Initialize state
        self.state = create_initial_state(
            feedback_items=feedback_items,
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
            temperature=self.config.temperature,
            batch_size=self.config.batch_size,
            metadata=metadata or {}
        )
        
        # Step 3: Execute workflow
        if verbose:
            print(f"\n🔄 Executing workflow...")
        
        try:
            final_state = self.orchestrator.run(self.state, verbose=verbose)
            self.state = final_state
            
            if verbose:
                print(f"\n✅ Workflow completed successfully")
            
        except Exception as e:
            if verbose:
                print(f"\n❌ Workflow failed: {str(e)}")
            raise
        
        # Step 4: Post-processing
        self._post_process(verbose=verbose)
        
        # Step 5: Create result object
        result = AnalysisResult(agent=self, state=final_state)
        
        # Step 6: Add to history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'items_processed': len(feedback_items),
            'status': final_state['workflow_status'],
            'summary': get_state_summary(final_state)
        })
        
        if verbose:
            print("\n" + "="*80)
            print("✅ ANALYSIS COMPLETE")
            print("="*80)
            result.print_summary()
        
        return result
    
    def _post_process(self, verbose: bool = False):
        """Post-process results (database, export, notifications)."""
        if not self.state:
            return
        
        # Database logging
        if self.config.enable_database:
            if verbose:
                print(f"\n💾 Logging to database...")
            self._log_to_database()
        
        # Export results
        if self.config.enable_export:
            if verbose:
                print(f"\n📤 Exporting results...")
            self._export_results()
        
        # Handle notifications
        if self.config.enable_notifications:
            if verbose:
                print(f"\n🔔 Processing notifications...")
            self._handle_notifications()
    
    def _log_to_database(self):
        """Log results to database."""
        try:
            enriched_feedback = self.state.get("enriched_feedback", [])
            
            for item in enriched_feedback:
                log_to_db_tool.invoke({"result": item})
            
        except Exception as e:
            print(f"⚠️  Database logging error: {str(e)}")
    
    def _export_results(self):
        """Export results to files."""
        try:
            report = self.state.get("action_plan_report")
            
            if report:
                export_analysis_results.invoke({
                    "results": {
                        "report": report.model_dump(),
                        "summary_stats": self.state.get("summary_stats"),
                        "metadata": self.state.get("metadata")
                    },
                    "output_dir": self.config.output_dir,
                    "formats": ["json", "markdown"]
                })
        
        except Exception as e:
            print(f"⚠️  Export error: {str(e)}")
    
    def _handle_notifications(self):
        """Handle critical issue notifications."""
        try:
            metadata = self.state.get("metadata", {})
            
            # Check for escalated/critical items
            if metadata.get("escalated"):
                critical_count = metadata.get("requires_immediate_attention", 0)
                
                if critical_count > 0:
                    # Create alert
                    create_alert_tool.invoke({
                        "title": f"{critical_count} Critical Issues Detected",
                        "description": f"Feedback analysis found {critical_count} issues requiring immediate attention",
                        "severity": "critical",
                        "category": "feedback_analysis",
                        "metadata": {"workflow_id": self.state.get("metadata", {}).get("workflow_id")}
                    })
                    
                    # Send notification (if configured)
                    if self.config.notification_email:
                        send_email_notification_tool.invoke({
                            "recipient": self.config.notification_email,
                            "subject": f"URGENT: {critical_count} Critical Feedback Issues",
                            "body": f"Analysis completed. {critical_count} critical issues require immediate attention.",
                            "priority": "critical"
                        })
        
        except Exception as e:
            print(f"⚠️  Notification error: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            "agent_version": "1.0.0",
            "initialized_at": self.initialized_at.isoformat(),
            "config": self.config.to_dict(),
            "total_runs": len(self.history),
            "current_state": get_state_summary(self.state) if self.state else None
        }
        
        if self.config.enable_database:
            try:
                db_stats = get_database_stats.invoke({})
                stats["database"] = db_stats
            except:
                pass
        
        return stats
    
    def print_history(self):
        """Print execution history."""
        print("\n" + "="*80)
        print("📜 EXECUTION HISTORY")
        print("="*80)
        
        if not self.history:
            print("No executions yet")
            return
        
        for i, entry in enumerate(self.history, 1):
            print(f"\n{i}. {entry['timestamp']}")
            print(f"   Items: {entry['items_processed']}")
            print(f"   Status: {entry['status']}")
            if 'summary' in entry:
                print(f"   Progress: {entry['summary'].get('progress_percentage', 0):.1f}%")
        
        print("\n" + "="*80)


class AnalysisResult:
    """
    Container for analysis results.
    
    Provides easy access to all analysis outputs and utilities
    for working with the results.
    """
    
    def __init__(self, agent: FeedbackAnalyzerAgent, state: FeedbackAnalysisState):
        """Initialize with agent and state."""
        self.agent = agent
        self.state = state
        self._summary = get_state_summary(state)
    
    @property
    def report(self):
        """Get the action plan report."""
        return self.state.get("action_plan_report")
    
    @property
    def sentiment_results(self):
        """Get sentiment analysis results."""
        return self.state.get("sentiment_results", [])
    
    @property
    def category_results(self):
        """Get category classification results."""
        return self.state.get("category_results", [])
    
    @property
    def enriched_feedback(self):
        """Get enriched feedback data."""
        return self.state.get("enriched_feedback", [])
    
    @property
    def stats(self):
        """Get summary statistics."""
        return self.state.get("summary_stats")
    
    @property
    def metadata(self):
        """Get metadata."""
        return self.state.get("metadata", {})
    
    def summary(self) -> Dict[str, Any]:
        """Get result summary."""
        return self._summary
    
    def print_summary(self):
        """Print a formatted summary."""
        print("\n" + "="*80)
        print("📊 ANALYSIS RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n✅ Status: {self._summary['status']}")
        print(f"📈 Processed: {self._summary['processed_items']}/{self._summary['total_items']}")
        print(f"📊 Success Rate: {self._summary['progress_percentage']:.1f}%")
        
        if self.report:
            print(f"\n📋 Action Plan Report:")
            print(f"   Title: {self.report.report_title}")
            print(f"   Action Items: {len(self.report.action_items)}")
            print(f"   Overall Sentiment: {self.report.overall_sentiment}")
            print(f"   Quick Wins: {len(self.report.quick_wins)}")
        
        if self.stats:
            print(f"\n📊 Statistics:")
            print(f"   Avg Sentiment: {self.stats.get('avg_sentiment_score', 0):.1f}/10")
            print(f"   Critical Items: {self.stats.get('critical_items_count', 0)}")
        
        if self.metadata.get("escalated"):
            print(f"\n🚨 Escalation:")
            print(f"   Critical Issues: {self.metadata.get('escalation_count', 0)}")
            print(f"   Immediate Attention: {self.metadata.get('requires_immediate_attention', 0)}")
        
        errors = self.state.get("errors", [])
        if errors:
            print(f"\n⚠️  Errors: {len(errors)}")
        
        print("\n" + "="*80)
    
    def export(self, output_dir: str = None, formats: List[str] = None):
        """Export results to files."""
        output_dir = output_dir or self.agent.config.output_dir
        formats = formats or ["json", "markdown"]
        
        if self.report:
            export_analysis_results.invoke({
                "results": {
                    "report": self.report.model_dump(),
                    "stats": self.stats,
                    "metadata": self.metadata
                },
                "output_dir": output_dir,
                "formats": formats
            })
            
            print(f"✅ Results exported to: {output_dir}")
    
    def get_top_issues(self, n: int = 5) -> List[Dict]:
        """Get top N issues by priority."""
        if not self.report:
            return []
        
        items = sorted(
            self.report.action_items,
            key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x.priority, 0),
                -x.user_pain_level
            ),
            reverse=True
        )
        
        return [
            {
                "title": item.title,
                "priority": item.priority,
                "category": item.category,
                "pain_level": item.user_pain_level,
                "affected_users": item.affected_users
            }
            for item in items[:n]
        ]
    
    def get_critical_alerts(self) -> List[Dict]:
        """Get critical items requiring immediate attention."""
        critical = []
        raw_feedback = self.state.get("raw_feedback_items", [])
        
        for i, (sent, cat) in enumerate(zip(self.sentiment_results, self.category_results)):
            # Bounds check
            if i >= len(raw_feedback):
                break
            if sent and cat:
                if cat.urgency == "critical" or sent.score <= 2:
                    critical.append({
                        "index": i,
                        "feedback": raw_feedback[i],
                        "sentiment": sent.sentiment,
                        "score": sent.score,
                        "category": cat.primary_category,
                        "urgency": cat.urgency
                    })
        
        return critical


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze(
    feedback_items: List[str] = None,
    feedback_file: str = None,
    llm_provider: str = "gemini",
    verbose: bool = True,
    **kwargs
) -> AnalysisResult:
    """
    Convenience function for quick analysis.
    
    Args:
        feedback_items: List of feedback strings
        feedback_file: Path to feedback CSV
        llm_provider: LLM provider
        verbose: Print progress
        **kwargs: Additional agent configuration
    
    Returns:
        AnalysisResult object
    
    Example:
        >>> result = analyze(feedback_file="feedback_data.csv", limit=10)
        >>> result.print_summary()
    """
    agent = FeedbackAnalyzerAgent(llm_provider=llm_provider, **kwargs)
    return agent.analyze_feedback(
        feedback_items=feedback_items,
        feedback_file=feedback_file,
        verbose=verbose
    )


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the agent."""
    print("\n" + "="*80)
    print("FEEDBACK ANALYZER AGENT DEMONSTRATION")
    print("="*80)
    
    # Create agent
    print("\n1️⃣  Creating agent...")
    agent = FeedbackAnalyzerAgent(
        llm_provider="gemini",
        temperature=0.0,
        enable_database=True,
        enable_export=True,
        enable_notifications=False
    )
    
    # Sample feedback
    sample_feedback = [
        "The app crashes every time I export data. Critical bug!",
        "Love the new interface! So much better than before.",
        "Customer support is terrible. No response in 5 days."
    ]
    
    print("\n2️⃣  Analyzing feedback...")
    print(f"   (Using {len(sample_feedback)} sample items)")
    
    # Check if API key is available
    from utils.llm import get_available_providers
    providers = get_available_providers()
    
    if not providers:
        print("\n⚠️  No LLM providers available")
        print("   Set API key to run full analysis:")
        print("   export GOOGLE_API_KEY='your-key'")
        print("\n✅ Agent structure demonstrated successfully")
        return
    
    # Run analysis
    result = agent.analyze_feedback(
        feedback_items=sample_feedback,
        verbose=True
    )
    
    # Show results
    print("\n3️⃣  Results:")
    print(f"   Top issues: {len(result.get_top_issues(3))}")
    print(f"   Critical alerts: {len(result.get_critical_alerts())}")
    
    # Show stats
    print("\n4️⃣  Agent statistics:")
    stats = agent.get_stats()
    print(f"   Total runs: {stats['total_runs']}")
    print(f"   Provider: {stats['config']['llm_provider']}")
    
    print("\n" + "="*80)
    print("✅ AGENT DEMONSTRATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


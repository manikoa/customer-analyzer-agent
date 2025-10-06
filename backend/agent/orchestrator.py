"""
Workflow Orchestrator
Coordinates workflow execution and manages the LangGraph workflow
"""

from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .config import AgentConfig
except ImportError:
    from config.settings import AgentConfig

from core.state import FeedbackAnalysisState
from core.workflow import create_workflow


class WorkflowOrchestrator:
    """
    Orchestrates the execution of the feedback analysis workflow.
    
    This class manages the LangGraph workflow, handles execution,
    and coordinates between different components of the system.
    
    Attributes:
        config: Agent configuration
        workflow: Compiled LangGraph workflow
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the orchestrator.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.workflow = None
        self._compile_workflow()
    
    def _compile_workflow(self):
        """Compile the LangGraph workflow."""
        workflow_graph = create_workflow()
        self.workflow = workflow_graph.compile()
        print("✅ Workflow compiled")
    
    def run(
        self,
        state: FeedbackAnalysisState,
        verbose: bool = True
    ) -> FeedbackAnalysisState:
        """
        Execute the workflow on the given state.
        
        Args:
            state: Initial workflow state
            verbose: Print progress information
        
        Returns:
            Final workflow state after execution
        
        Raises:
            Exception: If workflow execution fails
        """
        if not self.workflow:
            raise RuntimeError("Workflow not compiled")
        
        if verbose:
            print("\n🔄 Starting workflow execution...")
            print(f"   Total items: {state['total_items']}")
            print(f"   Provider: {state['llm_provider']}")
        
        try:
            # Execute workflow
            final_state = self.workflow.invoke(state)
            
            if verbose:
                print("✅ Workflow execution complete")
            
            return final_state
            
        except Exception as e:
            if verbose:
                print(f"❌ Workflow execution failed: {str(e)}")
            raise
    
    def stream(self, state: FeedbackAnalysisState):
        """
        Stream workflow execution (yields intermediate states).
        
        Args:
            state: Initial workflow state
        
        Yields:
            Intermediate states during execution
        """
        if not self.workflow:
            raise RuntimeError("Workflow not compiled")
        
        for s in self.workflow.stream(state):
            yield s
    
    def get_graph_structure(self) -> Dict[str, Any]:
        """
        Get the structure of the workflow graph.
        
        Returns:
            Dictionary describing the graph structure
        """
        if not self.workflow:
            return {"error": "Workflow not compiled"}
        
        return {
            "nodes": [
                "initialize",
                "sentiment_analysis",
                "category_classification",
                "enrich_feedback",
                "escalate_handler",
                "group_handler",
                "report_generation",
                "compute_statistics",
                "validate_results",
                "error_handler",
                "finalize"
            ],
            "conditional_edges": [
                {
                    "from": "enrich_feedback",
                    "type": "triage",
                    "routes": ["escalate_handler", "group_handler", "report_generation"]
                },
                {
                    "from": "report_generation",
                    "type": "error_check",
                    "routes": ["error_handler", "compute_statistics"]
                }
            ],
            "entry_point": "initialize",
            "end_point": "finalize"
        }
    
    def visualize_workflow(self):
        """Print a visual representation of the workflow."""
        print("\n" + "="*80)
        print("WORKFLOW STRUCTURE")
        print("="*80)
        
        structure = self.get_graph_structure()
        
        print("\n📊 Nodes:")
        for node in structure["nodes"]:
            print(f"   • {node}")
        
        print("\n🔀 Conditional Routing:")
        for edge in structure["conditional_edges"]:
            print(f"\n   {edge['from']} → [{edge['type'].upper()}]")
            for route in edge["routes"]:
                print(f"     ├─ {route}")
        
        print(f"\n🚀 Entry Point: {structure['entry_point']}")
        print(f"🏁 End Point: {structure['end_point']}")
        
        print("\n" + "="*80 + "\n")


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the orchestrator."""
    print("\n" + "="*80)
    print("WORKFLOW ORCHESTRATOR DEMONSTRATION")
    print("="*80)
    
    try:
        from .config import get_development_config
    except ImportError:
        from config import get_development_config
    
    # Create orchestrator
    print("\n1️⃣  Creating orchestrator...")
    config = get_development_config()
    orchestrator = WorkflowOrchestrator(config)
    
    # Visualize workflow
    print("\n2️⃣  Workflow structure:")
    orchestrator.visualize_workflow()
    
    # Get graph structure
    print("\n3️⃣  Graph structure details:")
    structure = orchestrator.get_graph_structure()
    print(f"   Total nodes: {len(structure['nodes'])}")
    print(f"   Conditional edges: {len(structure['conditional_edges'])}")
    
    print("\n" + "="*80)
    print("✅ ORCHESTRATOR DEMO COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


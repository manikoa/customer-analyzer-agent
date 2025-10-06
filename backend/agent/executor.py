"""
Workflow Executor
Advanced execution engine with retry logic, monitoring, and fault tolerance
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .config import AgentConfig
except ImportError:
    from config.settings import AgentConfig

from core.state import FeedbackAnalysisState


class WorkflowExecutor:
    """
    Advanced executor with retry logic and monitoring.
    
    Provides enhanced execution capabilities including:
    - Automatic retry on failure
    - Execution monitoring
    - Performance metrics
    - Fault tolerance
    
    Attributes:
        config: Agent configuration
        metrics: Execution metrics
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the executor.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.metrics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
    
    def execute_with_retry(
        self,
        workflow_func: Callable,
        state: FeedbackAnalysisState,
        max_retries: Optional[int] = None,
        verbose: bool = True
    ) -> FeedbackAnalysisState:
        """
        Execute workflow with automatic retry on failure.
        
        Args:
            workflow_func: Workflow function to execute
            state: Initial state
            max_retries: Maximum retry attempts (uses config default if None)
            verbose: Print retry information
        
        Returns:
            Final state after successful execution
        
        Raises:
            Exception: If all retries are exhausted
        """
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                if attempt > 0 and verbose:
                    print(f"🔄 Retry attempt {attempt}/{max_retries - 1}...")
                
                result = workflow_func(state)
                
                if verbose and attempt > 0:
                    print(f"✅ Succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    if verbose:
                        print(f"⚠️  Attempt {attempt + 1} failed: {str(e)}")
                        print(f"   Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    if verbose:
                        print(f"❌ All {max_retries} attempts failed")
                    raise
    
    def execute_with_monitoring(
        self,
        workflow_func: Callable,
        state: FeedbackAnalysisState,
        on_progress: Optional[Callable] = None,
        verbose: bool = True
    ) -> FeedbackAnalysisState:
        """
        Execute workflow with monitoring and progress updates.
        
        Args:
            workflow_func: Workflow function to execute
            state: Initial state
            on_progress: Optional callback for progress updates
            verbose: Print progress information
        
        Returns:
            Final state after execution
        """
        start_time = time.time()
        self.metrics["total_executions"] += 1
        
        try:
            if verbose:
                print(f"🚀 Starting monitored execution...")
                print(f"   Start time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Execute workflow
            result = workflow_func(state)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["successful_executions"] += 1
            self.metrics["total_execution_time"] += execution_time
            self.metrics["average_execution_time"] = (
                self.metrics["total_execution_time"] / 
                self.metrics["successful_executions"]
            )
            
            if verbose:
                print(f"✅ Execution completed")
                print(f"   Duration: {execution_time:.2f}s")
            
            if on_progress:
                on_progress({
                    "status": "completed",
                    "duration": execution_time,
                    "items_processed": result.get("processed_items", 0)
                })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics["failed_executions"] += 1
            
            if verbose:
                print(f"❌ Execution failed after {execution_time:.2f}s")
                print(f"   Error: {str(e)}")
            
            if on_progress:
                on_progress({
                    "status": "failed",
                    "duration": execution_time,
                    "error": str(e)
                })
            
            raise
    
    def execute_with_timeout(
        self,
        workflow_func: Callable,
        state: FeedbackAnalysisState,
        timeout: Optional[int] = None,
        verbose: bool = True
    ) -> FeedbackAnalysisState:
        """
        Execute workflow with timeout.
        
        Args:
            workflow_func: Workflow function to execute
            state: Initial state
            timeout: Timeout in seconds (uses config default if None)
            verbose: Print timeout information
        
        Returns:
            Final state if completed within timeout
        
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        timeout = timeout or self.config.timeout_seconds
        start_time = time.time()
        
        if verbose:
            print(f"⏱️  Executing with {timeout}s timeout...")
        
        try:
            result = workflow_func(state)
            
            elapsed = time.time() - start_time
            if verbose:
                print(f"✅ Completed in {elapsed:.2f}s (within timeout)")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                raise TimeoutError(f"Execution exceeded {timeout}s timeout")
            else:
                raise e
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics.
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_executions"] / 
                self.metrics["total_executions"] * 100
                if self.metrics["total_executions"] > 0 else 0
            )
        }
    
    def print_metrics(self):
        """Print execution metrics."""
        print("\n" + "="*60)
        print("EXECUTION METRICS")
        print("="*60)
        
        metrics = self.get_metrics()
        
        print(f"\n📊 Executions:")
        print(f"   Total: {metrics['total_executions']}")
        print(f"   Successful: {metrics['successful_executions']}")
        print(f"   Failed: {metrics['failed_executions']}")
        print(f"   Success Rate: {metrics['success_rate']:.1f}%")
        
        print(f"\n⏱️  Performance:")
        print(f"   Total Time: {metrics['total_execution_time']:.2f}s")
        print(f"   Average Time: {metrics['average_execution_time']:.2f}s")
        
        print("\n" + "="*60 + "\n")
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        print("✅ Metrics reset")


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the executor."""
    print("\n" + "="*80)
    print("WORKFLOW EXECUTOR DEMONSTRATION")
    print("="*80)
    
    try:
        from .config import get_development_config
    except ImportError:
        from config.settings import get_development_config
    
    # Create executor
    print("\n1️⃣  Creating executor...")
    config = get_development_config()
    executor = WorkflowExecutor(config)
    
    # Demo function
    def demo_workflow(state):
        print("   Executing demo workflow...")
        time.sleep(0.5)  # Simulate work
        state["processed_items"] = state["total_items"]
        return state
    
    # Create test state
    from core.state import create_initial_state
    state = create_initial_state(
        feedback_items=["test1", "test2", "test3"],
        llm_provider="gemini"
    )
    
    # Test monitored execution
    print("\n2️⃣  Monitored execution:")
    result = executor.execute_with_monitoring(demo_workflow, state, verbose=True)
    print(f"   Result: {result['processed_items']} items processed")
    
    # Show metrics
    print("\n3️⃣  Execution metrics:")
    executor.print_metrics()
    
    print("\n" + "="*80)
    print("✅ EXECUTOR DEMO COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


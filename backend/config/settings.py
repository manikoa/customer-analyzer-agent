"""
Agent Configuration
Configuration management for the Feedback Analyzer Agent
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class AgentConfig:
    """
    Configuration for the Feedback Analyzer Agent.
    
    Attributes:
        llm_provider: LLM provider (gemini, openai, anthropic)
        llm_model: Optional specific model name
        temperature: LLM temperature (0.0-1.0)
        batch_size: Processing batch size
        enable_database: Enable database logging
        enable_export: Enable automatic export
        enable_notifications: Enable notifications
        output_dir: Output directory for exports
        notification_email: Email for critical notifications
        db_path: Database file path
        max_retries: Maximum retry attempts
        timeout_seconds: Operation timeout
    """
    
    # LLM Configuration
    llm_provider: str = "gemini"
    llm_model: Optional[str] = None
    temperature: float = 0.0
    
    # Processing Configuration
    batch_size: int = 5
    max_retries: int = 3
    timeout_seconds: int = 300
    
    # Feature Flags
    enable_database: bool = True
    enable_export: bool = True
    enable_notifications: bool = True
    
    # Paths
    output_dir: str = "output"
    db_path: str = "feedback_analysis.db"
    
    # Notifications
    notification_email: Optional[str] = None
    slack_webhook: Optional[str] = None
    
    # Thresholds
    critical_threshold: float = 0.2  # 20% critical issues triggers alert
    negative_threshold: float = 0.5   # 50% negative sentiment triggers alert
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, file_path: str):
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Configuration saved to: {file_path}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'AgentConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Re-validate
        self.__post_init__()
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("\n" + "="*60)
        print("AGENT CONFIGURATION")
        print("="*60)
        
        print("\n🤖 LLM Settings:")
        print(f"   Provider: {self.llm_provider}")
        print(f"   Model: {self.llm_model or 'default'}")
        print(f"   Temperature: {self.temperature}")
        
        print("\n⚙️  Processing:")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Max Retries: {self.max_retries}")
        print(f"   Timeout: {self.timeout_seconds}s")
        
        print("\n🔧 Features:")
        print(f"   Database: {'✅' if self.enable_database else '❌'}")
        print(f"   Export: {'✅' if self.enable_export else '❌'}")
        print(f"   Notifications: {'✅' if self.enable_notifications else '❌'}")
        
        print("\n📁 Paths:")
        print(f"   Output Dir: {self.output_dir}")
        print(f"   Database: {self.db_path}")
        
        print("\n🔔 Alerts:")
        print(f"   Email: {self.notification_email or 'Not configured'}")
        print(f"   Slack: {self.slack_webhook or 'Not configured'}")
        
        print("\n📊 Thresholds:")
        print(f"   Critical: {self.critical_threshold*100}%")
        print(f"   Negative: {self.negative_threshold*100}%")
        
        print("\n" + "="*60 + "\n")


# Note: Preset configurations have been moved to presets.py
# Import them from config.presets or config package


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo configuration management."""
    print("\n" + "="*80)
    print("AGENT CONFIGURATION DEMONSTRATION")
    print("="*80)
    
    # 1. Create default config
    print("\n1️⃣  Default Configuration:")
    config = AgentConfig()
    config.print_config()
    
    # 2. Save and load
    print("\n2️⃣  Save/Load Configuration:")
    config.to_json("agent_config.json")
    loaded = AgentConfig.from_json("agent_config.json")
    print(f"   ✅ Configuration saved and loaded")
    print(f"   Provider: {loaded.llm_provider}")
    
    # 3. Update config
    print("\n3️⃣  Update Configuration:")
    config.update(temperature=0.5, batch_size=10)
    print(f"   ✅ Updated temperature: {config.temperature}")
    print(f"   ✅ Updated batch_size: {config.batch_size}")
    
    # 4. Note about presets
    print("\n4️⃣  Configuration Presets:")
    print("   ℹ️  Presets have been moved to config.presets")
    print("   Import: from config.presets import get_development_config")
    print("   Or: from config import get_development_config")
    
    print("\n" + "="*80)
    print("✅ CONFIGURATION DEMO COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


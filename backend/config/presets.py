"""
Configuration Presets
Pre-configured settings for common use cases
"""

from .settings import AgentConfig


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_development_config() -> AgentConfig:
    """
    Get development configuration.
    
    Features:
    - All features enabled except notifications
    - Small batch size for faster iteration
    - Verbose logging
    - Development output directory
    
    Returns:
        AgentConfig configured for development
    """
    return AgentConfig(
        llm_provider="gemini",
        temperature=0.0,
        batch_size=5,
        enable_database=True,
        enable_export=True,
        enable_notifications=False,  # Disabled in dev
        output_dir="dev_output"
    )


def get_production_config() -> AgentConfig:
    """
    Get production configuration.
    
    Features:
    - All features enabled
    - Larger batch size for efficiency
    - Higher retry attempts
    - Longer timeout
    - Production output directory
    
    Returns:
        AgentConfig configured for production
    """
    return AgentConfig(
        llm_provider="gemini",
        temperature=0.0,
        batch_size=10,
        enable_database=True,
        enable_export=True,
        enable_notifications=True,
        output_dir="production_output",
        max_retries=5,
        timeout_seconds=600
    )


def get_minimal_config() -> AgentConfig:
    """
    Get minimal configuration.
    
    Features:
    - All features disabled
    - Large batch size for speed
    - Minimal overhead
    - Temporary output directory
    
    Returns:
        AgentConfig configured for minimal operation
    """
    return AgentConfig(
        llm_provider="gemini",
        temperature=0.0,
        batch_size=20,
        enable_database=False,
        enable_export=False,
        enable_notifications=False,
        output_dir="temp_output"
    )


def get_testing_config() -> AgentConfig:
    """
    Get testing configuration.
    
    Features:
    - All features disabled
    - Small batch size for unit tests
    - Fast iteration
    - Test output directory
    
    Returns:
        AgentConfig configured for testing
    """
    return AgentConfig(
        llm_provider="gemini",
        temperature=0.0,
        batch_size=2,
        enable_database=False,
        enable_export=False,
        enable_notifications=False,
        output_dir="test_output"
    )


def get_custom_config(
    provider: str = "gemini",
    batch_size: int = 5,
    enable_features: bool = True
) -> AgentConfig:
    """
    Get a custom configuration with common parameters.
    
    Args:
        provider: LLM provider (gemini, openai, anthropic)
        batch_size: Processing batch size
        enable_features: Enable all features (database, export, notifications)
    
    Returns:
        Custom AgentConfig
    """
    return AgentConfig(
        llm_provider=provider,
        temperature=0.0,
        batch_size=batch_size,
        enable_database=enable_features,
        enable_export=enable_features,
        enable_notifications=enable_features,
        output_dir="custom_output"
    )


# ============================================================================
# PRESET MAPPING
# ============================================================================

PRESETS = {
    "development": get_development_config,
    "dev": get_development_config,
    "production": get_production_config,
    "prod": get_production_config,
    "minimal": get_minimal_config,
    "min": get_minimal_config,
    "testing": get_testing_config,
    "test": get_testing_config,
}


def get_preset(name: str) -> AgentConfig:
    """
    Get a configuration preset by name.
    
    Args:
        name: Preset name (development, production, minimal, testing)
    
    Returns:
        AgentConfig for the specified preset
    
    Raises:
        ValueError: If preset name is not recognized
    
    Example:
        >>> config = get_preset("production")
        >>> config = get_preset("dev")  # alias for development
    """
    name_lower = name.lower()
    
    if name_lower not in PRESETS:
        available = ", ".join(sorted(set(PRESETS.keys())))
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {available}"
        )
    
    return PRESETS[name_lower]()


def list_presets() -> list:
    """
    List all available preset names.
    
    Returns:
        List of preset names
    
    Example:
        >>> presets = list_presets()
        >>> print(f"Available presets: {presets}")
    """
    return sorted(set(PRESETS.keys()))


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo configuration presets."""
    print("\n" + "="*80)
    print("CONFIGURATION PRESETS DEMONSTRATION")
    print("="*80)
    
    # List all presets
    print("\n1️⃣  Available Presets:")
    for preset_name in list_presets():
        print(f"   • {preset_name}")
    
    # Development config
    print("\n2️⃣  Development Configuration:")
    dev_config = get_development_config()
    print(f"   Provider: {dev_config.llm_provider}")
    print(f"   Batch Size: {dev_config.batch_size}")
    print(f"   Notifications: {dev_config.enable_notifications}")
    
    # Production config
    print("\n3️⃣  Production Configuration:")
    prod_config = get_production_config()
    print(f"   Batch Size: {prod_config.batch_size}")
    print(f"   Max Retries: {prod_config.max_retries}")
    print(f"   Timeout: {prod_config.timeout_seconds}s")
    
    # Get by name
    print("\n4️⃣  Get Preset by Name:")
    config = get_preset("prod")
    print(f"   ✅ Retrieved 'prod' preset")
    print(f"   Provider: {config.llm_provider}")
    
    # Custom config
    print("\n5️⃣  Custom Configuration:")
    custom = get_custom_config(provider="openai", batch_size=15, enable_features=True)
    print(f"   Provider: {custom.llm_provider}")
    print(f"   Batch Size: {custom.batch_size}")
    
    print("\n" + "="*80)
    print("✅ PRESETS DEMO COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


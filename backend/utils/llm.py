"""
LLM Configuration
Handles initialization and configuration of language models
"""

import os
from enum import Enum
from typing import Optional, Dict, Any


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


# ============================================================================
# PROVIDER CONFIGURATION
# ============================================================================

DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.GEMINI: "gemini-2.0-flash-exp",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022"
}

ENV_VAR_NAMES = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.GEMINI: "GOOGLE_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY"
}


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def get_llm(
    provider: str = "gemini",
    temperature: float = 0.0,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
):
    """
    Initialize and return a configured language model.
    
    Args:
        provider: LLM provider name ("openai", "gemini", or "anthropic")
        temperature: Model temperature (0.0 for deterministic, higher for creative)
        model: Optional specific model name (uses default if not provided)
        api_key: Optional API key (uses environment variable if not provided)
        **kwargs: Additional provider-specific parameters
    
    Returns:
        Configured LLM instance
    
    Raises:
        ValueError: If provider is invalid or API key is missing
        ImportError: If provider package is not installed
    
    Example:
        >>> llm = get_llm(provider="gemini", temperature=0.0)
        >>> llm = get_llm(provider="openai", model="gpt-4", temperature=0.7)
    """
    provider = provider.lower()
    
    try:
        provider_enum = LLMProvider(provider)
    except ValueError:
        valid_providers = [p.value for p in LLMProvider]
        raise ValueError(
            f"Invalid provider: {provider}. Choose from: {', '.join(valid_providers)}"
        )
    
    # Get API key from parameter or environment
    if api_key is None:
        env_var = ENV_VAR_NAMES[provider_enum]
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"{env_var} environment variable not set. "
                f"Set it with: export {env_var}='your-api-key'"
            )
    
    # Get model name
    model_name = model or DEFAULT_MODELS[provider_enum]
    
    # Initialize provider-specific LLM
    if provider_enum == LLMProvider.OPENAI:
        return _init_openai(model_name, temperature, api_key, **kwargs)
    elif provider_enum == LLMProvider.GEMINI:
        return _init_gemini(model_name, temperature, api_key, **kwargs)
    elif provider_enum == LLMProvider.ANTHROPIC:
        return _init_anthropic(model_name, temperature, api_key, **kwargs)


def _init_openai(model: str, temperature: float, api_key: str, **kwargs):
    """Initialize OpenAI LLM."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "OpenAI not available. Install with: pip install langchain-openai"
        )
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs
    )


def _init_gemini(model: str, temperature: float, api_key: str, **kwargs):
    """Initialize Google Gemini LLM."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "Gemini not available. Install with: pip install langchain-google-genai"
        )
    
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        **kwargs
    )


def _init_anthropic(model: str, temperature: float, api_key: str, **kwargs):
    """Initialize Anthropic Claude LLM."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "Anthropic not available. Install with: pip install langchain-anthropic"
        )
    
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=api_key,
        **kwargs
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_provider_available(provider: str) -> tuple[bool, str]:
    """
    Check if a provider is available (package installed and API key set).
    
    Args:
        provider: Provider name
    
    Returns:
        Tuple of (is_available, message)
    
    Example:
        >>> available, msg = check_provider_available("gemini")
        >>> if not available:
        ...     print(msg)
    """
    provider = provider.lower()
    
    try:
        provider_enum = LLMProvider(provider)
    except ValueError:
        return False, f"Invalid provider: {provider}"
    
    # Check if package is installed
    package_checks = {
        LLMProvider.OPENAI: "langchain_openai",
        LLMProvider.GEMINI: "langchain_google_genai",
        LLMProvider.ANTHROPIC: "langchain_anthropic"
    }
    
    try:
        __import__(package_checks[provider_enum])
    except ImportError:
        return False, f"Package not installed for {provider}"
    
    # Check if API key is set
    env_var = ENV_VAR_NAMES[provider_enum]
    if not os.getenv(env_var):
        return False, f"{env_var} not set"
    
    return True, f"{provider} is available"


def get_available_providers() -> list[str]:
    """
    Get list of available providers (installed and configured).
    
    Returns:
        List of provider names that are ready to use
    
    Example:
        >>> providers = get_available_providers()
        >>> print(f"Available: {', '.join(providers)}")
    """
    available = []
    for provider in LLMProvider:
        is_available, _ = check_provider_available(provider.value)
        if is_available:
            available.append(provider.value)
    return available


def get_default_provider() -> Optional[str]:
    """
    Get the first available provider as default.
    
    Returns:
        Provider name or None if no providers are available
    
    Example:
        >>> provider = get_default_provider()
        >>> if provider:
        ...     llm = get_llm(provider=provider)
    """
    available = get_available_providers()
    return available[0] if available else None


def get_provider_info(provider: str) -> Dict[str, Any]:
    """
    Get information about a provider.
    
    Args:
        provider: Provider name
    
    Returns:
        Dictionary with provider information
    
    Example:
        >>> info = get_provider_info("gemini")
        >>> print(f"Default model: {info['default_model']}")
    """
    provider = provider.lower()
    
    try:
        provider_enum = LLMProvider(provider)
    except ValueError:
        return {"error": f"Invalid provider: {provider}"}
    
    is_available, availability_msg = check_provider_available(provider)
    
    return {
        "provider": provider,
        "available": is_available,
        "status": availability_msg,
        "default_model": DEFAULT_MODELS[provider_enum],
        "env_var": ENV_VAR_NAMES[provider_enum],
        "env_var_set": bool(os.getenv(ENV_VAR_NAMES[provider_enum]))
    }


# ============================================================================
# CLI HELPER
# ============================================================================

def print_provider_status():
    """Print status of all providers (for debugging)."""
    print("\n" + "="*70)
    print("LLM PROVIDER STATUS")
    print("="*70)
    
    for provider in LLMProvider:
        info = get_provider_info(provider.value)
        status_icon = "✅" if info["available"] else "❌"
        print(f"\n{status_icon} {provider.value.upper()}")
        print(f"  Default Model: {info['default_model']}")
        print(f"  Env Variable:  {info['env_var']}")
        print(f"  Status:        {info['status']}")
    
    print("\n" + "="*70)
    available = get_available_providers()
    if available:
        print(f"✓ Available providers: {', '.join(available)}")
    else:
        print("✗ No providers available. Set API keys to use LLMs.")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Print provider status when run directly
    print_provider_status()


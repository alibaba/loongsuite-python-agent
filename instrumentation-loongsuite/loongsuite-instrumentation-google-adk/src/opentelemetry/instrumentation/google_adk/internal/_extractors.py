"""
ADK Attribute Extractors following OpenTelemetry GenAI Semantic Conventions.

This module extracts trace attributes from Google ADK objects according
to OpenTelemetry GenAI semantic conventions (latest version).
"""


class AdkAttributeExtractors:
    """
    Attribute extractors for Google ADK following OpenTelemetry GenAI semantic conventions.

    Extracts trace attributes from ADK objects according to:
    - gen_ai.* attributes for GenAI-specific information
    - Standard OpenTelemetry attributes for general information
    """

    def _extract_provider_name(self, model_name: str) -> str:
        """
        Extract provider name from model name according to OTel GenAI conventions.

        Args:
            model_name: Model name string

        Returns:
            Provider name following OTel GenAI standard values
        """
        if not model_name:
            return "google_adk"

        model_lower = model_name.lower()

        # Google models - use standard values from OTel spec
        if "gemini" in model_lower:
            return "gcp.gemini"  # AI Studio API
        elif "vertex" in model_lower:
            return "gcp.vertex_ai"  # Vertex AI
        # OpenAI models
        elif "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        # Anthropic models
        elif "claude" in model_lower:
            return "anthropic"
        # Other providers
        elif "llama" in model_lower or "meta" in model_lower:
            return "meta"
        elif "mistral" in model_lower:
            return "mistral_ai"
        elif "cohere" in model_lower:
            return "cohere"
        else:
            # Default to google_adk for unknown models
            return "google_adk"

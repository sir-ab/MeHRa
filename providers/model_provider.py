from typing import List, Dict

class ModelProvider:
    """Base class for model providers."""

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the model."""
        raise NotImplementedError("Subclasses must implement this method")

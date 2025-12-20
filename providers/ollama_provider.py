import aiohttp
import requests
import json
from typing import Dict, List, Any, AsyncGenerator
from .model_provider import ModelProvider

class OllamaProvider(ModelProvider):
    """Ollama model provider."""
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        """Initialize the Ollama provider.
        
        Args:
            model_name: Name of the model to use
            base_url: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/chat"
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a complete response using Ollama.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters to pass to the Ollama API
                temperature: Controls randomness (0.0 to 1.0)
                top_p: Controls diversity (0.0 to 1.0)
                max_tokens: Maximum number of tokens to generate
        
        Returns:
            Complete text response
        """
        # For non-streaming, collect all chunks and return as single string
        full_response = ""
        for chunk in self.generate_response_stream(messages, **kwargs):
            full_response += chunk
        return full_response
    
    async def generate_response_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Ollama asynchronously.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            **kwargs: Additional parameters to pass to the Ollama API.

        Yields:
            Text chunks as they are generated.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            **kwargs
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_endpoint, json=payload) as response:
                    response.raise_for_status()

                    # Process the streaming response asynchronously
                    async for line in response.content:
                        line_text = line.decode('utf-8').strip()

                        if not line_text:
                            continue

                        try:
                            chunk_data = json.loads(line_text)

                            # Extract and yield content
                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                yield chunk_data["message"]["content"]
                            elif "done" in chunk_data and chunk_data["done"]:
                                break  # End of stream
                        except json.JSONDecodeError:
                            continue  # Skip invalid JSON lines

        except aiohttp.ClientError as e:
            print(f"Error calling Ollama API: {e}")
            yield f"Error: Failed to get response from model. {str(e)}"
    
    def list_available_models(self) -> List[str]:
        """List available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            result = response.json()
            return [model["name"] for model in result["models"]]
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []
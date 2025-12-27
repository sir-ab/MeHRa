from typing import Dict, List, Any, Generator
from .model_provider import ModelProvider

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama-cpp-python is not installed. "
        "Install it with: pip install llama-cpp-python"
    )


class LlamaCppProvider(ModelProvider):
    """Llama.cpp model provider for running GGUF format models locally."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        **kwargs
    ):
        """Initialize the Llama.cpp provider.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (default: 2048)
            n_threads: Number of threads for inference (default: 8)
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only, default: 0)
            verbose: Enable verbose output (default: False)
            **kwargs: Additional parameters to pass to Llama
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a complete response using Llama.cpp.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters
                temperature: Controls randomness (0.0 to 1.0, default: 0.7)
                top_p: Controls diversity (0.0 to 1.0, default: 0.95)
                max_tokens: Maximum number of tokens to generate (default: 512)
        
        Returns:
            Complete text response
        """
        # Collect all chunks from streaming and return as single string
        full_response = ""
        for chunk in self.generate_response_stream(messages, **kwargs):
            full_response += chunk
        return full_response
    
    def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate a streaming response using Llama.cpp.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters
                temperature: Controls randomness (0.0 to 1.0, default: 0.7)
                top_p: Controls diversity (0.0 to 1.0, default: 0.95)
                max_tokens: Maximum number of tokens to generate (default: 512)
        
        Yields:
            Text chunks as they are generated
        """
        # Extract generation parameters
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.95)
        max_tokens = kwargs.get("max_tokens", 512)
        
        # Format messages into a prompt
        prompt = self._format_messages_to_prompt(messages)
        
        try:
            # Use llama_cpp's streaming capability
            response = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )
            
            # Yield text chunks from the stream
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("text", "")
                    if delta:
                        yield delta
                        
        except Exception as e:
            yield f"Error: Failed to generate response. {str(e)}"
    
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a prompt string.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
        
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for message in messages:
            role = message.get("role", "user").lower()
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"{content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        # Add prompt for assistant response
        prompt += "Assistant:"
        return prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        try:
            return {
                "model_path": self.model_path,
                "context_size": self.n_ctx,
                "threads": self.n_threads,
                "gpu_layers": self.n_gpu_layers,
                "model_loaded": self.model is not None,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def __del__(self):
        """Clean up model resources."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model = None
            except Exception:
                pass

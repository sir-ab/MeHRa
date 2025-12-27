from .tts_interface import TTSEngineInterface
from .pyttsx3_engine import PyTTSX3Engine
from .kokoro.interface import KokoroEngineInterface
from .kokoro_engine import KokoroEngine

# Factory function to get the appropriate TTS engine
def get_tts_engine(engine_type="pyttsx3", **kwargs):
    """
    Factory function to create and return the specified TTS engine.
    
    Args:
        engine_type (str): The type of TTS engine to create ("pyttsx3" or "kokoro")
        **kwargs: Additional arguments to pass to the TTS engine constructor
    
    Returns:
        TTSEngineInterface: An initialized TTS engine
    """
    if engine_type.lower() == "pyttsx3":
        return PyTTSX3Engine(voice="female", rate=140, **kwargs).initialize()
    
    elif engine_type.lower() == "kokoro":
        return KokoroEngine(**kwargs).initialize()
    else:
        raise ValueError(f"Unknown TTS engine type: {engine_type}")

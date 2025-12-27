from .whisper_engine import WhisperEngine

# Factory function to get the appropriate STT engine
def get_stt_engine(engine_type="whisper", **kwargs):
    """
    Factory function to create and return the specified STT engine.
    
    Args:
        engine_type (str): The type of STT engine to create ("whisper")
        **kwargs: Additional arguments to pass to the STT engine constructor
    
    Returns:
        STTEngineInterface: An initialized STT engine
    """
    
    if engine_type.lower() == "whisper":
        return WhisperEngine(**kwargs).initialize()
    else:
        raise ValueError(f"Unknown STT engine type: {engine_type}")
import abc

class TTSEngineInterface(abc.ABC):
    """Abstract base class interface for TTS engines."""
    
    @abc.abstractmethod
    def initialize(self):
        """Initialize the TTS engine."""
        pass
    
    @abc.abstractmethod
    def say(self, text):
        """Convert text to speech and play it."""
        pass
    
    @abc.abstractmethod
    def update_subtitle(self, text):
        """Update subtitle file with current text."""
        pass
    
    @abc.abstractmethod
    def shutdown(self):
        """Clean up resources used by the TTS engine."""
        pass
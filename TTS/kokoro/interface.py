from .engine import KokoroEngine
from . import text_processor, audio_generator, audio_player, config
from ..tts_interface import TTSEngineInterface
from kokoro import KPipeline

class KokoroEngineInterface(TTSEngineInterface):
    def __init__(self, **kwargs):
        self.engine = KokoroEngine(**kwargs)

    def initialize(self):
        return self.engine.initialize()
    
    def say(self, text):
        text_processor.text_queue.put(text)
    
    def update_subtitle(self, text):
        audio_player._update_subtitle(self.engine, text)
    
    def shutdown(self):
        self.engine.shutdown()
    
    def set_voice(self, voice):
        """Change the voice used for synthesis."""
        self.engine.config.voice = voice
        print(f"Voice changed to: {voice}")
    
    def set_language(self, lang_code):
        """Change the language used for synthesis."""
        # This requires reinitializing the pipeline
        self.engine.config.lang_code = lang_code
        try:
            # Create a new pipeline with the updated language
            self.engine.pipeline = KPipeline(lang_code=lang_code)
            print(f"Language changed to: {lang_code}")
        except Exception as e:
            print(f"Error changing language: {e}")
    
    def set_rate(self, rate):
        """Change the speech rate."""
        self.engine.config.rate = float(rate)
        print(f"Speech rate changed to: {self.engine.config.rate}")
        
    def set_batch_delay(self, delay):
        """Set the delay time for batching queue items (in seconds)."""
        self.engine.config.batch_delay = float(delay)
        print(f"Batch delay set to: {self.engine.config.batch_delay} seconds")

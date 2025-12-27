import threading
import queue
import torch
import time
import soundfile as sf
import sounddevice as sd
import concurrent.futures
import os

from . import text_processor, audio_generator, audio_player, config
from kokoro import KPipeline, KModel

class KokoroEngine:
    """TTS engine implementation using Kokoro TTS with parallel processing."""
    
    def __init__(self, **kwargs):
        self.config = config.KokoroConfig(**kwargs)
        self.stop_event = threading.Event()
        self.pipeline = None
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.processor_thread = None
        self.player_thread = None
        self.audio_output_dir = "audio_output"
        
        # Ensure audio output directory exists
        os.makedirs(self.audio_output_dir, exist_ok=True)
    
    def initialize(self):
        """Initialize the Kokoro TTS engine and start worker threads."""
        try:
            # Initialize Kokoro pipeline with the specified language code
            self.pipeline = KPipeline(lang_code=self.config.lang_code, device='cuda')
            print(f"Kokoro TTS initialized with language code: {self.config.lang_code}")
            
            # Start the processor thread (collects text and generates audio)
            self.processor_thread = threading.Thread(
                target=text_processor.process_text,
                args=(self,),
                daemon=True
            )
            self.processor_thread.start()
            
            # Start the player thread (plays generated audio)
            self.player_thread = threading.Thread(
                target=audio_player.play_audio,
                args=(self,),
                daemon=True
            )
            self.player_thread.start()
            
            return self
        except Exception as e:
            print(f"Error initializing Kokoro TTS: {e}")
            raise
    
    def shutdown(self):
        """Stop all worker threads and clean up resources."""
        self.stop_event.set()
        
        # Send termination signals to both queues
        text_processor.text_queue.put(None)
        audio_generator.audio_queue.put((None, None, None))
        
        # Wait for threads to terminate
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2)
        
        if self.player_thread and self.player_thread.is_alive():
            self.player_thread.join(timeout=2)
            
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)
        
        print("Kokoro TTS engine shut down")

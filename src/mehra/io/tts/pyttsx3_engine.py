"""
This module implements the TTS engine using pyttsx3.
"""
import os
import pyttsx3
import threading
import queue
from .tts_interface import TTSEngineInterface

class PyTTSX3Engine(TTSEngineInterface):
    """TTS engine implementation using pyttsx3."""
    
    def __init__(self, voice="female", rate=140, subtitle_path="subtitles.txt"):
        self.voice_type = voice
        self.rate = rate
        self.subtitle_path = subtitle_path
        self.engine = None
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
    
    def initialize(self):
        """Initialize the TTS engine and start the worker thread."""
        # Start the worker thread
        self.worker_thread = threading.Thread(
            target=self._tts_worker,
            daemon=True
        )
        self.worker_thread.start()
        return self
    
    def say(self, text):
        """Add text to the TTS queue."""
        self.queue.put(text)
    
    def update_subtitle(self, text):
        """Update subtitle file with current text."""
        try:
            with open(self.subtitle_path, 'w', encoding="utf-8") as file:
                file.write(text)
        except Exception as e:
            print(f"Error updating subtitle file: {e}")
    
    def shutdown(self):
        """Stop the worker thread and clean up resources."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.worker_thread.join(timeout=2)
    
    def _tts_worker(self):
        """Worker thread that processes the TTS queue."""
        # Initialize engine in the worker thread
        self.engine = pyttsx3.init()
        
        # Configure engine
        voices = self.engine.getProperty('voices')
        if self.voice_type == "female" and len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
        else:
            self.engine.setProperty('voice', voices[0].id)
        
        self.engine.setProperty('rate', self.rate)
        
        # Process queue until stopped
        while not self.stop_event.is_set():
            try:
                # Non-blocking get with timeout
                text = self.queue.get(timeout=0.5)
                # print(f"TTS worker processing: {text}")
                
                # Update subtitle file
                self.update_subtitle(str(text))
                
                # Perform TTS
                self.engine.say(text)
                self.engine.runAndWait()
                
                # print(f"TTS worker finished playing: {text}")
                self.queue.task_done()
            except queue.Empty:
                # No items in queue, just continue
                continue
            except Exception as e:
                print(f"Error in TTS worker: {e}")
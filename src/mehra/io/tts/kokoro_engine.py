import os
import threading
import queue
import torch
import time
import soundfile as sf
import sounddevice as sd
import concurrent.futures
import logging
from kokoro import KPipeline, KModel
from .tts_interface import TTSEngineInterface

import warnings

# Suppress Hugging Face repo_id warning
warnings.filterwarnings("ignore", message="Defaulting repo_id")

# Suppress PyTorch RNN dropout warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

# Suppress PyTorch weight normalization deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

logging.basicConfig(level=logging.INFO)

class KokoroEngine(TTSEngineInterface):
    """TTS engine implementation using Kokoro TTS with parallel processing."""
    
    def __init__(self, voice="bf_isabella", lang_code='a', rate=1.0, subtitle_path="subtitles.txt", batch_delay=0.1):
        self.voice = voice
        self.lang_code = lang_code
        self.rate = rate
        self.subtitle_path = subtitle_path
        self.text_queue = queue.Queue()  # Queue for incoming text
        self.audio_queue = queue.Queue()  # Queue for processed audio segments
        self.spoken_text = queue.Queue()  
        self.stop_event = threading.Event()
        self.pipeline = None
        self.audio_output_dir = "audio_output"
        self.batch_delay = batch_delay
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.processor_thread = None
        self.player_thread = None
        
        #checking if talking
        self.is_talking = False # tracking if tts is talking to know between iteration of chatting (for resetting batch delay)
        self.is_generating = False # NO IDEA WHAT IT DOES NOW
        
        # Ordered storage for audio segments
        self.ordered_audio_segments = {}
        self.next_segment_to_play = 1
        self.audio_ordering_lock = threading.Lock()

        self.interrupt_event = None
        
        # Ensure audio output directory exists
        os.makedirs(self.audio_output_dir, exist_ok=True)
    
    def initialize(self):
        """Initialize the Kokoro TTS engine and start worker threads."""
        try:
            # Initialize Kokoro pipeline with the specified language code
            self.pipeline = KPipeline(lang_code=self.lang_code, repo_id='hexgrad/Kokoro-82M', device='cuda')
            logging.debug(f"Kokoro TTS initialized with language code: {self.lang_code}")
            
            # Start the processor thread (collects text and generates audio)
            self.processor_thread = threading.Thread(
                target=self._processor_worker,
                daemon=True
            )
            self.processor_thread.start()
            
            # Start the player thread (plays generated audio)
            self.player_thread = threading.Thread(
                target=self._player_worker,
                daemon=True
            )
            self.player_thread.start()
            
            return self
        except Exception as e:
            logging.error(f"Error initializing Kokoro TTS: {e}")
            raise
    
    def say(self, text):
        """Add text to the TTS queue."""
        self.text_queue.put(text)
    
    def update_subtitle(self, text):
        """Update subtitle file with current text."""
        try:
            with open(self.subtitle_path, 'w', encoding="utf-8") as file:
                file.write(text)
        except Exception as e:
            logging.error(f"Error updating subtitle file: {e}")
    
    def interrupt(self):
        """Interrupt TTS, stop worker threads, reset states, and restart everything."""
        logging.info("Interrupting TTS...")

        # Signal threads to stop
        self.stop_event.set()
            
        # Wait for worker threads to finish
        if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=0.5)  # Give threads 2 seconds to stop
        if hasattr(self, 'player_thread') and self.player_thread.is_alive():
            self.player_thread.join(timeout=0.5)

        self.is_talking = False         

        # Reset all queues and states
        self.text_queue = queue.Queue()  # Fresh queue for text input
        self.audio_queue = queue.Queue()  # Fresh queue for audio output

        self.ordered_audio_segments.clear()  # Clear any stored audio segments
        self.next_segment_to_play = 1  # Reset playback order
        self.sentence_index = 0  # Reset sentence tracking
        self.batch_delay = 0.1  # Reset any timing delays

        self.stop_event.clear()

        # Restart worker threads with new queues and states
        self.processor_thread = threading.Thread(target=self._processor_worker, daemon=True)
        self.processor_thread.start()
        self.player_thread = threading.Thread(target=self._player_worker, daemon=True)
        self.player_thread.start()

        # Allow new tasks to proceed
        logging.info("TTS interruption complete.")

    def shutdown(self):
        """Stop all worker threads and clean up resources."""
        self.stop_event.set()
        
        # Send termination signals to both queues
        self.text_queue.put(None)
        self.audio_queue.put((None, None, None))
        
        # Wait for threads to terminate
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2)
        
        if self.player_thread and self.player_thread.is_alive():
            self.player_thread.join(timeout=2)
            
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)
        
        logging.debug("Kokoro TTS engine shut down")

    def _processor_worker(self):
        """Worker thread that processes text queue and generates audio segments."""
        logging.debug("Processor worker thread started")
        
        self.sentence_index = 0

        while not self.stop_event.is_set():
            try:
                # Batch multiple queue items into a single text
                # print(self.is_talking)
                combined_text = self._batch_queue_items()
                
                # If no text was collected or exit signal received
                if combined_text is None:
                    continue
                    
                logging.debug(f"Processing batch: '{combined_text[:50]}...' ({len(combined_text)} chars)")

                sentences_number = len(combined_text.split(". "))
                self.sentence_index += 1  # counting each batch not each sentence
                print(sentences_number)
                # Process text in the thread pool to allow for parallel processing
                future = self.thread_pool.submit(self._generate_audio_for_text, combined_text, self.sentence_index)
                
                self.batch_delay *= 3

                # Mark all items in this batch as done
                for _ in range(sentences_number-1):
                    self.text_queue.task_done()
                    
            except Exception as e:
                logging.error(f"Error in processor worker: {e}")
                # Try to mark the task as done even if there was an error
                try:
                    self.text_queue.task_done()
                except:
                    pass
    
    def _batch_queue_items(self):
        """Batch multiple queue items into a single text with a short delay."""
        combined_text = []
        
        # Get the first item
        try:
            item = self.text_queue.get()
            if item is None:  # Exit signal
                return None
            combined_text.append(item)
        except queue.Empty:
            return None
            
        # Wait for a short time to see if more items arrive
        batch_end_time = time.time() + self.batch_delay

        # if self.is_talking:
        #     # Keep collecting items until the batch delay expires or queue is empty
        #     while time.time() < batch_end_time:
        #         try:
        #             item = self.text_queue.get_nowait()
        #             if item is None:  # Exit signal
        #                 # Mark all collected items as done and return None
        #                 for _ in range(len(combined_text)):
        #                     self.text_queue.task_done()
        #                 return None
        #             combined_text.append(item)
        #         except queue.Empty:
        #             continue
        # #if tts not talking, then no need to get next item, also reset batch delay to prevent exploding time
        if not self.is_talking:
            self.batch_delay = 0.1
                
        # Combine all collected text items with a period between them
        return " ".join(combined_text) if combined_text else None
    
    def _generate_audio_for_text(self, text, sentence_index):
        """Generate audio segments for text and queue them for playback."""
        try:
            # Generate speech using Kokoro
            generator = self.pipeline(
                text, 
                voice=self.voice,
                speed=self.rate, 
                # split_pattern=r'\n+'
            )
            
            self.is_talking = True

            # Submit each segment for processing
            segment_count = 0
            for i, (gs, ps, audio) in enumerate(generator):
                if self.stop_event.is_set():
                    break
                
                # Optional: Save the audio to file
                output_file = os.path.join(self.audio_output_dir, f"segment_{time.time()}_{i}.wav")
                sf.write(output_file, audio, 24000)
                
                # Queue the audio segment for playback
                self.audio_queue.put((audio, gs, sentence_index))
                segment_count += 1
                
            return segment_count, text
            
        except Exception as e:
            logging.error(f"Error generating audio: {e}")
            return 0, text

    def _player_worker(self):
        """Worker thread that plays audio segments from the audio queue."""
        logging.debug("Player worker thread started")
        
        # Track the next segment that should be played
        self.next_segment_to_play = 1

        while not self.stop_event.is_set():
            try:
                # Get the next audio segment to play
                audio, text, sentence_index = self.audio_queue.get()
                
                # Check for termination signal
                if audio is None:
                    self.audio_queue.task_done()
                    break
                
                with self.audio_ordering_lock:
                    # Store this segment in our ordered dictionary
                    self.ordered_audio_segments[sentence_index] = (audio, text)
                    
                    # Process as many segments as we can in order
                    self._process_ordered_segments()

                # Mark this audio segment as done
                self.audio_queue.task_done()

                if self.audio_queue.empty():
                    self.is_talking = False

                
            except queue.Empty:
                # No audio to play, check if we have segments ready that were previously queued
                with self.audio_ordering_lock:
                    self._process_ordered_segments()
                continue
            except Exception as e:
                logging.error(f"Error in player worker: {e}")
                # Try to mark the task as done even if there was an error
                # try:
                #     self.audio_queue.task_done()
                # except:
                pass

    def _process_ordered_segments(self):
        """Process segments in order based on sentence_index."""
        # This method should be called with audio_ordering_lock held
        while self.next_segment_to_play in self.ordered_audio_segments:
            # Get the next segment to play
            audio, text = self.ordered_audio_segments.pop(self.next_segment_to_play)
            
            display_text = text[:30] + "..." if len(text) > 30 else text
            logging.debug(f"Playing segment {self.next_segment_to_play}: '{display_text}'")
            
            # Play the audio
            self.play_audio(audio, text)
            
            # Move to the next segment
            self.next_segment_to_play += 1

    def play_audio(self, audio_data, text, sample_rate=24000):
        """Play audio data and update subtitle."""
        if not self.interrupt_event.is_set():
            try:
                # Cut the beginning of the audio by 0.2 seconds
                # Calculate how many samples to cut (0.2 seconds * sample_rate)
                start_to_cut = int(0.24 * sample_rate)
                end_to_cut = int(0.14 * sample_rate)
                
                # Make sure we don't try to cut more than we have
                if len(audio_data) > start_to_cut + end_to_cut:
                    trimmed_audio = audio_data[start_to_cut:-end_to_cut]
                else:
                    trimmed_audio = audio_data
                
                # Play the trimmed audio synchronously
                sd.play(trimmed_audio, sample_rate*1.15, device="Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO), Windows DirectSound")
                sd.wait()  # Wait until playback is finished
                self.spoken_text.put(text)

            except Exception as e:
                logging.error(f"Error playing audio: {e}")
            finally:
                # Reset subtitle after this segment is done
                self.update_subtitle("")
        else:
            self.update_subtitle("")

    def set_voice(self, voice):
        """Change the voice used for synthesis."""
        self.voice = voice
        logging.info(f"Voice changed to: {voice}")
    
    def set_language(self, lang_code):
        """Change the language used for synthesis."""
        # This requires reinitializing the pipeline
        self.lang_code = lang_code
        try:
            # Create a new pipeline with the updated language
            self.pipeline = KPipeline(lang_code=lang_code)
            logging.info(f"Language changed to: {lang_code}")
        except Exception as e:
            logging.error(f"Error changing language: {e}")
    
    def set_rate(self, rate):
        """Change the speech rate."""
        self.rate = float(rate)
        logging.info(f"Speech rate changed to: {self.rate}")
        
    def set_batch_delay(self, delay):
        """Set the delay time for batching queue items (in seconds)."""
        self.batch_delay = float(delay)
        logging.info(f"Batch delay set to: {self.batch_delay} seconds")
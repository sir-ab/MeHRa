import pyaudio
import numpy as np
import threading
import queue
import time
from faster_whisper import WhisperModel
from collections import deque
import torch
import asyncio

class WhisperEngine:
    def __init__(self, model_path="Systran/faster-whisper-small", vad_threshold=0.5):
        # Model configuration
        self.model_path = model_path
        self.model = None
        
        # Audio settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        
        # Processing queues and buffers
        self.audio_queue = queue.Queue()
        self.smart_audio = []
        self.audio_buffer = deque(maxlen=80)  # Store ~2.5 seconds of audio
        
        # PyAudio elements
        self.audio_stream = None
        self.p = None
        
        # Thread control
        self.recording = False
        self.transcription_thread = None
        self.recording_thread = None
        self.vad_thread = None
        
        # VAD settings
        self.vad_threshold = vad_threshold
        self.vad_model = None
        self.silence_counter = 0
        self.speaking = False
        self.interrupt_event = asyncio.Event()
        self.current_audio_chunk = []
        
        # Performance tracking
        self.processing_times = deque(maxlen=10)
        self.transcript_queue = queue.Queue()

    def initialize(self):
        """Initialize the WhisperEngine with models and audio interface"""
        print("Initializing Whisper model...")
        # Detect if CUDA is available
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device = "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            # compute_type = "int8"
        except ImportError:
            device = "cpu"
            compute_type = "int8"
            
        self.model = WhisperModel(
            self.model_path, 
            device=device, 
            compute_type=compute_type
        )
        
        # Initialize VAD model
        print("Initializing VAD model...")
        import torch
        self.vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)
        self.vad_model.eval()
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.audio_stream = self.p.open(
            format=self.format, 
            channels=self.channels, 
            rate=self.rate,
            input=True, 
            frames_per_buffer=self.chunk
        )

        self.start_stream() 

        return self

    
    
    def start_stream(self):
        """Start all processing threads"""
        if not self.model:
            print("Model not initialized. Call initialize() first.")
            return False
            
        # Start transcription thread
        if self.transcription_thread is None or not self.transcription_thread.is_alive():
            self.transcription_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.transcription_thread.start()
        
        # Start VAD thread
        if self.vad_thread is None or not self.vad_thread.is_alive():
            self.vad_thread = threading.Thread(target=self._vad_processing, daemon=True)
            self.vad_thread.start()
            
        # Start recording thread
        if self.recording_thread is None or not self.recording_thread.is_alive():
            self.recording_thread = threading.Thread(target=self._start_recording, daemon=True)
            self.recording_thread.start()
            return True
        else:
            print("Transcription stream already running.")
            return False
    
    def _start_recording(self):
        """Record audio from microphone and add to processing queue"""
        self.recording = True
        print("Recording started")
        
        while self.recording:
            try:
                data = self.audio_stream.read(self.chunk, exception_on_overflow=False)
                audio_array = self._bytes_to_float_array(data)
                self.audio_buffer.append(audio_array)
                # print(len(self.audio_buffer))
            except Exception as e:
                print(f"Error in recording: {e}")
                time.sleep(0.1)  # Prevent tight loop on error

    def _vad_processing(self):
        """Process audio with Voice Activity Detection"""
        print("VAD processing started")

        self.recording = True
        window_size = 512  # Set window size for VAD
        num_chunk = 2
        percent_of_speech = (self.chunk * num_chunk // window_size) // 20 #0.2 litrally mean if >= 0
        self.speech_cut_off = (self.rate / (self.chunk * num_chunk)) * 60 // 1  # 20 seconds  
        # print(self.speech_cut_off // 1)
        while self.recording:
            if len(self.audio_buffer) < num_chunk:  # Wait for any audio data
                time.sleep(0.05)
                continue
                
            # Get the latest audio data
            audio_data = np.concatenate(list(self.audio_buffer))
            self.audio_buffer.clear()

            # Process audio in appropriate chunks for VAD
            speech_detected = False
            speech_counter = 0

            for i in range(0, len(audio_data), window_size):
                chunk = audio_data[i:i+window_size]
                if len(chunk) > 0:  # Only process complete chunks
                    speech_prob = self._get_speech_probability(chunk)
                    if speech_prob > self.vad_threshold:
                        speech_counter += 1
                        if speech_counter > percent_of_speech:
                            # print("Speech detected")
                            # print(len(self.current_audio_chunk))
                            speech_detected = True
                            break
            
            if speech_detected:
                self.interrupt_event.set()
                # Voice detected
                if not self.speaking:
                    print("Speech detected, starting transcription")
                    self.speaking = True
                    self.current_audio_chunk = []
                
                # Add audio to current chunk
                self.current_audio_chunk.append(audio_data)
                self.silence_counter = 0

                if len(self.current_audio_chunk) >= self.speech_cut_off:
                    full_audio = np.concatenate(self.current_audio_chunk)
                    self.audio_queue.put(full_audio)
                    # print(f"Speech segment queued for processing ({len(full_audio)/self.rate:.1f}s)")
                    self.current_audio_chunk = []

            else:
                # No voice detected
                if self.speaking:
                    self.silence_counter += 1
                    
                    # If silence for more than ~0.128*3 second, process the chunk
                    if self.silence_counter >= 3:
                        self.interrupt_event.clear()
                        if self.current_audio_chunk:
                            full_audio = np.concatenate(self.current_audio_chunk)
                            self.audio_queue.put(full_audio)
                            # print(f"Speech segment queued for processing ({len(full_audio)/self.rate:.1f}s)")
                        
                        self.speaking = False
                        self.current_audio_chunk = []
                        self.silence_counter = 0
            
            # Clear buffer to prevent reprocessing the same audio
            # time.sleep(0.1)  # Prevent tight CPU usage
    

    def _get_speech_probability(self, audio_data):
        """Calculate probability of speech in audio segment"""
        if self.vad_model is None:
            return 0.5  # Default if VAD not initialized
        
        try:
            # Silero VAD expects chunks of 512 samples
            # Make sure audio_data is a numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
                
            # Take only the first 512 samples or pad if needed
            if len(audio_data) >= 512:
                audio_sample = audio_data[:512]
            else:
                audio_sample = np.pad(audio_data, (0, 512 - len(audio_data)))
            
            # Convert to tensor correctly - use unsqueeze to add batch dimension
            with torch.no_grad():
                vad_tensor = torch.from_numpy(audio_sample).unsqueeze(0)
                speech_prob = self.vad_model(vad_tensor, self.rate).item()
            return speech_prob
        except Exception as e:
            print(f"VAD error: {e}")
            return 0.0
        
    def _process_audio(self):
        """Process audio segments with Whisper for transcription"""
        print("Transcription processing thread started")
        
        while True:
            try:
                # Get audio from queue (blocks until available)
                audio_data = self.audio_queue.get()

                # None is the signal to stop
                if audio_data is None:
                    break
                
                # self.smart_audio.append(audio_data)

                # if len(self.smart_audio) >= 2 or not self.speaking:
                # audio_data = np.concatenate(self.smart_audio)
                # self.smart_audio = []

                # Measure processing time
                start_time = time.time()
                
                # Process with Whisper
                segments, info = self.model.transcribe(
                    audio_data, 
                    language="en",
                    beam_size=3,
                    # vad_filter=True,  # Use Whisper's built-in VAD as additional filter
                    # vad_parameters={"threshold": 0.5}
                )
                
                # Get transcription text
                transcript = ""
                for segment in segments:
                    transcript += segment.text + " "
                
                # Calculate and store processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                avg_time = sum(self.processing_times) / len(self.processing_times)
                
                # Only print non-empty transcripts
                if transcript.strip():
                    print(f"\nTranscript: {transcript.strip()}")
                    self.transcript_queue.put(transcript.strip())

                    # print(f"Processed {len(audio_data)/self.rate:.1f}s audio in {processing_time:.2f}s (avg: {avg_time:.2f}s)")
            
                # Signal that processing is complete
                self.audio_queue.task_done()
                
            except Exception as e:
                print(f"Error in transcription: {e}")


    def stop_stream(self):
        """Stop all processing threads safely"""
        print("Stopping transcription stream...")
        self.recording = False  # Stop recording
        
        # Safely stop recording thread
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
            
        # Signal transcription thread to stop and wait
        self.audio_queue.put(None)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2.0)
            
        # Clean up stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
            
        print("Transcription stream stopped")

    def _bytes_to_float_array(self, audio_bytes):
        """Convert raw audio bytes to normalized float array"""
        try:
            raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
            return raw_data.astype(np.float32) / 32768.0
        except Exception as e:
            print(f"Error converting audio: {e}")
            return np.zeros(1, dtype=np.float32)

    def shutdown(self):
        """Completely shut down the engine"""
        print("Shutting down WhisperEngine...")
        self.stop_stream()
        if self.p:
            self.p.terminate()
            self.p = None
        print("WhisperEngine shutdown complete")

    def adjust_vad_threshold(self, new_threshold):
        """Dynamically adjust VAD sensitivity"""
        if 0.0 <= new_threshold <= 1.0:
            self.vad_threshold = new_threshold
            print(f"VAD threshold adjusted to {new_threshold}")
            return True
        else:
            print("VAD threshold must be between 0.0 and 1.0")
            return False


# Example usage
if __name__ == "__main__":
    engine = WhisperEngine()
    
    try:
        engine.initialize()
        engine.start_stream()
        
        print("Listening for speech... (Press Ctrl+C to stop)")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        engine.shutdown()

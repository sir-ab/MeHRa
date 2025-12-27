import sounddevice as sd
import threading
from . import audio_generator
import queue

ordered_audio_segments = {}
audio_ordering_lock = threading.Lock()
next_segment_to_play = 1

def play_audio(engine):
    """Worker thread that plays audio segments from the audio queue."""
    print("Player worker thread started")
    
    # Track the next segment that should be played
    global next_segment_to_play
    next_segment_to_play = 1

    while not engine.stop_event.is_set():
        try:
            # Get the next audio segment to play
            audio, text, sentence_index = audio_generator.audio_queue.get()
            
            # Check for termination signal
            if audio is None:
                audio_generator.audio_queue.task_done()
                break
            
            with audio_ordering_lock:
                # Store this segment in our ordered dictionary
                ordered_audio_segments[sentence_index] = (audio, text)
                
                # Process as many segments as we can in order
                _process_ordered_segments(engine)
            
            # Mark this audio segment as done
            audio_generator.audio_queue.task_done()
            
        except queue.Empty:
            # No audio to play, check if we have segments ready that were previously queued
            with audio_ordering_lock:
                _process_ordered_segments(engine)
            continue
        except Exception as e:
            print(f"Error in player worker: {e}")
            # Try to mark the task as done even if there was an error
            try:
                audio_generator.audio_queue.task_done()
            except:
                pass

def _process_ordered_segments(engine):
    """Process segments in order based on sentence_index."""
    # This method should be called with audio_ordering_lock held
    global next_segment_to_play
    while next_segment_to_play in ordered_audio_segments:
        # Get the next segment to play
        audio, text = ordered_audio_segments.pop(next_segment_to_play)
        
        display_text = text[:30] + "..." if len(text) > 30 else text
        print(f"Playing segment {next_segment_to_play}: '{display_text}'")
        
        # Play the audio
        _play_audio(engine, audio, text)
        
        # Move to the next segment
        next_segment_to_play += 1

def _play_audio(engine, audio_data, text, sample_rate=24000):
    """Play audio data and update subtitle."""
    try:
        # Update subtitle with current segment being spoken
        _update_subtitle(engine, text)
        
        # Play the audio synchronously
        sd.play(audio_data, sample_rate)
        sd.wait()  # Wait until playback is finished
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        # Reset subtitle after this segment is done
        _update_subtitle(engine, "")

def _update_subtitle(engine, text):
    """Update subtitle file with current text."""
    try:
        with open(engine.config.subtitle_path, 'w', encoding="utf-8") as file:
            file.write(text)
    except Exception as e:
        print(f"Error updating subtitle file: {e}")

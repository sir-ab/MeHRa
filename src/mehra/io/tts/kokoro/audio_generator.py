import os
import time
import soundfile as sf
import queue

audio_queue = queue.Queue()

def generate_audio_for_text(engine, text, sentence_index):
    """Generate audio segments for text and queue them for playback."""
    try:
        # Generate speech using Kokoro
        generator = engine.pipeline(
            text, 
            voice=engine.config.voice,
            speed=engine.config.rate, 
            # split_pattern=r'\n+'
        )
        
        # Submit each segment for processing
        segment_count = 0
        for i, (gs, ps, audio) in enumerate(generator):
            if engine.stop_event.is_set():
                break
            
            # Optional: Save the audio to file
            output_file = os.path.join(engine.audio_output_dir, f"segment_{time.time()}_{i}.wav")
            sf.write(output_file, audio, 24000)
            
            # Queue the audio segment for playback
            audio_queue.put((audio, gs, sentence_index))
            segment_count += 1
            
        return segment_count, text
        
    except Exception as e:
        print(f"Error generating audio: {e}")
        return 0, text

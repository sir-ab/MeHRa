import queue
import time
from . import audio_generator

text_queue = queue.Queue()

def process_text(engine):
    """Worker thread that processes text queue and generates audio segments."""
    print("Processor worker thread started")
    
    sentence_index = 0

    while not engine.stop_event.is_set():
        try:
            # Batch multiple queue items into a single text
            combined_text = _batch_queue_items(engine)
            
            # If no text was collected or exit signal received
            if combined_text is None:
                continue
                
            print(f"Processing batch: '{combined_text[:50]}...' ({len(combined_text)} chars)")

            sentences_number = len(combined_text.split(". "))
            sentence_index += 1  # counting each batch not each sentence

            # Process text in the thread pool to allow for parallel processing
            future = engine.thread_pool.submit(audio_generator.generate_audio_for_text, engine, combined_text, sentence_index)
            
            # Mark all items in this batch as done
            for _ in range(sentences_number):
                text_queue.task_done()
                
        except Exception as e:
            print(f"Error in processor worker: {e}")
            # Try to mark the task as done even if there was an error
            try:
                text_queue.task_done()
            except:
                pass

def _batch_queue_items(engine):
    """Batch multiple queue items into a single text with a short delay."""
    combined_text = []
    
    # Get the first item
    try:
        item = text_queue.get()
        if item is None:  # Exit signal
            return None
        combined_text.append(item)
    except queue.Empty:
        return None
        
    # Wait for a short time to see if more items arrive
    batch_end_time = time.time() + engine.config.batch_delay
    
    # Keep collecting items until the batch delay expires or queue is empty
    while time.time() < batch_end_time:
        try:
            item = text_queue.get_nowait()
            if item is None:  # Exit signal
                # Mark all collected items as done and return None
                for _ in range(len(combined_text)):
                    text_queue.task_done()
                return None
            combined_text.append(item)
        except queue.Empty:
            break
            
    # Combine all collected text items with a period between them
    return ". ".join(combined_text) if combined_text else None

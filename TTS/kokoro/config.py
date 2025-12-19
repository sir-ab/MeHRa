class KokoroConfig:
    def __init__(
            self, 
            voice="bf_isabella", 
            lang_code='a', 
            rate=1.0, 
            subtitle_path="subtitles.txt", 
            batch_delay=0.2
        ):
        
        self.voice = voice
        self.lang_code = lang_code
        self.rate = rate
        self.subtitle_path = subtitle_path
        self.batch_delay = batch_delay

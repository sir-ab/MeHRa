import abc

class STTEngineInterface(abc.ABC):
    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def transcribe(self, audio_data):
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass

    @abc.abstractmethod
    def start_stream(self):
        pass

    @abc.abstractmethod
    def stop_stream(self):
        pass

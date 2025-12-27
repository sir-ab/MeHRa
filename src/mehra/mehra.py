from typing import List, Any, AsyncIterator, Dict
from .core.conversation import Conversation
from .models.providers.model_provider import ModelProvider
from .io.tts.tts_interface import TTSEngineInterface
from .io.stt.stt_interface import STTEngineInterface # Add STT engine
from .tools.tool import Tool
from .tools.rag_tool import RAGTool
import threading
import asyncio
import time
import re
import queue

class MeHRa:
    """Modular AI agent that can be extended with various capabilities."""

    def __init__(
        self,
        model_provider: ModelProvider,
        tts_engine : TTSEngineInterface,
        stt_engine: STTEngineInterface,  # Add STT engine
        system_prompt: str = "",
        tools: List[Tool] = None
    ):
        """Initialize the AI agent.

        Args:
            model_provider: Provider for language model inference
            system_prompt: System prompt for the assistant
            tools: List of tools available to the agent
        """
        self.model_provider = model_provider
        self.conversation = Conversation()
        self.tools = tools or []

        # Create a new event loop and start it in a separate thread.
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()
        self.critical_section = asyncio.Lock()

        # Add system prompt as the first message
        self.add_message("assistant", system_prompt)

        # Optional components (to be implemented later)
        self.retriever = None  # For RAG
        self.web_search = None  # For web search
        self.tts_engine = tts_engine  # For text-to-speech
        self.stt_engine = stt_engine  # For speech-to-text


        # STT, TTS, and other children engines state, flags, and queues
        self.transcript_queue = self.stt_engine.transcript_queue if self.stt_engine else None
        self.interrupt_event = self.stt_engine.interrupt_event if self.stt_engine else None
        self.spoken_text = self.tts_engine.spoken_text if self.tts_engine else None
        self.tts_engine.interrupt_event = self.interrupt_event
        self.tts_engine.is_generating = False

        # Flag to interrupt operations
        self.stop_processing = False 

        ##### THREAD
        # Start processing transcript queue in a separate thread
        if self.transcript_queue:
            self.process_transcript_queue_thread = threading.Thread(target=self.process_transcript_queue, daemon=True)
            self.process_transcript_queue_thread.start()        
            
        # Start interrupt detection in separate thread
        # self.interrupt_thread = threading.Thread(target=self.watch_for_interrupt, daemon=False)
        # self.interrupt_thread.start()


    def _start_loop(self):
        """Set the event loop for the thread and run it forever."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent.

        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        
    async def consume_chat(self, user_input: str, **kwargs) -> None:
        """Consume the async generator returned by chat to ensure it runs to completion."""
        async for sentence in self.chat(user_input, **kwargs):
            # You can process each sentence here if needed.
            # print(sentence, end=" ", flush=True)
            print(sentence)
            # pass

    # def watch_for_interrupt(self):
    #     """Watch for an interrupt signal from the STT engine."""
    #     while True:
    #         if self.interrupt_event.is_set():
    #             if self.tts_engine.is_talking:
    #                 self.tts_engine.interrupt()


    def process_transcript_queue(self):
        """Process transcripts from the queue.
            Probably dont have to worry about mehra not getting full context 
            cause we are merging user inputs in Convesation History (see checking user input in mehra.chat)
        """
        while True:
            transcript = self.transcript_queue.get()
            if transcript:
                # Schedule the chat coroutine in the main event loop from this thread.
                asyncio.run_coroutine_threadsafe(self.consume_chat(transcript), self.loop)
                self.transcript_queue.task_done()
            else:
                time.sleep(0.015)

    async def chat(self, user_input: str, **kwargs) -> AsyncIterator[str]:
        """Process user input and generate a response, segmenting by sentence.

        IT IS ASYNC CAUSE OLLAMA.GENERATE RESPONSE STREAM IS ASYNC

        Args:
            user_input: Input text from the user
            **kwargs: Additional parameters to pass to the model

        Yields:
            Sentences from the agent's response
        """
        if not user_input:
            return

        async with self.critical_section:
            # Add user message to conversation
            self.add_message("user", user_input)

            if self.interrupt_event.is_set() or not self.transcript_queue.empty():
                return

            time_to_sleep = min(2.5, len(user_input) / 500)
            print("Time to sleep:", time_to_sleep)
            await asyncio.sleep(time_to_sleep)

            if self.interrupt_event.is_set() or not self.transcript_queue.empty():
                return
            
            # Get streaming response from the model provider
            partial_sentence = ""
            full_response = ""
            # while self.tts_engine.is_talking:
            #     await asyncio.sleep(0.025)

            # await asyncio.sleep(0.05)
            ##############################
            async for chunk in self.model_provider.generate_response_stream(
                self.get_history(),
                **kwargs
            ):
                if self.interrupt_event.is_set() or not self.transcript_queue.empty():
                    break

                self.tts_engine.is_generating = True #sending info to tts_engine (no idea what it does, probably useless)
                full_response += chunk
                partial_sentence += chunk

                # More robust sentence segmentation with regex
                sentences = re.split(r'(?<=[.!?])\s+', partial_sentence)


                # Yield complete sentences
                for i in range(len(sentences) - 1):
                    sentence = sentences[i].strip()


                    # Send to TTS engine if configured
                    if self.tts_engine:
                        self.tts_engine.say(sentence)
                    yield sentence

                # Keep the last partial sentence
                partial_sentence = sentences[-1]

            ########################################
            self.tts_engine.is_generating = False

            spoken_text = []

            while not self.spoken_text.empty() or self.tts_engine.is_talking:
                try:
                    spoken = self.spoken_text.get(timeout=0.05)
                    # print("spoken:", spoken)

                    if spoken:
                        spoken_text.append(spoken)
                        self.spoken_text.task_done()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
            
            if spoken_text != []:
                spoken_text = " ".join(spoken_text)
                self.add_message("assistant", spoken_text)

            # if not self.interrupt_event.is_set():
            #     # Add the full response to the conversation
            #     self.add_message("assistant", full_response)

            #     # Yield any remaining partial sentence
            #     if partial_sentence:
            #         final_sentence = partial_sentence.strip()
            #         # Send to TTS engine if configured
            #         if self.tts_engine and final_sentence:
            #             self.tts_engine.say(final_sentence)
            #         yield partial_sentence.strip()



    def set_retriever(self, retriever: Any) -> None:
        """Set a retriever component for RAG capabilities.

        Args:
            retriever: Retriever component
        """
        self.retriever = retriever

    def set_web_search(self, web_search: Any) -> None:
        """Set a web search component.

        Args:
            web_search: Web search component
        """
        self.web_search = web_search

    def set_tts_engine(self, tts_engine: Any) -> None:
        """Set a text-to-speech engine.

        Args:
            tts_engine: Text-to-speech engine
        """
        self.tts_engine = tts_engine

    def set_stt_engine(self, stt_engine: STTEngineInterface) -> None:
        """Set a speech-to-text engine.

        Args:
            stt_engine: Speech-to-text engine
        """

        self.stt_engine = stt_engine

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.
        Args:
            role: Role of the message (e.g., 'user', 'assistant')
            content: Content of the message
        """
        self.conversation.add_message(role, content)

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history in a format suitable for LLM APIs."""
        return self.conversation.get_history()


    def reset_conversation(self) -> None:
        """Reset the conversation while preserving the system prompt."""
        system_prompt = next((msg.content for msg in self.conversation.messages if msg.role == "system"), None)
        self.conversation.clear()
        if system_prompt:
            self.add_message("system", system_prompt)

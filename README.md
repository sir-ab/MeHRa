# MeHRa: Modular AI Agent Framework

MeHRa is a modular artificial intelligence agent framework designed for extensibility and flexibility. The system enables integration of multiple language models, text-to-speech (TTS) engines, and speech-to-text (STT) engines through a clean, abstracted interface. MeHRa provides out-of-the-box support for Discord-based deployments and supports advanced features including streaming responses and persistent conversation management.

## Key Features

- **Modular Architecture**: Pluggable components for model providers, TTS engines, and STT engines, enabling rapid prototyping and production deployment
- **Multi-Platform Integration**: Native Discord bot support with extensible platform adapters
- **Streaming Capabilities**: Sentence-level response segmentation for optimized latency and user experience
- **Context Management**: Persistent conversation history with intelligent context retention across sessions
- **Extensible Tool System**: Framework for integrating domain-specific tools (e.g., Retrieval-Augmented Generation)

## Project Structure

```
mehra/
├── mehra.py                  # Core agent orchestration
├── core/                     # Core components
│   ├── conversation.py       # Conversation state management
│   ├── message.py           # Message data structures
│   ├── memory.py            # Memory management system
│   └── memory_backends.py   # Pluggable memory implementations
├── models/
│   └── providers/           # LLM provider interfaces and implementations
│       ├── model_provider.py
│       └── ollama_provider.py
├── io/
│   ├── stt/                 # Speech-to-Text engines
│   │   ├── stt_interface.py
│   │   └── whisper_engine.py
│   └── tts/                 # Text-to-Speech engines
│       ├── tts_interface.py
│       ├── pyttsx3_engine.py
│       └── kokoro_engine.py
├── integrations/
│   └── discord/             # Discord bot implementation
│       ├── discord_bot.py
│       └── main.py
└── tools/                   # Extensible tool framework
    ├── tool.py
    └── rag_tool.py
```

## System Requirements

- **Python**: 3.8 or later
- **Memory**: Minimum 4GB RAM (8GB+ recommended for model inference)
- **Storage**: Depends on selected LLM model (typically 2-10GB)

### Optional Dependencies

- **Ollama**: For local LLM inference (recommended for privacy and performance)
- **Discord Bot Token**: Required for Discord integration

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sir-ab/MeHRa.git
cd MeHRa
```

### 2. Set Up Python Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root with the required configuration:

```env
# Discord Configuration (if using Discord integration)
DISCORD_TOKEN=your_discord_bot_token_here

# LLM Configuration
LLM_PROVIDER=ollama  # or other configured provider
LLM_MODEL=neural-chat  # or your preferred model

# Optional: TTS/STT Configuration
TTS_ENGINE=pyttsx3  # or kokoro for enhanced quality
STT_ENGINE=whisper
```

## Quick Start

### Running the Discord Bot

```bash
python src/mehra/run.py
```

The agent will connect to Discord using the token specified in your `.env` file.

### Basic Usage Example

```python
from src.mehra.mehra import MeHRa

# Initialize the agent
agent = MeHRa()

# Process a user message
response = agent.process_message("Hello, how can you help me?")
print(response)
```

## Documentation

For detailed information on extending and customizing MeHRa, refer to:
- [Memory Integration Guide](./MEMORY_INTEGRATION.md) - Implementing custom memory backends
- Source code documentation within individual modules

## Contributing

Contributions are welcome and encouraged. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a Pull Request with a clear description of your changes

Please ensure all code adheres to PEP 8 standards and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

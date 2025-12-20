# MeHRa

MeHRa is a modular AI agent that can be extended with various capabilities, including language models, text-to-speech (TTS), and speech-to-text (STT).

## Features

- **Modular Design**: Easily swap out model providers, TTS engines, and STT engines.
- **Discord Integration**: Run the AI as a Discord bot.
- **Streaming Response**: Supports streaming responses with sentence-level segmentation.
- **Conversation History**: Maintains context across multiple interactions.

## Project Structure

- `mehra.py`: Core logic for the MeHRa agent.
- `discord_bot/`: Discord bot implementation.
- `providers/`: Model provider interfaces and implementations (e.g., Ollama).
- `TTS/`: Text-to-Speech engine interfaces and implementations.
- `STT/`: Speech-to-Text engine interfaces and implementations.
- `core/`: Core classes for conversation and message handling.
- `tools/`: Extensible tools for the agent (e.g., RAG).

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) (for local LLM inference)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sir-ab/MeHRa.git
   cd mehra
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Copy `.env.example` to `.env` and fill in your Discord token:
   ```bash
   cp .env.example .env
   ```

### Usage

To run the agent with the Discord bot:
```bash
python run.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

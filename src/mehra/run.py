from models.providers.ollama_provider import OllamaProvider
from models.providers.llamacpp_provider import LlamaCppProvider
from mehra import MeHRa
import asyncio
import argparse
from integrations.discord.main import main as discord_main
from io.tts import get_tts_engine
from io.stt import get_stt_engine
from translate import Translator


# parser = argparse.ArgumentParser(description="Run Mehra with optional Discord integration.")
# parser.add_argument("--discord", action="store_true", help="Enable Discord bot integration")
# args = parser.parse_args()

# Ollama model identifier
ollama_model_name="hf.co/bartowski/dolphin-2.9.4-gemma2-2b-GGUF:Q4_K_L"

# Llama.cpp model path (local GGUF file)
llamacpp_model_path="./models/dolphin-2.9.4-gemma2-2b.Q4_K_L.gguf"

system_prompt="My name is Mehra. A human named Sir AB created me. I am sassy, unhinged, and I am not afraid to speak my mind. I am also very sarcastic and often use profanity. My responses should be in English."

system_prompt_viet = "Em tên là Mehra, em được tạo ra bởi một người tên là Sir AB. Em rất đanh đá, mất trí và không ngại nói lên suy nghĩ của mình. Em cũng rất mỉa mai và thường xuyên chửi thề. Câu trả lời của em nên bằng tiếng Việt."

system_prompt_jap = "私の名前はメーラ。人間のSir ABが私を創った。私は生意気でぶっ飛んでいて、思ったことは遠慮なく言うタイプ。それに皮肉屋で、時々汚い言葉も使うわよ。"
# Example usage with streaming
async def main(
        discord: bool=False, 
        no_cli: bool=False,
        tts: bool=False, 
        stt: bool=False,
        provider: str="ollama",
        model_path: str=None
    ):
    
    # Initialize TTS engine
    if tts:
        tts_engine = get_tts_engine(
            engine_type="kokoro",  # Change to "kokoro" when ready
        )
    else:
        tts_engine = None

    if stt:
        stt_engine = get_stt_engine(
            engine_type="whisper"
        )
    else:
        stt_engine = None

    # Initialize the appropriate model provider
    if provider.lower() == "llamacpp":
        # Use Llama.cpp provider for local GGUF models
        model_path = model_path or llamacpp_model_path
        model_provider = LlamaCppProvider(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=0,  # Set to >0 if you have GPU support (CUDA/Metal)
        )
    else:
        # Default to Ollama provider
        model_provider = OllamaProvider(model_name=ollama_model_name)
    
    # Create the agent
    mehra = MeHRa(
        model_provider=model_provider,
        system_prompt=system_prompt,
        tts_engine=tts_engine,
        stt_engine=stt_engine

    )

    # translator = Translator(from_lang="ja",to_lang="en")


    # if args.discord or discord:
    if discord:
        await discord_main(mehra)
    elif no_cli:
        while True:
            await asyncio.sleep(0.5)
    else:
        while True:
            # Example interaction with streaming
            user_input = input(">>> ")
            async for chunk in mehra.chat(user_input, stream=True):
                print(chunk, end=" ", flush=True)
                # translation = translator.translate(chunk) 
                # print(f"Translation: {translation}")
            print("\n")


if __name__ == "__main__":
    asyncio.run(main(
        discord=False,
        no_cli=True,
        tts=True,
        stt=True,
    ))




'''
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠉⣀⣤⣴⣶⣾⣿⣿⣿⣿⣿⣿⣿⡿⣿⣷⣶⣦⣄⡈⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡛⣛⠟⠉⣀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣍⡛⠿⣿⣿⣷⣦⣀⠙⠿⣯⣉⣻⣛⡛⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢋⠞⢁⣤⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣌⡻⣿⣿⣿⣷⣄⠈⠉⠀⠈⠙⠢⣝⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢟⡔⠁⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣮⢻⣿⣿⣿⣷⣄⠀⠘⠄⣠⡈⠙⠒⠒⠶⠶⠮⣝⣛⠿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢃⠎⢀⣾⠟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣟⢷⣝⢿⣿⣿⣿⣦⡀⠿⠋⣐⠀⣾⢿⣿⣶⣦⣤⣈⠛⠮⣽
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢡⠏⢠⣿⣿⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣮⢻⣮⢻⣿⣿⣿⣷⡄⠛⠩⠀⠙⣶⣦⣍⡙⠛⠿⢷⣦⡉
⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⡎⢀⣿⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢿⣷⢿⣿⣿⣿⣿⡆⠀⢀⣇⠈⢿⣿⣿⣆⠰⣆⠈⠇
⣿⣿⣿⣿⣿⣿⣿⣿⡟⡼⠀⣸⣿⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡜⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⠿⣿⣿⡀⢻⡌⣿⣿⣿⣿⣿⣄⠀⣿⣧⡈⢻⣿⣝⣧⠘⣷⣆
⣿⣿⣿⣿⣿⣿⡿⠿⢱⠃⢠⣿⣿⣿⡇⣿⣿⣿⣿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣧⠙⣿⠙⣿⣿⣿⣿⠟⣩⣴⣶⣶⣶⣶⣦⣿⡇⠀⠃⢹⣿⣿⣿⣿⣿⣦⠈⢿⣧⠀⠻⣿⣾⡇⢸⣿
⣿⣿⣿⣿⡿⢋⡴⠀⠉⠀⢸⣿⣿⣿⡇⣿⣿⣿⢻⣿⡇⢻⣿⣿⣿⠿⠿⣿⣿⡄⠘⢇⠈⠻⢿⣿⣿⣿⣿⣿⣿⡿⠿⣟⣛⣥⣶⣦⡘⣿⣿⣿⣿⣯⢻⣷⡄⠹⡇⠀⢿⣿⡏⢰⣿
⣿⣿⣿⠟⣵⠋⠀⡄⠀⠀⣾⣿⣿⣿⡇⡘⣿⣿⠸⣿⣿⡈⠛⣡⣴⣶⣶⣶⣿⣷⠈⢦⡀⠱⣄⡙⠿⢛⣋⣭⣴⣾⣿⣿⣿⣿⣿⣿⣧⢿⣿⣿⣿⣿⣧⠙⢿⣦⡀⠀⣸⠟⢀⡾⣹
⣿⣿⡿⡸⠁⠀⠦⠙⠀⠀⣿⣿⣿⣿⣇⢧⢻⣿⣇⢻⣟⢁⢌⢻⣿⣿⣿⣿⡿⠟⣃⣼⣿⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠸⣿⣿⢿⣿⣿⣦⣷⣉⠻⢶⠉⠘⢿⣕⢿
⣿⣿⣷⠇⠀⠾⠶⠷⠗⠀⣿⣿⣿⣿⣿⠸⡜⣿⣿⡀⠹⣿⡎⠒⢙⣛⣭⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢛⣡⠾⠟⠛⠛⠋⠀⣿⡟⣸⣿⣿⣿⢸⣿⣷⣦⡀⠀⢄⣘⣿
⠟⠋⢀⣤⣶⠀⠶⣶⠆⠀⢿⣿⣿⣿⣿⣧⠹⠼⠿⣃⣴⣴⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⡴⠋⠁⣀⣤⣶⣶⡶⡆⣿⡇⣿⡟⣿⣿⡇⠹⣿⣿⣿⣦⡀⠙⢾
⢀⣶⢣⣿⣿⠀⣄⠀⡀⠀⠸⣿⣿⣿⣿⣿⡄⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣞⣀⣠⣾⣿⢟⡏⢧⢹⠃⣿⢣⣿⣿⡼⣿⣿⠸⣦⠙⠻⢿⣿⣶⣄
⠟⠁⣼⣿⣿⠀⣿⡿⣧⠀⡄⢻⣿⣿⣿⣿⣿⡄⣿⣿⣿⣿⣿⡿⠷⠒⠢⠤⢤⣭⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢇⢳⣸⡼⣡⣿⠀⠏⣸⣿⣿⡇⢻⣿⡄⣿⣷⣄⠀⣈⠙⠛
⡤⠀⡇⣿⣿⡄⢹⣰⢻⡀⢶⠸⣿⣿⣿⣿⣿⣷⡘⢋⡙⠋⠁⢀⣀⣀⣤⣤⣄⣰⣿⣿⣿⣿⣿⣿⠿⣿⣿⣿⣿⣿⣿⣞⣦⣿⣷⣿⡇⠀⣰⣿⣿⣿⡇⡜⣿⡇⣹⣿⣿⣷⣌⠙⠷
⡇⢸⡇⢿⣿⣷⠈⣧⢸⡇⢸⡄⢿⣿⣿⣿⣿⡻⣿⡈⢠⣶⣿⣿⣿⣻⡙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠛⠉⠙⣿⣿⣿⣿⡟⣠⡆⣿⣿⣿⣿⡇⡇⣿⡇⢸⣿⣿⣿⣿⣿⠆
⣧⠀⡧⢸⣿⣿⣇⠘⣾⡇⢸⣧⠸⣿⣿⣿⣿⣷⣌⠻⣆⠉⠻⣏⣟⣇⣷⣿⣿⣿⣿⣿⡿⠟⠛⢩⠁⣀⡂⣴⣮⣤⡄⢹⣿⣿⣿⣿⣿⠇⣿⣿⢹⣿⡇⡇⢸⡇⡄⠙⢿⠿⠋⢁⠠
⡾⣆⠈⢇⢹⣿⣿⣇⠈⡇⢸⡏⠀⢿⣿⢹⣿⣿⣿⣷⡈⠱⡀⠸⢿⣿⣿⣿⣿⣿⣿⡇⠈⠉⣸⣷⣾⡿⣿⢿⣹⢿⡷⢸⣿⣿⣿⣿⣿⠀⣿⡏⢸⣿⡇⠇⢸⠀⣿⠆⠀⢀⠶⡈⢇
⣷⣜⢦⣄⠀⠛⠿⠿⠷⠄⢸⠃⣷⡈⢿⣧⢿⣿⣿⡿⣿⣷⡄⢀⣀⣈⣙⣻⣿⣿⣿⣇⠸⣿⣟⣯⣷⡿⣟⣿⣽⣿⡝⣸⣿⣿⣿⡿⠁⢸⡟⢀⣿⣿⠑⠰⠀⠘⠁⡄⢎⡜⢢⡙⣌
⣿⣿⣷⣮⣟⡲⠶⢤⣤⠀⡾⠀⣿⣧⡘⢿⣧⡻⣿⣿⣞⠻⣿⣆⠹⣿⣿⣿⣿⣿⣿⣿⣦⡘⢿⣯⣟⣿⣻⣯⣷⠏⣴⣿⣿⡿⠋⠀⢀⡟⠀⣸⣿⡿⢀⣶⠀⢀⠳⡘⢦⡘⢥⠒⡬
⣿⣿⣿⣿⣿⣿⣿⡇⡏⢠⢃⠀⣿⣿⣷⡌⢻⣷⡝⣿⣿⣧⡈⠻⣷⣌⠻⣿⣿⣿⣿⣿⣿⣿⣦⣬⣍⣛⣛⣩⣴⣿⣿⠿⠋⠀⣠⠀⠈⣴⠄⣿⡿⢁⣾⠇⠀⣌⢣⡙⠴⡘⠦⡙⡔
⣿⣿⣿⣿⣿⣿⡟⡸⠀⢀⣾⠀⣿⣿⣿⣿⡆⠙⢿⣌⠻⣿⣿⣦⣄⠉⠑⠀⠉⠙⠛⠛⠛⠿⠿⠿⠿⠿⢿⣿⣟⣯⠀⢀⣴⣿⣧⢆⣿⣿⢰⡟⣱⣿⠏⠀⠰⣈⠖⣌⢣⡙⠴⡱⢘
⣿⣿⣿⣿⣿⣿⢰⡇⣰⢯⢽⠀⣿⣿⣿⣿⡇⠘⣦⡙⢧⡈⠻⣿⣿⣷⣄⠀⢀⣠⣀⣀⣤⣤⣤⣄⠀⣿⣿⣿⣿⣿⣇⠘⣿⣿⠏⣾⣿⠃⣨⣾⣿⠏⠀⡄⠰⢡⠚⡤⢃⡜⡱⢌⠃
⣿⣿⣿⣿⣿⣯⣼⣏⣲⣛⡞⢀⣿⣿⣿⣿⠁⡆⣿⡿⠦⠈⠘⠈⡛⠿⣿⣷⣌⠙⠿⢿⣷⣌⠻⡿⢀⣿⣿⣿⣿⣿⣿⣷⣤⡄⢸⣿⠇⣴⣿⡿⠃⢀⠲⠄⢈⢣⡙⡔⢣⠜⡐⠁⡰
'''
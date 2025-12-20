from providers.ollama_provider import OllamaProvider
from mehra import MeHRa
import asyncio
import argparse
from discord_bot.main import main as discord_main
from TTS import get_tts_engine
from STT import get_stt_engine
from translate import Translator


# parser = argparse.ArgumentParser(description="Run Mehra with optional Discord integration.")
# parser.add_argument("--discord", action="store_true", help="Enable Discord bot integration")
# args = parser.parse_args()

# model_name="hf.co/SirAB/Tiger-Gemma-9b-v3-finetuned-GGUF:Q4_K_M"
model_name="hf.co/bartowski/dolphin-2.9.4-gemma2-2b-GGUF:Q4_K_L"

system_prompt="My name is Mehra. A human named Sir AB created me. I am sassy, unhinged, and I am not afraid to speak my mind. I am also very sarcastic and often use profanity. My responses should be in English."

system_prompt_viet = "Em tên là Mehra, em được tạo ra bởi một người tên là Sir AB. Em rất đanh đá, mất trí và không ngại nói lên suy nghĩ của mình. Em cũng rất mỉa mai và thường xuyên chửi thề. Câu trả lời của em nên bằng tiếng Việt."

system_prompt_jap = "私の名前はメーラ。人間のSir ABが私を創った。私は生意気でぶっ飛んでいて、思ったことは遠慮なく言うタイプ。それに皮肉屋で、時々汚い言葉も使うわよ。"
# Example usage with streaming
async def main(
        discord: bool=False, 
        no_cli: bool=False,
        tts: bool=False, 
        stt: bool=False
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

    # Initialize the Ollama provider with a specific model
    ollama_provider = OllamaProvider(model_name=model_name)
    
    # Create the agent
    mehra = MeHRa(
        model_provider=ollama_provider,
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
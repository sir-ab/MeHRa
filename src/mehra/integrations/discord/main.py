import asyncio
import discord
from .discord_bot import DiscordBot

import os
from dotenv import load_dotenv

load_dotenv()

# Discord bot token and user ID from environment variables
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
SPECIFIC_USER_ID = os.getenv('SPECIFIC_USER_ID')
GURA_ID = '1286022692926918677'

async def main(mehra_instance, **kwargs):
    intents = discord.Intents.default()
    intents.message_content = True
    
    # Create and run the Discord bot
    bot = DiscordBot(
        mehra_instance=mehra_instance,
        read0nly_ai_id=GURA_ID,
        intents=intents,
        read0nly = False,
        enable_history=False,
        **kwargs
    )
    
    try:
        await bot.start(DISCORD_TOKEN)
    finally:
        # Cleanup
        pass

if __name__ == "__main__":
    pass

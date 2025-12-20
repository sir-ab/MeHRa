import discord
import asyncio

class DiscordBot(discord.Client):
    def __init__(
            self, 
            mehra_instance,
            read0nly_ai_id: str, 
            enable_history: bool=False,
            read0nly: bool=False,
            *args, **kwargs
        ):

        super().__init__(*args, **kwargs)
        self.mehra = mehra_instance
        self.read0nly_ai_id = read0nly_ai_id
        self.read0nly = read0nly
        self.assistant_user_id = None  # Initialize to None
        self.saved_channel_history = set()  # Store IDs of channels whose history has been saved

        self.enable_history = enable_history

    async def on_ready(self):
        print(f'Logged in as {self.user}')
        self.assistant_user_id = self.user.id  # Set ID when bot is ready
        # print(f"Bot's user ID: {self.assistant_user_id}") # confirm that the bot has obtained the id
    
    async def on_message(self, message):
        if str(message.author.id) != str(self.assistant_user_id):
            print(f"Message from {message.author}: {message.content}")
            
            if not self.read0nly:
                # Check if we've already saved this channel's history
                if self.enable_history and message.channel.id not in self.saved_channel_history:
                    await self.get_channel_history(message.channel)
                    self.saved_channel_history.add(message.channel.id)  # Mark channel as saved
                    print(f"Saved channel history for channel: {message.channel.name}")

                message_template = f"{message.author.name}: {message.content}\n"

                async for sentence in self.mehra.chat(message_template):
                    # Create tasks for both sending message and queueing audio
                    # send_task = asyncio.create_task(self.send_message(message.channel, sentence))
                    await self.send_message(message.channel, sentence)

            elif self.read0nly:
                if str(message.author.id) == self.read0nly_ai_id:
                    print("Gura detected")
                    message_segment = str(message.content).split(". ")

                    i = 0
                    while len(message_segment) > 0:
                        growing_number = max(2 * i, 1)  # Ensure growing_number is at least 1

                        text_stream_batch = '. '.join(message_segment[:growing_number]) + '. '  # Properly join sentences

                        self.mehra.tts_engine.say(text_stream_batch)

                        # Remove processed segments
                        del message_segment[:growing_number]
                        i += 1
    
    async def send_message(self, channel, sentence):
        await channel.send(sentence)
        # print(f"Sent message: {sentence}")

    async def get_channel_history(self, channel, limit=30):
        """
        Retrieves the channel history and saves it to mehra.

        Args:
            channel: The Discord channel object.
            limit: The number of messages to retrieve (optional).  If None, retrieves all history.
        """
        try:
            # self.mehra.add_message("user", "here were the last {limit} messages in this channel")

            messages = [message async for message in channel.history(limit=limit, oldest_first=True)]  # Convert async generator to list

            for index, message in enumerate(messages):
                if index == len(messages) - 1:  # Stop before the last iteration
                    break  

                author_name = str(message.author.name)
                author_id = str(message.author.id)
                content = message.content
                history_message_template = f"{author_name}: {content}\n"

                if author_id != str(self.assistant_user_id):  # Changed specific_user_id to string
                    role = "user"
                elif author_id == str(self.assistant_user_id):  # Comparing to the bot user ID
                    role = "assistant"
                else:
                    print(f"Skipping message from unknown user {message.author.name} ({message.author.id})")
                    continue  # Skip message from unknown user

                if role == "assistant":
                    self.mehra.add_message(role, content)
                    # print(f"Added to mehra: {role} - {content}")

                elif role == "user":
                    self.mehra.add_message(role, history_message_template)
                    # print(f"Added to mehra: {role} - {history_message_template}")



        except discord.errors.Forbidden:
            print(f"Error: Insufficient permissions to access channel history in channel {channel.name} ({channel.id}).  Bot may need 'Read Message History' permission.")
        except Exception as e:
            print(f"An error occurred while saving channel history: {e}")
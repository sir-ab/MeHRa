from dataclasses import dataclass

@dataclass
class Message:
    """Class representing a message in a conversation."""
    role: str  # "system", "user", or "assistant"
    content: str

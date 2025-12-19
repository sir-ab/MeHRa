from dataclasses import dataclass, field
from typing import List, Dict
from core.message import Message


@dataclass
class Conversation:
    """Class representing a conversation history."""
    messages: List[Message] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.
        If the last message has the same role, append the new content to it.
        """
        if self.messages and self.messages[-1].role == role:
            self.messages[-1].content += " " + content
        else:
            self.messages.append(Message(role=role, content=content))

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history in a format suitable for LLM APIs."""
        # print("#######################")
        # for x in self.messages:
        #     print({"role": x.role, "content": x.content})
        # print("#######################")
            

        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []

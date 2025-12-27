class Tool:
    """Base class for tools that the agent can use."""

    def __init__(self, name: str, description: str):
        """Initialize a tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
        """
        self.name = name
        self.description = description

    def run(self, input_data: str) -> str:
        """Run the tool with the given input.

        Args:
            input_data: Input for the tool

        Returns:
            Result from running the tool
        """
        raise NotImplementedError("Subclasses must implement this method")

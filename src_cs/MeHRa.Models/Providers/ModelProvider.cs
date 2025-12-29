namespace MeHRa.Models.Providers;

/// <summary>
/// Base class for model providers.
/// </summary>
public abstract class ModelProvider
{
    /// <summary>
    /// Generate a response from the model.
    /// </summary>
    /// <param name="messages">List of message objects with role and content</param>
    /// <param name="cancellationToken">Cancellation token for async operations</param>
    /// <returns>Complete text response</returns>
    public abstract Task<string> GenerateResponseAsync(
        List<Message> messages,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generate a streaming response from the model.
    /// </summary>
    /// <param name="messages">List of message objects with role and content</param>
    /// <param name="cancellationToken">Cancellation token for async operations</param>
    /// <returns>Async enumerable of text chunks</returns>
    public abstract IAsyncEnumerable<string> GenerateResponseStreamAsync(
        List<Message> messages,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents a message with role and content.
/// </summary>
public class Message
{
    /// <summary>
    /// Role of the message author (system, user, assistant)
    /// </summary>
    public required string Role { get; set; }

    /// <summary>
    /// Content of the message
    /// </summary>
    public required string Content { get; set; }
}

/// <summary>
/// Generation parameters for model inference.
/// </summary>
public class GenerationParams
{
    /// <summary>
    /// Controls randomness (0.0 to 1.0, default: 0.7)
    /// </summary>
    public float Temperature { get; set; } = 0.7f;

    /// <summary>
    /// Controls diversity (0.0 to 1.0, default: 0.95)
    /// </summary>
    public float TopP { get; set; } = 0.95f;

    /// <summary>
    /// Maximum number of tokens to generate (default: 512)
    /// </summary>
    public int MaxTokens { get; set; } = 512;

    /// <summary>
    /// Additional parameters as key-value pairs
    /// </summary>
    public Dictionary<string, object> AdditionalParams { get; set; } = new();
}

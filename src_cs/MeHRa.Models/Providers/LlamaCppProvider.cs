using LLama;
using LLama.Common;
using LLama.Native;

namespace MeHRa.Models.Providers;

/// <summary>
/// Represents latency metrics for inference operations.
/// </summary>
public class LatencyMetrics
{
    /// <summary>
    /// Time to prepare inference (format prompt, setup params) in milliseconds.
    /// </summary>
    public double SetupLatency { get; set; }

    /// <summary>
    /// Time to first token from model in milliseconds.
    /// </summary>
    public double FirstTokenLatency { get; set; }

    /// <summary>
    /// Average time per token in milliseconds.
    /// </summary>
    public double TimePerToken { get; set; }

    /// <summary>
    /// Total inference time in milliseconds (including setup).
    /// </summary>
    public double TotalLatency { get; set; }

    /// <summary>
    /// Number of tokens generated.
    /// </summary>
    public int TokensGenerated { get; set; }

    /// <summary>
    /// Tokens generated per second.
    /// </summary>
    public double TokensPerSecond { get; set; }

    public override string ToString()
    {
        return $"LatencyMetrics(" +
               $"Setup={SetupLatency:F2}ms, " +
               $"TTFT={FirstTokenLatency:F2}ms, " +
               $"TPS={TokensPerSecond:F2} tokens/s, " +
               $"Tokens={TokensGenerated}, " +
               $"Total={TotalLatency:F2}ms)";
    }
}

/// <summary>
/// Llama.cpp model provider for running GGUF format models locally using LLamaSharp.
/// </summary>
public class LlamaCppProvider : ModelProvider, IAsyncDisposable
{
    private readonly string _modelPath;
    private readonly int _contextSize;
    private readonly int _threads;
    private readonly int _gpuLayers;
    private LLamaWeights? _model;
    private LLamaContext? _context;
    private LatencyMetrics _lastMetrics = new();

    /// <summary>
    /// Initialize the Llama.cpp provider.
    /// </summary>
    /// <param name="modelPath">Path to the GGUF model file</param>
    /// <param name="contextSize">Context window size (default: 2048)</param>
    /// <param name="threads">Number of threads for inference (default: 8)</param>
    /// <param name="gpuLayers">Number of layers to offload to GPU (0 = CPU only, default: 0)</param>
    /// <exception cref="FileNotFoundException">Thrown if model file is not found</exception>
    /// <exception cref="InvalidOperationException">Thrown if model loading fails</exception>
    public LlamaCppProvider(
        string modelPath,
        int contextSize = 2048,
        int threads = 8,
        int gpuLayers = 0)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}");
        }

        _modelPath = modelPath;
        _contextSize = contextSize;
        _threads = threads;
        _gpuLayers = gpuLayers;

        LoadModel();
    }

    /// <summary>
    /// Load the model from the specified path.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if model loading fails</exception>
    private void LoadModel()
    {
        try
        {
            var parameters = new ModelParams(_modelPath)
            {
                ContextSize = (uint)_contextSize,
                ThreadCount = _threads,
                GpuLayerCount = _gpuLayers,
                Verbose = false,
            };

            _model = LLamaWeights.LoadFromFile(parameters);
            _context = _model.CreateContext(parameters);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load model from {_modelPath}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Generate a complete response using Llama.cpp.
    /// </summary>
    /// <param name="messages">List of message objects with role and content</param>
    /// <param name="cancellationToken">Cancellation token for async operations</param>
    /// <returns>Complete text response</returns>
    public override async Task<string> GenerateResponseAsync(
        List<Message> messages,
        CancellationToken cancellationToken = default)
    {
        var response = "";
        await foreach (var chunk in GenerateResponseStreamAsync(messages, cancellationToken))
        {
            response += chunk;
        }
        return response;
    }

    /// <summary>
    /// Generate a streaming response using Llama.cpp.
    /// </summary>
    /// <param name="messages">List of message objects with role and content</param>
    /// <param name="cancellationToken">Cancellation token for async operations</param>
    /// <returns>Async enumerable of text chunks</returns>
    public override async IAsyncEnumerable<string> GenerateResponseStreamAsync(
        List<Message> messages,
        CancellationToken cancellationToken = default)
    {
        if (_model == null || _context == null)
        {
            throw new InvalidOperationException("Model not loaded. Call LoadModel first.");
        }

        // Record overall start time (including setup)
        var overallStart = DateTime.UtcNow;

        // Setup phase: format messages to prompt
        var setupStart = DateTime.UtcNow;
        var prompt = FormatMessagesToPrompt(messages);
        var setupEnd = DateTime.UtcNow;

        // Initialize latency tracking
        var metrics = new LatencyMetrics();
        metrics.SetupLatency = (setupEnd - setupStart).TotalMilliseconds;

        var executor = new StatelessExecutor(_model, _context.Params);
        var inferenceParams = new InferenceParams
        {
            Temperature = 0.7f,
            TopP = 0.95f,
            AntiPrompts = new List<string> { "User:", "Assistant:" },
        };

        DateTime? firstTokenTime = null;
        var tokenCount = 0;
        var modelStartTime = setupEnd;

        await foreach (var token in executor.InferAsync(prompt, inferenceParams, cancellationToken))
        {
            tokenCount++;

            // Record first token latency (from model start)
            if (firstTokenTime == null)
            {
                firstTokenTime = DateTime.UtcNow;
                metrics.FirstTokenLatency = (firstTokenTime.Value - modelStartTime).TotalMilliseconds;
            }

            yield return token;
        }

        // Calculate final metrics
        var endTime = DateTime.UtcNow;
        metrics.TotalLatency = (endTime - overallStart).TotalMilliseconds;
        metrics.TokensGenerated = tokenCount;

        if (tokenCount > 0)
        {
            var modelDuration = (endTime - modelStartTime).TotalSeconds;
            metrics.TimePerToken = (metrics.TotalLatency - metrics.SetupLatency) / tokenCount;
            metrics.TokensPerSecond = modelDuration > 0 ? tokenCount / modelDuration : 0;
        }

        _lastMetrics = metrics;
    }

    /// <summary>
    /// Convert messages to a prompt string.
    /// </summary>
    /// <param name="messages">List of message objects with role and content</param>
    /// <returns>Formatted prompt string</returns>
    private string FormatMessagesToPrompt(List<Message> messages)
    {
        var prompt = "";

        foreach (var message in messages)
        {
            var role = message.Role.ToLower();

            prompt += role switch
            {
                "system" => $"{message.Content}\n\n",
                "user" => $"User: {message.Content}\n",
                "assistant" => $"Assistant: {message.Content}\n",
                _ => $"{role}: {message.Content}\n"
            };
        }

        prompt += "Assistant:";
        return prompt;
    }

    /// <summary>
    /// Get information about the loaded model.
    /// </summary>
    /// <returns>Dictionary containing model information</returns>
    public Dictionary<string, object> GetModelInfo()
    {
        return new Dictionary<string, object>
        {
            { "ModelPath", _modelPath },
            { "ContextSize", _contextSize },
            { "Threads", _threads },
            { "GpuLayers", _gpuLayers },
            { "ModelLoaded", _model != null },
        };
    }

    /// <summary>
    /// Get the latency metrics from the last inference.
    /// </summary>
    /// <returns>LatencyMetrics object with measurement data</returns>
    public LatencyMetrics GetLatencyMetrics()
    {
        return _lastMetrics;
    }

    /// <summary>
    /// Dispose of model resources.
    /// </summary>
    public async ValueTask DisposeAsync()
    {
        _context?.Dispose();
        _model?.Dispose();
        await Task.CompletedTask;
    }
}

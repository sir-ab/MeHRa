using MeHRa.Models.Providers;

namespace MeHRa.Examples;

/// <summary>
/// Example usage of the LlamaCppProvider.
/// </summary>
public class LlamaCppExample
{
    public static async Task Main(string[] args)
    {
        const string modelPath = "./models/dolphin-2.9.4-gemma2-2b.Q4_K_L.gguf";

        try
        {
            // Initialize the provider
            var provider = new LlamaCppProvider(
                modelPath: modelPath,
                contextSize: 2048,
                threads: 8,
                gpuLayers: 0  // Set to > 0 if you have GPU support
            );

            // Create sample messages
            var messages = new List<Message>
            {
                new Message 
                { 
                    Role = "system", 
                    Content = "You are a helpful assistant named MeHRa, created by Sir AB." 
                },
                new Message 
                { 
                    Role = "user", 
                    Content = "What is your name?" 
                }
            };

            // Example 1: Generate complete response
            Console.WriteLine("=== Complete Response ===");
            var response = await provider.GenerateResponseAsync(messages);
            Console.WriteLine(response);

            // Example 2: Generate streaming response
            Console.WriteLine("\n=== Streaming Response ===");
            var streamingMessages = new List<Message>
            {
                new Message 
                { 
                    Role = "system", 
                    Content = "You are a helpful assistant named MeHRa, created by Sir AB." 
                },
                new Message 
                { 
                    Role = "user", 
                    Content = "Tell me a short joke." 
                }
            };

            await foreach (var chunk in provider.GenerateResponseStreamAsync(streamingMessages))
            {
                Console.Write(chunk);
            }
            Console.WriteLine();

            // Example 3: Get model info
            Console.WriteLine("\n=== Model Information ===");
            var modelInfo = provider.GetModelInfo();
            foreach (var kvp in modelInfo)
            {
                Console.WriteLine($"{kvp.Key}: {kvp.Value}");
            }

            // Cleanup
            await provider.DisposeAsync();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            Environment.Exit(1);
        }
    }
}

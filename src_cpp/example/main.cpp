#include "mehra/models/providers/llamacpp_provider.hpp"
#include <iostream>
#include <vector>

using namespace mehra::models::providers;

int main(int argc, char* argv[]) {
    try {
        // Initialize the provider
        const std::string model_path = "./models/dolphin-2.9.4-gemma2-2b.Q4_K_L.gguf";
        
        LlamaCppProvider provider(
            model_path,
            2048,  // context size
            8,     // threads
            0      // gpu layers (0 = CPU only)
        );
        
        // Create sample messages
        std::vector<Message> messages = {
            {
                "system",
                "You are a helpful assistant named MeHRa, created by Sir AB."
            },
            {
                "user",
                "What is your name?"
            }
        };
        
        // Example 1: Generate complete response
        std::cout << "=== Complete Response ===" << std::endl;
        std::string response = provider.generate_response(messages);
        std::cout << response << std::endl;
        
        // Print latency metrics
        std::cout << "\n=== Metrics ===" << std::endl;
        std::cout << provider.get_latency_metrics().to_string() << std::endl;
        
        // Example 2: Generate streaming response
        std::cout << "\n=== Streaming Response ===" << std::endl;
        
        std::vector<Message> stream_messages = {
            {
                "system",
                "You are a helpful assistant named MeHRa, created by Sir AB."
            },
            {
                "user",
                "Tell me a short joke."
            }
        };
        
        provider.generate_response_stream(
            stream_messages,
            [](const std::string& chunk) {
                std::cout << chunk << std::flush;
            }
        );
        
        std::cout << "\n\n=== Metrics ===" << std::endl;
        std::cout << provider.get_latency_metrics().to_string() << std::endl;
        
        // Example 3: Get model info
        std::cout << "\n=== Model Information ===" << std::endl;
        std::cout << provider.get_model_info() << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

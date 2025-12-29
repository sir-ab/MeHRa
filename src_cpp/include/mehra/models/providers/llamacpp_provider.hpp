#pragma once

#include "model_provider.hpp"
#include <llama.h>
#include <vector>
#include <memory>

namespace mehra::models::providers {

/**
 * @class LlamaCppProvider
 * @brief Llama.cpp model provider for running GGUF format models locally
 */
class LlamaCppProvider : public ModelProvider {
public:
    /**
     * @brief Initialize the Llama.cpp provider
     * @param model_path Path to the GGUF model file
     * @param n_ctx Context window size (default: 2048)
     * @param n_threads Number of threads for inference (default: 8)
     * @param n_gpu_layers Number of layers to offload to GPU (default: 0)
     * @throws std::runtime_error if model loading fails
     */
    LlamaCppProvider(
        const std::string& model_path,
        int n_ctx = 2048,
        int n_threads = 8,
        int n_gpu_layers = 0);

    /**
     * @brief Destructor - cleans up model resources
     */
    ~LlamaCppProvider() override;

    /**
     * @brief Generate a complete response from the model
     * @param messages List of message objects
     * @param params Generation parameters
     * @return Complete text response
     */
    std::string generate_response(
        const std::vector<Message>& messages,
        const GenerationParams& params = GenerationParams()) override;

    /**
     * @brief Generate a streaming response from the model
     * @param messages List of message objects
     * @param callback Function to call for each generated token
     * @param params Generation parameters
     */
    void generate_response_stream(
        const std::vector<Message>& messages,
        std::function<void(const std::string&)> callback,
        const GenerationParams& params = GenerationParams()) override;

    /**
     * @brief Get latency metrics from the last inference
     * @return LatencyMetrics object with measurement data
     */
    LatencyMetrics get_latency_metrics() const override { return last_metrics_; }

    /**
     * @brief Get information about the loaded model
     * @return Dictionary-like structure with model info
     */
    std::string get_model_info() const;

private:
    /**
     * @brief Load the model from the specified path
     * @throws std::runtime_error if model loading fails
     */
    void load_model();

    /**
     * @brief Convert messages to a prompt string
     * @param messages List of message objects
     * @return Formatted prompt string
     */
    std::string format_messages_to_prompt(const std::vector<Message>& messages) const;

    // Member variables
    std::string model_path_;
    int n_ctx_;
    int n_threads_;
    int n_gpu_layers_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    LatencyMetrics last_metrics_;
};

} // namespace mehra::models::providers

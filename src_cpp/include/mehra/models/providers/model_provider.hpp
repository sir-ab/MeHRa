#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace mehra::models::providers {

/**
 * @struct Message
 * @brief Represents a message with role and content
 */
struct Message {
    std::string role;      ///< Role of the message author (system, user, assistant)
    std::string content;   ///< Content of the message
};

/**
 * @struct LatencyMetrics
 * @brief Container for latency measurements
 */
struct LatencyMetrics {
    double setup_latency = 0.0;         ///< Time to prepare inference (format, params) in ms
    double first_token_latency = 0.0;   ///< Time to first token from model in ms
    double time_per_token = 0.0;        ///< Average ms per token
    double total_latency = 0.0;         ///< Total inference time in ms
    int tokens_generated = 0;           ///< Number of tokens generated
    double tokens_per_second = 0.0;     ///< Tokens generated per second

    /**
     * @brief Convert metrics to string representation
     * @return Formatted string with all metrics
     */
    std::string to_string() const;
};

/**
 * @struct GenerationParams
 * @brief Parameters for model inference
 */
struct GenerationParams {
    float temperature = 0.7f;    ///< Controls randomness (0.0 to 1.0)
    float top_p = 0.95f;        ///< Controls diversity (0.0 to 1.0)
    int max_tokens = 512;       ///< Maximum number of tokens to generate
};

/**
 * @class ModelProvider
 * @brief Abstract base class for model providers
 */
class ModelProvider {
public:
    virtual ~ModelProvider() = default;

    /**
     * @brief Generate a complete response from the model
     * @param messages List of message objects
     * @param params Generation parameters
     * @return Complete text response
     */
    virtual std::string generate_response(
        const std::vector<Message>& messages,
        const GenerationParams& params = GenerationParams()) = 0;

    /**
     * @brief Generate a streaming response from the model
     * @param messages List of message objects
     * @param callback Function to call for each generated token
     * @param params Generation parameters
     */
    virtual void generate_response_stream(
        const std::vector<Message>& messages,
        std::function<void(const std::string&)> callback,
        const GenerationParams& params = GenerationParams()) = 0;

    /**
     * @brief Get latency metrics from the last inference
     * @return LatencyMetrics object with measurement data
     */
    virtual LatencyMetrics get_latency_metrics() const = 0;
};

} // namespace mehra::models::providers

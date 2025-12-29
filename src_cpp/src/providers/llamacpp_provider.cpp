#include "mehra/models/providers/llamacpp_provider.hpp"
#include <chrono>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <iostream>

namespace mehra::models::providers {

LlamaCppProvider::LlamaCppProvider(
    const std::string& model_path,
    int n_ctx,
    int n_threads,
    int n_gpu_layers)
    : model_path_(model_path),
      n_ctx_(n_ctx),
      n_threads_(n_threads),
      n_gpu_layers_(n_gpu_layers) {
    
    // Check if model file exists
    std::ifstream f(model_path);
    if (!f.good()) {
        throw std::runtime_error("Model file not found: " + model_path);
    }
    
    load_model();
}

LlamaCppProvider::~LlamaCppProvider() {
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_free_model(model_);
    }
}

void LlamaCppProvider::load_model() {
    try {
        // Initialize llama backend
        llama_backend_init(false);  // false = use CPU, true = use GPU
        
        // Load model parameters
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers_;
        
        // Load the model
        model_ = llama_load_model_from_file(model_path_.c_str(), model_params);
        if (!model_) {
            throw std::runtime_error("Failed to load model from " + model_path_);
        }
        
        // Create context
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx_;
        ctx_params.n_threads = n_threads_;
        ctx_params.n_threads_batch = n_threads_;
        
        ctx_ = llama_new_context_with_model(model_, ctx_params);
        if (!ctx_) {
            llama_free_model(model_);
            model_ = nullptr;
            throw std::runtime_error("Failed to create context");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

std::string LlamaCppProvider::format_messages_to_prompt(
    const std::vector<Message>& messages) const {
    
    std::string prompt;
    
    for (const auto& msg : messages) {
        std::string role = msg.role;
        std::transform(role.begin(), role.end(), role.begin(), ::tolower);
        
        if (role == "system") {
            prompt += msg.content + "\n\n";
        } else if (role == "user") {
            prompt += "User: " + msg.content + "\n";
        } else if (role == "assistant") {
            prompt += "Assistant: " + msg.content + "\n";
        } else {
            prompt += role + ": " + msg.content + "\n";
        }
    }
    
    prompt += "Assistant:";
    return prompt;
}

std::string LlamaCppProvider::generate_response(
    const std::vector<Message>& messages,
    const GenerationParams& params) {
    
    std::string response;
    generate_response_stream(messages, 
        [&response](const std::string& chunk) {
            response += chunk;
        }, 
        params);
    
    return response;
}

void LlamaCppProvider::generate_response_stream(
    const std::vector<Message>& messages,
    std::function<void(const std::string&)> callback,
    const GenerationParams& params) {
    
    if (!model_ || !ctx_) {
        throw std::runtime_error("Model not loaded");
    }
    
    // Record overall start time
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    // Setup phase: format messages to prompt
    auto setup_start = std::chrono::high_resolution_clock::now();
    std::string prompt = format_messages_to_prompt(messages);
    auto setup_end = std::chrono::high_resolution_clock::now();
    
    // Initialize latency tracking
    LatencyMetrics metrics;
    metrics.setup_latency = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
    
    // Prepare batch
    llama_batch batch = llama_batch_init(512, 0, 1);
    
    // Tokenize input
    std::vector<llama_token> tokens_list;
    tokens_list.resize(prompt.size() + 256);  // Extra space for safety
    int n_tokens = llama_tokenize(model_, prompt.c_str(), tokens_list.data(), tokens_list.size(), false);
    
    if (n_tokens < 0) {
        llama_batch_free(batch);
        throw std::runtime_error("Tokenization failed");
    }
    
    tokens_list.resize(n_tokens);
    
    auto model_start_time = std::chrono::high_resolution_clock::now();
    bool first_token = true;
    int token_count = 0;
    
    // Add tokens to batch
    for (int i = 0; i < n_tokens; ++i) {
        llama_batch_add(batch, tokens_list[i], i, {0}, false);
    }
    
    batch.n_tokens = n_tokens;
    
    // Generate tokens
    int n_cur = n_tokens;
    while (n_cur < params.max_tokens) {
        // Decode batch
        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed");
        }
        
        // Get next token
        llama_token next_token = llama_sampler_sample(ctx_, nullptr);
        
        if (llama_token_is_eog(model_, next_token)) {
            break;
        }
        
        // Decode token to string
        char buf[16];
        int piece_size = llama_token_to_piece(model_, next_token, buf, sizeof(buf));
        std::string piece(buf, piece_size);
        
        // Record first token latency
        if (first_token) {
            auto first_token_time = std::chrono::high_resolution_clock::now();
            metrics.first_token_latency = 
                std::chrono::duration<double, std::milli>(first_token_time - model_start_time).count();
            first_token = false;
        }
        
        token_count++;
        callback(piece);
        
        // Prepare batch for next token
        batch.clear();
        llama_batch_add(batch, next_token, n_cur, {0}, true);
        batch.n_tokens = 1;
        
        n_cur++;
    }
    
    // Calculate final metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    metrics.total_latency = std::chrono::duration<double, std::milli>(end_time - overall_start).count();
    metrics.tokens_generated = token_count;
    
    if (token_count > 0) {
        metrics.time_per_token = (metrics.total_latency - metrics.setup_latency) / token_count;
        double model_duration = std::chrono::duration<double>(end_time - model_start_time).count();
        metrics.tokens_per_second = model_duration > 0 ? token_count / model_duration : 0;
    }
    
    last_metrics_ = metrics;
    
    // Cleanup
    llama_batch_free(batch);
    llama_backend_free();
}

std::string LlamaCppProvider::get_model_info() const {
    std::ostringstream oss;
    oss << "ModelPath: " << model_path_ << "\n"
        << "ContextSize: " << n_ctx_ << "\n"
        << "Threads: " << n_threads_ << "\n"
        << "GpuLayers: " << n_gpu_layers_ << "\n"
        << "ModelLoaded: " << (model_ != nullptr ? "true" : "false");
    return oss.str();
}

} // namespace mehra::models::providers

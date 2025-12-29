#include "mehra/models/providers/model_provider.hpp"
#include <sstream>
#include <iomanip>

namespace mehra::models::providers {

std::string LatencyMetrics::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "LatencyMetrics("
        << "Setup=" << setup_latency << "ms, "
        << "TTFT=" << first_token_latency << "ms, "
        << "TPS=" << tokens_per_second << " tokens/s, "
        << "Tokens=" << tokens_generated << ", "
        << "Total=" << total_latency << "ms)";
    return oss.str();
}

} // namespace mehra::models::providers

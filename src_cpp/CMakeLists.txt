cmake_minimum_required(VERSION 3.15)
project(MeHRa_CPP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find llama.cpp
find_package(Llama REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/model_provider.cpp
    src/providers/llamacpp_provider.cpp
)

# Create library
add_library(mehra_models STATIC ${SOURCES})

# Link libraries
target_link_libraries(mehra_models PUBLIC Llama::llama)

# Install headers
install(DIRECTORY include/ DESTINATION include)
install(TARGETS mehra_models DESTINATION lib)

# Optional: Build example
option(BUILD_EXAMPLE "Build example executable" ON)

if(BUILD_EXAMPLE)
    add_executable(mehra_example example/main.cpp)
    target_link_libraries(mehra_example PRIVATE mehra_models)
endif()

include(FetchContent)

# Fetch tokenizers-cpp
FetchContent_Declare(
  tokenizers-cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
  GIT_TAG main
)

# Make tokenizers-cpp available
FetchContent_MakeAvailable(tokenizers-cpp)

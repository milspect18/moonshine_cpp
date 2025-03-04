# moonshine_cpp
C++ inference library for Moonshine STT models

## Integration

The library is currently built for static linking only.  The static library does have a shared library dependency on the onnx-runtime.

### Using FetchContent

You can integrate this library into your CMake project using FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(
    moonshine_cpp
    GIT_REPOSITORY https://github.com/milspect18/moonshine_cpp.git
    GIT_TAG main
)

FetchContent_MakeAvailable(moonshine_cpp)

# Link against the library in your target
target_link_libraries(your_target PRIVATE
    moonshine::moonshine_cpp
)
```

## Dependencies

This library depends on the [onnx runtime](https://github.com/microsoft/onnxruntime).  The build process attempts to fetch a pre-built binary of v1.20 from the onxx-runtime repo.

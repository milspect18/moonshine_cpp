# moonshine_cpp
C++ inference library for Moonshine STT models

## Integration

### Using FetchContent

You can integrate this library into your CMake project using FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(
    moonshine_cpp
    GIT_REPOSITORY https://github.com/yourusername/moonshine_cpp.git
    GIT_TAG v0.1.0  # specify the version you want
)
FetchContent_MakeAvailable(moonshine_cpp)

# Link against the library in your target
target_link_libraries(your_target PRIVATE moonshine::moonshine_cpp)
```

### Installation

Alternatively, you can install the library system-wide:

```bash
git clone https://github.com/yourusername/moonshine_cpp.git
cd moonshine_cpp
mkdir build && cd build
cmake ..
make
sudo make install
```

Then in your project's CMakeLists.txt:

```cmake
find_package(moonshine_cpp REQUIRED)
target_link_libraries(your_target PRIVATE moonshine::moonshine_cpp)
```

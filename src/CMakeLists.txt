include(FetchPrebuiltONNXRuntime)
include(FetchTokenizers)

add_library(moonshine_cpp STATIC
    moonshine_onnx_model.cpp
    moonshine_transcribe.cpp
)

# Create a namespaced alias for the library
add_library(moonshine::moonshine_cpp ALIAS moonshine_cpp)

target_include_directories(moonshine_cpp PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${tokenizers-cpp_SOURCE_DIR}/include
)

target_link_libraries(moonshine_cpp PRIVATE
    ONNXRuntime
    tokenizers_cpp
)

# Set target properties
set_target_properties(moonshine_cpp PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME moonshine_cpp
    OUTPUT_NAME moonshine_cpp
)

# Install the library
install(
    TARGETS moonshine_cpp
    EXPORT moonshine_cpp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

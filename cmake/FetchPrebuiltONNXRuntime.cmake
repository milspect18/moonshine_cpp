# FindONNXRuntime.cmake
# Downloads and extracts pre-built ONNX Runtime v1.20.1 binaries
# Sets up variables for easy linking into your project

cmake_minimum_required(VERSION 3.12)

# Set the version we want
set(ORT_VERSION "1.20.1" CACHE STRING "ONNX Runtime version to use")

# Allow the user to override the installation directory
set(ORT_DIR "${CMAKE_BINARY_DIR}/ORT" CACHE PATH "Directory to install ONNX Runtime")
file(MAKE_DIRECTORY ${ORT_DIR})

# Check if we've already downloaded and extracted ORT
set(ORT_MARKER_FILE "${ORT_DIR}/onnxruntime-v${ORT_VERSION}-installed.marker")
if(EXISTS ${ORT_MARKER_FILE})
  message(STATUS "ONNX Runtime v${ORT_VERSION} is already installed at ${ORT_DIR}")
else()
  # Determine platform and architecture
  if(WIN32)
    set(ORT_PLATFORM "win")
    set(ORT_ARCHIVE_EXT "zip")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(ORT_ARCH "x64")
    else()
      message(FATAL_ERROR "ORT v${ORT_VERSION} only supports x64 on Windows")
    endif()
  elseif(APPLE)
    set(ORT_PLATFORM "osx")
    set(ORT_ARCHIVE_EXT "tgz")
    execute_process(
      COMMAND uname -m
      OUTPUT_VARIABLE ARCH_NAME
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(ARCH_NAME STREQUAL "arm64")
      set(ORT_ARCH "arm64")
    else()
      set(ORT_ARCH "x64")
    endif()
  elseif(UNIX AND NOT APPLE)
    set(ORT_PLATFORM "linux")
    set(ORT_ARCHIVE_EXT "tgz")
    execute_process(
      COMMAND uname -m
      OUTPUT_VARIABLE ARCH_NAME
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(ARCH_NAME STREQUAL "x86_64")
      set(ORT_ARCH "x64")
    elseif(ARCH_NAME MATCHES "aarch64|arm64")
      set(ORT_ARCH "aarch64")
    else()
      message(FATAL_ERROR "Unsupported Linux architecture: ${ARCH_NAME}")
    endif()
  else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime")
  endif()

  # Construct the URL for the appropriate binary
  set(ORT_DOWNLOAD_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_ARCH}-${ORT_VERSION}.${ORT_ARCHIVE_EXT}")

  # Setup download and extraction
  set(ORT_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/onnxruntime-download")
  file(MAKE_DIRECTORY ${ORT_DOWNLOAD_DIR})
  set(ORT_ARCHIVE "${ORT_DOWNLOAD_DIR}/onnxruntime-${ORT_PLATFORM}-${ORT_ARCH}-${ORT_VERSION}.${ORT_ARCHIVE_EXT}")

  # Download the archive
  message(STATUS "Downloading ONNX Runtime from ${ORT_DOWNLOAD_URL}")
  file(DOWNLOAD ${ORT_DOWNLOAD_URL} ${ORT_ARCHIVE} SHOW_PROGRESS STATUS DOWNLOAD_STATUS)
  list(GET DOWNLOAD_STATUS 0 DOWNLOAD_STATUS_CODE)
  if(NOT DOWNLOAD_STATUS_CODE EQUAL 0)
    list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR_MESSAGE)
    message(FATAL_ERROR "Failed to download ONNX Runtime: ${DOWNLOAD_ERROR_MESSAGE}")
  endif()

  # Extract the archive
  message(STATUS "Extracting ONNX Runtime to ${ORT_DIR}")
  file(ARCHIVE_EXTRACT INPUT ${ORT_ARCHIVE} DESTINATION ${ORT_DIR})

  # Create a marker file to indicate successful installation
  file(WRITE ${ORT_MARKER_FILE} "ONNX Runtime v${ORT_VERSION} installed on ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Find the actual directory name inside the archive
file(GLOB ORT_EXTRACTED_DIRS "${ORT_DIR}/onnxruntime-*")
list(LENGTH ORT_EXTRACTED_DIRS ORT_EXTRACTED_DIRS_LENGTH)
if(ORT_EXTRACTED_DIRS_LENGTH EQUAL 0)
  # If no directory with "onnxruntime-" prefix is found, look for other variations
  file(GLOB ORT_EXTRACTED_DIRS "${ORT_DIR}/onnxruntime*")
  list(LENGTH ORT_EXTRACTED_DIRS ORT_EXTRACTED_DIRS_LENGTH)
  if(ORT_EXTRACTED_DIRS_LENGTH EQUAL 0)
    # If still no matches, assume the files were extracted directly into ORT_DIR
    set(ORT_EXTRACTED_DIR ${ORT_DIR})
  else()
    list(GET ORT_EXTRACTED_DIRS 0 ORT_EXTRACTED_DIR)
  endif()
else()
  list(GET ORT_EXTRACTED_DIRS 0 ORT_EXTRACTED_DIR)
endif()

# Set variables for include and library directories
set(ONNXRUNTIME_INCLUDE_DIRS "${ORT_EXTRACTED_DIR}/include" CACHE PATH "ONNX Runtime include directories")

# Set library paths based on platform
if(WIN32)
  set(ONNXRUNTIME_LIB_DIR "${ORT_EXTRACTED_DIR}/lib" CACHE PATH "ONNX Runtime library directory")
  set(ONNXRUNTIME_BIN_DIR "${ORT_EXTRACTED_DIR}/bin" CACHE PATH "ONNX Runtime binary directory")

  # Find the import library
  if(EXISTS "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
    set(ONNXRUNTIME_IMPORT_LIB "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib" CACHE FILEPATH "ONNX Runtime import library")
  else()
    file(GLOB ORT_LIBS "${ONNXRUNTIME_LIB_DIR}/*.lib")
    if(ORT_LIBS)
      list(GET ORT_LIBS 0 ONNXRUNTIME_IMPORT_LIB)
      set(ONNXRUNTIME_IMPORT_LIB "${ONNXRUNTIME_IMPORT_LIB}" CACHE FILEPATH "ONNX Runtime import library" FORCE)
    else()
      message(WARNING "Could not find ONNX Runtime import library (*.lib)")
    endif()
  endif()

  # Find the DLL
  if(EXISTS "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll")
    set(ONNXRUNTIME_SHARED_LIB "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll" CACHE FILEPATH "ONNX Runtime shared library")
  else()
    file(GLOB ORT_DLLS "${ONNXRUNTIME_BIN_DIR}/*.dll")
    if(ORT_DLLS)
      list(GET ORT_DLLS 0 ONNXRUNTIME_SHARED_LIB)
      set(ONNXRUNTIME_SHARED_LIB "${ONNXRUNTIME_SHARED_LIB}" CACHE FILEPATH "ONNX Runtime shared library" FORCE)
    else()
      message(WARNING "Could not find ONNX Runtime shared library (*.dll)")
    endif()
  endif()

  # Set the library to use for linking
  set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_IMPORT_LIB}" CACHE FILEPATH "ONNX Runtime library for linking")

elseif(APPLE)
  set(ONNXRUNTIME_LIB_DIR "${ORT_EXTRACTED_DIR}/lib" CACHE PATH "ONNX Runtime library directory")

  # Find the shared library
  if(EXISTS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib")
    set(ONNXRUNTIME_SHARED_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib" CACHE FILEPATH "ONNX Runtime shared library")
  else()
    file(GLOB ORT_LIBS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime*")
    if(ORT_LIBS)
      list(GET ORT_LIBS 0 ONNXRUNTIME_SHARED_LIB)
      set(ONNXRUNTIME_SHARED_LIB "${ONNXRUNTIME_SHARED_LIB}" CACHE FILEPATH "ONNX Runtime shared library" FORCE)
    else()
      message(WARNING "Could not find ONNX Runtime shared library (libonnxruntime*)")
    endif()
  endif()

  # Set the library to use for linking
  set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_SHARED_LIB}" CACHE FILEPATH "ONNX Runtime library for linking")

else() # Linux
  set(ONNXRUNTIME_LIB_DIR "${ORT_EXTRACTED_DIR}/lib" CACHE PATH "ONNX Runtime library directory")

  # Find the shared library
  if(EXISTS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")
    set(ONNXRUNTIME_SHARED_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so" CACHE FILEPATH "ONNX Runtime shared library")
  else()
    file(GLOB ORT_LIBS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime*")
    if(ORT_LIBS)
      list(GET ORT_LIBS 0 ONNXRUNTIME_SHARED_LIB)
      set(ONNXRUNTIME_SHARED_LIB "${ONNXRUNTIME_SHARED_LIB}" CACHE FILEPATH "ONNX Runtime shared library" FORCE)
    else()
      message(WARNING "Could not find ONNX Runtime shared library (libonnxruntime*)")
    endif()
  endif()

  # Set the library to use for linking
  set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_SHARED_LIB}" CACHE FILEPATH "ONNX Runtime library for linking")
endif()

# Verify that we found the library
if(NOT ONNXRUNTIME_LIBRARIES OR NOT EXISTS "${ONNXRUNTIME_LIBRARIES}")
  message(FATAL_ERROR "Could not find ONNX Runtime library for linking")
endif()

# Add the library to the project
add_library(ONNXRuntime SHARED IMPORTED GLOBAL)
set_target_properties(ONNXRuntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_SHARED_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
)

# For Windows, set IMPORTED_IMPLIB
if(WIN32 AND ONNXRUNTIME_IMPORT_LIB)
  set_property(TARGET ONNXRuntime PROPERTY IMPORTED_IMPLIB "${ONNXRUNTIME_IMPORT_LIB}")
endif()

# Set package found status
set(ONNXRUNTIME_FOUND TRUE)

# Print status
message(STATUS "ONNX Runtime found:")
message(STATUS "  Version: ${ORT_VERSION}")
message(STATUS "  Include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "  Libraries: ${ONNXRUNTIME_LIBRARIES}")
message(STATUS "  Imported target: ONNXRuntime")
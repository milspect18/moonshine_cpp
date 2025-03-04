# Include external dependencies
include(FetchAudioFile)

add_executable(moonshine_transcribe_wav main.cpp)

target_include_directories(moonshine_transcribe_wav PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${audiofile_SOURCE_DIR}
)

target_link_libraries(moonshine_transcribe_wav PRIVATE
    moonshine_cpp
    ${AudioFile_LIBRARIES}
)
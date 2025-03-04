# Fetch AudioFile from GitHub
include(FetchContent)

message(STATUS "Fetching audiofile @ https://github.com/adamstark/AudioFile.git")

set(FETCHCONTENT_QUIET FALSE)
set(AudioFileVersion "1.1.2")

FetchContent_Declare(
    audiofile
    GIT_REPOSITORY  https://github.com/adamstark/AudioFile.git
    GIT_TAG         ${AudioFileVersion}
    GIT_PROGRESS    TRUE
)

FetchContent_MakeAvailable(audiofile)
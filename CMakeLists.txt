cmake_minimum_required(VERSION 4.2)
project(volleyball_cv)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "D:/hygge/opencv/build/")
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "D:/hygge/onnxruntime")

add_executable(volleyball_cv main.cpp
        main.cpp
        video.cpp
        video.h
        court.cpp
        court.h
        calibration.cpp
        calibration.h
        detector.cpp
        detector.h
        track.cpp
        track.h)

target_include_directories(volleyball_cv PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_DIR}/include)

target_link_libraries(volleyball_cv PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)

add_custom_command(TARGET volleyball_cv POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:volleyball_cv>)

add_custom_command(TARGET volleyball_cv POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
"${CMAKE_SOURCE_DIR}/models"
"$<TARGET_FILE_DIR:volleyball_cv>/models")

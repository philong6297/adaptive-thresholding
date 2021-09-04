add_subdirectory(imgproc)

add_executable(main)
target_compile_options(main PRIVATE ${LONGLP_DESIRED_COMPILE_OPTIONS})
target_compile_features(main PRIVATE ${LONGLP_DESIRED_COMPILE_FEATURES})
target_include_directories(main PRIVATE ${LONGLP_PROJECT_SRC_DIR})
target_sources(main PRIVATE main.cpp)
target_link_libraries(main PRIVATE imgproc opencv_highgui)

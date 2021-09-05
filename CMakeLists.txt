cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# ---- Project ----
project(
  image-processing
  VERSION 1.0.0
  LANGUAGES CXX
)

set(LONGLP_PROJECT_DIR ${PROJECT_SOURCE_DIR})
set(LONGLP_PROJECT_SRC_DIR ${LONGLP_PROJECT_DIR}/src)
set(LONGLP_PROJECT_OUTPUT_DIR ${PROJECT_BINARY_DIR})
set(LONGLP_PROJECT_TEST_DIR ${LONGLP_PROJECT_DIR}/test)
set(LONGLP_PROJECT_DOC_DIR ${LONGLP_PROJECT_DIR}/docs)
set(LONGLP_PROJECT_NAME ${PROJECT_NAME})
set(LONGLP_PROJECT_VERSION ${PROJECT_VERSION})
set(LONGLP_PROJECT_CMAKE_DIR ${LONGLP_PROJECT_DIR}/cmake)

# ---- Related tools ----
add_subdirectory(cmake)
set(LONGLP_DESIRED_COMPILE_FEATURES cxx_std_20)
longlp_desired_compile_options(LONGLP_DESIRED_COMPILE_OPTIONS)

# ---- Third parties ----
find_package(fmt REQUIRED)
find_package(benchmark REQUIRED)
find_package(doctest REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
find_package(OpenMP REQUIRED)
find_package(nameof REQUIRED)

add_subdirectory(${LONGLP_PROJECT_SRC_DIR})

# ---- Test ----
# include(CTest) enable_testing() add_subdirectory(${LONGLP_PROJECT_TEST_DIR})

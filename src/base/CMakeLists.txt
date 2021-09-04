add_library(base STATIC)
target_compile_options(base PRIVATE ${LONGLP_DESIRED_COMPILE_OPTIONS})
target_compile_features(base PRIVATE ${LONGLP_DESIRED_COMPILE_FEATURES})
target_include_directories(base PRIVATE ${LONGLP_PROJECT_SRC_DIR})
target_sources(
  base
  PRIVATE # CRTP
          crtp.cpp
          crtp.hpp
          # Type Traits
          type_traits.cpp
          type_traits.hpp
)
target_link_libraries(base PRIVATE opencv_core)

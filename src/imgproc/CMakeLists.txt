add_library(imgproc STATIC)
target_compile_options(imgproc PRIVATE ${LONGLP_DESIRED_COMPILE_OPTIONS})
target_compile_features(imgproc PRIVATE ${LONGLP_DESIRED_COMPILE_FEATURES})
target_include_directories(imgproc PRIVATE ${LONGLP_PROJECT_SRC_DIR})
target_link_libraries(
  imgproc
  PUBLIC # third parties
         opencv_core opencv_imgproc opencv_ximgproc OpenMP::OpenMP_CXX fmt::fmt
         # internal library
)
target_sources(
  imgproc
  PRIVATE imgproc.cpp
          imgproc.cpp
          common/constant.cpp
          common/constant.hpp
          common/integral_image_calculator.cpp
          common/integral_image_calculator.hpp
          binarization/binarization.cpp
          binarization/binarization.hpp
          binarization/bernsen.cpp
          binarization/bernsen.hpp
          binarization/niblack.cpp
          binarization/niblack.hpp
          binarization/sauvola.cpp
          binarization/sauvola.hpp
)

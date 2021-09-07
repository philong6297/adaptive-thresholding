// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "imgproc/imgproc.hpp"

namespace {
  namespace imgproc = longlp::imgproc;

  inline auto show(const std::string_view name, const cv::Mat& img) {
    cv::namedWindow(name.data(), cv::WindowFlags::WINDOW_KEEPRATIO);
    cv::imshow(name.data(), img);
  }

  template <longlp::imgproc::BinarizationMethodInterface T>
  inline auto test(const cv::Mat& input) {
    imgproc::BinarizationAlgorithm<T> algo{};
    cv::Mat output;
    algo.Binarize(input, output, true);
    show(algo.name(), output);
  }
}   // namespace

auto main() -> int32_t {
  const auto input =
    cv::imread("C:/tools/adaptive-thresholding/data/input/2.png",
               cv::ImreadModes::IMREAD_GRAYSCALE);
  show("input", input);

  test<imgproc::Bernsen>(input);
  test<imgproc::NiBlack>(input);
  test<imgproc::Sauvola>(input);
  test<imgproc::Otsu2D>(input);

  // cv::Mat guided_image;
  // // guided image as average image
  // cv::blur(input,
  //          guided_image,
  //          cv::Size(75, 75),    // kernel size
  //          cv::Point(-1, -1),   // anchor at kernel center
  //          cv::BORDER_REFLECT   // symmetric padding
  // );
  // test<improc::Otsu2D>(input,
  //                      {
  //                        guided_image,                        // guided
  //                        image cv::ThresholdTypes::THRESH_BINARY,   //
  //                        threshold type false,   // edge is foreground
  //                        true,    // noise is background
  //                      });

  cv::waitKeyEx(0);
  return 0;
}

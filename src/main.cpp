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

  template <longlp::imgproc::BinarizationInterface T>
  inline auto test(const cv::Mat& input) {
    imgproc::BinarizationAlgorithm<T> algo{};
    cv::Mat output;
    algo.Binarize(input, output, true);
    show(algo.getDefaultName().data(), output);
  }
}   // namespace

auto main() -> int32_t {
  const auto input =
    cv::imread("C:/tools/adaptive-thresholding/data/input/2.png",
               cv::ImreadModes::IMREAD_GRAYSCALE);
  show("input", input);

  // test<improc::Bernsen>(
  //   input,
  //   {
  //     25,   // contrast limit
  //     cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
  //                               cv::Size(75, 75)),   // kernel
  //     100,                                           // global threshold
  //     cv::ThresholdTypes::THRESH_BINARY              // threshold type
  //   });
  test<imgproc::Bernsen>(input);

  // test<improc::OtsuOpenCV>(
  //   input,
  //   {
  //     cv::ThresholdTypes::THRESH_BINARY   // threshold type
  //   });

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
  //                        guided_image,                        // guided image
  //                        cv::ThresholdTypes::THRESH_BINARY,   // threshold
  //                        type false,   // edge is foreground true,    //
  //                        noise is background
  //                      });

  // test<improc::NiBlackOpenCV>(
  //   input,
  //   {
  //     75,                                 // block size
  //     -0.2,                               // k
  //     cv::ThresholdTypes::THRESH_BINARY   // threshold type
  //   });
  // test<improc::NiBlackIntegralImage>(
  //   input,
  //   {
  //     75,                                 // rectangle kernel width
  //     75,                                 // rectangle kernel height
  //     -0.2,                               // k
  //     cv::ThresholdTypes::THRESH_BINARY   // threshold type
  //   });
  // test<improc::SauvolaOpenCV>(
  //   input,
  //   {
  //     75,                                 // block size
  //     0.2,                                // k
  //     128,                                // r
  //     cv::ThresholdTypes::THRESH_BINARY   // threshold type
  //   });
  // test<improc::SauvolaIntegralImage>(
  //   input,
  //   {
  //     75,                                 // rectangle kernel width
  //     75,                                 // rectangle kernel height
  //     0.2,                                // k
  //     128,                                // r
  //     cv::ThresholdTypes::THRESH_BINARY   // threshold type
  //   });

  cv::waitKeyEx(0);
  return 0;
}

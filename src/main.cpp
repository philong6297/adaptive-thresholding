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
  inline auto test(
    const cv::Mat& input,
    const typename imgproc::BinarizationAlgorithm<T>::Params& params) {
    imgproc::BinarizationAlgorithm<T> algo{};
    cv::Mat output;
    algo.Binarize(input, output, true, params);
    show(algo.name(), output);
  }
}   // namespace

auto main() -> int32_t {
  const auto input =
    cv::imread("C:/tools/adaptive-thresholding/data/input/2.png",
               cv::ImreadModes::IMREAD_GRAYSCALE);
  show("input", input);

  test<imgproc::Bernsen>(
    input,
    {25.0 /* constrast limit */,
     100.0 /* global threshold */,
     cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
                               cv::Size(75, 75))} /* kernel */);

  test<imgproc::NiBlack>(input,
                         {cv::Size{75, 75} /* kernel size */, -0.2 /* k */});

  test<imgproc::Sauvola>(
    input,
    {cv::Size{75, 75} /* kernel size */, 0.2 /* k */, 128.0 /* r */});

  test<imgproc::Otsu2D>(input,
                        {
                          cv::Size{75, 75} /* kernel size */,
                          false /* edge is foreground */,
                          true /* noise is background */
                        });

  cv::waitKeyEx(0);
  return 0;
}

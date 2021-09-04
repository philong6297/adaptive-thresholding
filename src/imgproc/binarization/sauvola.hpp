// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_SAUVOLA_HPP_
#define IMGPROC_BINARIZATION_SAUVOLA_HPP_

#include <opencv2/imgproc.hpp>

namespace longlp::imgproc {
  class Sauvola final {
   public:
    auto BinarizeImpl(const cv::Mat& input,
                      cv::Mat& output,
                      bool use_background_white_color) const -> void;
    auto AdditionalInputConstraints(
      [[maybe_unused]] const cv::Mat& input) const noexcept {}
    auto AdditionalOutputConstraints(
      [[maybe_unused]] const cv::Mat& input,
      [[maybe_unused]] const cv::Mat& output) const noexcept {}

   private:
    cv::Size kernel_size_{75, 75};
    double k_{0.2};
    double r_{128.0};
  };

}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_SAUVOLA_HPP_

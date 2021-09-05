// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_OTSU_HPP_
#define IMGPROC_BINARIZATION_OTSU_HPP_

#include <opencv2/core.hpp>

namespace longlp::imgproc {

  class Otsu2D final {
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
    bool edge_role_as_background_{false};   // treat detected edge as either
                                            // background or foreground
    bool noise_role_as_background_{true};   // treat detected noise as either
                                            // background or foreground
  };
}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_OTSU_HPP_

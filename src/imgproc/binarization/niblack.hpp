// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_NIBLACK_HPP_
#define IMGPROC_BINARIZATION_NIBLACK_HPP_

#include <opencv2/core.hpp>

namespace longlp::imgproc {
  class NiBlack final {
   public:
    struct Params {
      // size area must be > 0
      cv::Size kernel_size{};
      double k{};
    };

    auto BinarizeUnsafe(const cv::Mat& input,
                        cv::Mat& output,
                        bool use_background_white_color,
                        const Params& params) const -> void;

    auto InvalidateParams(const cv::Mat& input, const Params& params) const
      -> void;
  };

}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_NIBLACK_HPP_

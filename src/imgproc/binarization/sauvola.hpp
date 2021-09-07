// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_SAUVOLA_HPP_
#define IMGPROC_BINARIZATION_SAUVOLA_HPP_

#include <opencv2/core.hpp>

namespace longlp::imgproc {
  class Sauvola final {
   public:
    struct Params {
      // size area must be > 0
      cv::Size kernel_size{};

      double k{};

      // must be in range [0.0 - 255.0]
      double r{};
    };
    auto BinarizeUnsafe(const cv::Mat& input,
                        cv::Mat& output,
                        bool use_background_white_color,
                        const Params& params) const -> void;

    auto InvalidateParams(const Params& params) const -> void;
  };

}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_SAUVOLA_HPP_

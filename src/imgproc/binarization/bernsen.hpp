// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_BERNSEN_HPP_
#define IMGPROC_BINARIZATION_BERNSEN_HPP_

#include <opencv2/imgproc.hpp>

namespace longlp::imgproc {

  class Bernsen final {
   public:
    struct Params {
      // value must be in range [0.0 - 255.0]
      double contrast_limit{};

      // value must be in range [0.0 - 255.0]
      double global_threshold{};

      // usually is created with cv::getStructuringElement
      cv::Mat kernel{};
    };

    auto BinarizeUnsafe(const cv::Mat& input,
                        cv::Mat& output,
                        bool use_background_white_color,
                        const Params& params) const -> void;

    auto ValidateParams(const Params& params) const -> void;
  };
}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_BERNSEN_HPP_

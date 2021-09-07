// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_OTSU_HPP_
#define IMGPROC_BINARIZATION_OTSU_HPP_

#include <opencv2/core.hpp>

namespace longlp::imgproc {

  class Otsu2D final {
   public:
    struct Params {
      cv::Size kernel_size{};
      bool edge_role_as_background{};    // treat detected edge as either
                                         // background or foreground
      bool noise_role_as_background{};   // treat detected noise as either
                                         // background or foreground

      cv::Mat
        guided_image{};   // usually created with cv::blur to get average image
    };

    auto BinarizeUnsafe(const cv::Mat& input,
                        cv::Mat& output,
                        bool use_background_white_color,
                        const Params& params) const -> void;

    auto InvalidateParams(const cv::Mat& input, const Params& params) const
      -> void;
  };
}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_OTSU_HPP_

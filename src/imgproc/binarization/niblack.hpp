// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_NIBLACK_HPP_
#define IMGPROC_BINARIZATION_NIBLACK_HPP_

#include <opencv2/core.hpp>
#include <opencv2/core/softfloat.hpp>

namespace longlp::imgproc {
  class NiBlack final {
   public:
    auto BinarizeImpl(const cv::Mat& input,
                      cv::Mat& output,
                      bool use_background_white_color) const -> void;

   private:
    cv::Size kernel_size_{75, 75};
    cv::softdouble k_{-0.2};
  };

}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_NIBLACK_HPP_

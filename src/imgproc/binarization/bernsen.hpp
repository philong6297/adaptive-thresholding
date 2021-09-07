// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_BERNSEN_HPP_
#define IMGPROC_BINARIZATION_BERNSEN_HPP_

#include <opencv2/core/softfloat.hpp>
#include <opencv2/imgproc.hpp>

namespace longlp::imgproc {

  class Bernsen final {
   public:
    auto BinarizeImpl(const cv::Mat& input,
                      cv::Mat& output,
                      bool use_background_white_color) const -> void;

   private:
    // value must be in range [0.0 - 255.0]
    cv::softdouble contrast_limit_{25.0};

    cv::Mat kernel_{cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
                                              cv::Size(75, 75))};

    // value must be in range [0.0 - 255.0]
    cv::softdouble global_threshold_{100.0};
  };
}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_BERNSEN_HPP_

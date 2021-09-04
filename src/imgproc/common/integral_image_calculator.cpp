// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#include "imgproc/common/integral_image_calculator.hpp"

namespace {
  using longlp::imgproc::IntegralImageCalculator;
}   // namespace

// static
// The border type is referenced in
// https://www.mathworks.com/help/images/ref/stdfilt.html
auto IntegralImageCalculator::MakePaddedInputForIntegral(
  const cv::Mat& input,
  const int top_padding_size,
  const int bottom_padding_size,
  const int left_padding_size,
  const int right_padding_size) noexcept -> cv::Mat {
  // pre-conditions
  CV_Assert((top_padding_size >= 0 && bottom_padding_size >= 0 &&
             left_padding_size >= 0 && right_padding_size >= 0));

  // Create padding with kernel
  cv::Mat padded_input = input.clone();
  cv::copyMakeBorder(padded_input,
                     padded_input,
                     top_padding_size,
                     bottom_padding_size,
                     left_padding_size,
                     right_padding_size,
                     cv::BorderTypes::BORDER_REFLECT,   // symmetric padding
                     cv::Scalar()   // no-op when using BORDER_REFLECT
  );

  // post-conditions
  CV_Assert(padded_input.type() == input.type() &&
            padded_input.size() ==
              cv::Size(input.cols + left_padding_size + right_padding_size,
                       input.rows + top_padding_size + bottom_padding_size));
  return padded_input;
}

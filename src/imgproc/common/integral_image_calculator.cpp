// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#include "imgproc/common/integral_image_calculator.hpp"

namespace {
  using longlp::imgproc::IntegralImageCalculator;

  using ErrorCode = cv::Error::Code;
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
  if (top_padding_size < 0 || bottom_padding_size < 0 ||
      left_padding_size < 0 || right_padding_size < 0) {
    CV_Error(ErrorCode::StsBadArg, "padding size must be positive integer");
  }

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
  if (padded_input.type() != input.type() || padded_input.dims != input.dims ||
      padded_input.size() !=
        cv::Size(input.cols + left_padding_size + right_padding_size,
                 input.rows + top_padding_size + bottom_padding_size)) {
    CV_Error(ErrorCode::StsInternal,
             "padded image neither have same type and dims as input image nor "
             "size is not Size(src.cols+left+right, src.rows+top+bottom)");
  }
  return padded_input;
}

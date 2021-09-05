// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#include "imgproc/binarization/bernsen.hpp"

#include "imgproc/common/constant.hpp"
#include "imgproc/common/integral_image_calculator.hpp"

namespace {
  using longlp::imgproc::Bernsen;
  using longlp::imgproc::BinaryColorPair;
  using longlp::imgproc::IntegralImageCalculator;
  using KernelVertices = IntegralImageCalculator::KernelVertices;
  using IntegralImages = IntegralImageCalculator::IntegralImages<1>;

}   // namespace

// https://www.academia.edu/30363617/Implementation_of_Bernsen_s_Locally_Adaptive_Binarization_Method_for_Gray_Scale_Images
auto Bernsen::BinarizeImpl(const cv::Mat& input,
                           cv::Mat& output,
                           const bool use_background_white_color) const
  -> void {
  output = input.clone();
  output.convertTo(output, CV_64F);

  const auto binary_colors = use_background_white_color
                               ? BinaryColorPair::Get()
                               : BinaryColorPair::GetInverse();

  // Image contains max values based on neighbor pixels, which are constructed
  // by kernel
  cv::Mat min_filter;
  cv::erode(
    output,
    min_filter,
    kernel_,
    /* anchor, at kernel center */ cv::Point{-1, -1},
    /* iterations */ 1,
    /* border type */ cv::BorderTypes::BORDER_CONSTANT,
    /* use default constant value */ cv::morphologyDefaultBorderValue());

  // Image contains min values based on neighbor pixels, which are constructed
  // by kernel
  cv::Mat max_filter;
  cv::dilate(
    output,
    max_filter,
    kernel_,
    /* anchor, at kernel center */ cv::Point{-1, -1},
    /* iterations */ 1,
    /* border type */ cv::BorderTypes::BORDER_CONSTANT,
    /* use default constant value */ cv::morphologyDefaultBorderValue());

  IntegralImageCalculator::ConstructIntegralAndIterate<double, 1>(
    output,
    kernel_.size(),
    [this, &binary_colors, &min_filter, &max_filter](
      double& pixel,
      const int* position,
      const IntegralImages& integral_images,
      const KernelVertices& kernel_vertices) {
      const auto& integral_1st_order = integral_images[0];

      const auto y = position[0];
      const auto x = position[1];

      const auto min = *min_filter.ptr<double>(y, x);
      const auto max = *max_filter.ptr<double>(y, x);

      const auto local_contrast = max - min;

      const auto i1i1 = *integral_1st_order.ptr<double>(kernel_vertices.bottom,
                                                        kernel_vertices.right);
      const auto i1i2 = *integral_1st_order.ptr<double>(kernel_vertices.top,
                                                        kernel_vertices.left);
      const auto i1i3 = *integral_1st_order.ptr<double>(kernel_vertices.bottom,
                                                        kernel_vertices.left);
      const auto i1i4 = *integral_1st_order.ptr<double>(kernel_vertices.top,
                                                        kernel_vertices.right);

      const auto N = kernel_.total();

      const auto s1   = i1i1 + i1i2 - i1i3 - i1i4;
      const auto mean = s1 * 1.0 / static_cast<double>(N);

      if (local_contrast < contrast_limit_) {
        pixel = (mean < global_threshold_) ? binary_colors.object
                                           : binary_colors.background;
      }
      else {
        pixel =
          (pixel < mean) ? binary_colors.object : binary_colors.background;
      }
    });

  output.convertTo(output, CV_8U);
}

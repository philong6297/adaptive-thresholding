// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#include "imgproc/binarization/bernsen.hpp"

#include <opencv2/core/softfloat.hpp>

#include "imgproc/common/constant.hpp"
#include "imgproc/common/integral_image_calculator.hpp"

namespace {
  using longlp::imgproc::Bernsen;
  using longlp::imgproc::BinaryColorPair;
  using longlp::imgproc::IntegralImageCalculator;
  using KernelVertices = IntegralImageCalculator::KernelVertices;
  using IntegralImages = IntegralImageCalculator::IntegralImages<1>;

  using ErrorCode = cv::Error::Code;
  using cv::softdouble;

}   // namespace

auto Bernsen::ValidateParams([[maybe_unused]] const cv::Mat& input,
                             const Params& params) const -> void {
  if (const softdouble ct{params.contrast_limit};
      !(ct >= softdouble::zero() && ct <= softdouble{255.0})) {
    CV_Error(ErrorCode::StsBadArg, "contrast limit is not in range [0-255]");
  }

  if (const softdouble gt{params.global_threshold};
      !(gt >= softdouble::zero() && gt <= softdouble{255.0})) {
    CV_Error(ErrorCode::StsBadArg, "contrast limit is not in range [0-255]");
  }

  if (params.kernel.empty() || params.kernel.dims != 2) {
    CV_Error(ErrorCode::StsBadArg,
             "kernel either is empty or has invalid dims (!= 2)");
  }
}

// https://www.academia.edu/30363617/Implementation_of_Bernsen_s_Locally_Adaptive_Binarization_Method_for_Gray_Scale_Images
auto Bernsen::BinarizeUnsafe(const cv::Mat& input,
                             cv::Mat& output,
                             const bool use_background_white_color,
                             const Params& params) const -> void {
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
    params.kernel,
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
    params.kernel,
    /* anchor, at kernel center */ cv::Point{-1, -1},
    /* iterations */ 1,
    /* border type */ cv::BorderTypes::BORDER_CONSTANT,
    /* use default constant value */ cv::morphologyDefaultBorderValue());

  IntegralImageCalculator::ConstructIntegralAndIterate<double, 1>(
    output,
    params.kernel.size(),
    [&binary_colors,
     &min_filter,
     &max_filter,
     N  = softdouble{params.kernel.total()},
     gt = softdouble{params.global_threshold},
     ct = softdouble{params.contrast_limit}](
      double& pixel,
      const int* position,
      const IntegralImages& integral_images,
      const KernelVertices& kernel_vertices) {
      const auto& integral_1st_order = integral_images[0];

      const auto y = position[0];
      const auto x = position[1];

      const softdouble min{*min_filter.ptr<double>(y, x)};
      const softdouble max{*max_filter.ptr<double>(y, x)};

      const auto local_contrast = max - min;

      const softdouble i1i1{
        *integral_1st_order.ptr<double>(kernel_vertices.bottom,
                                        kernel_vertices.right)};
      const softdouble i1i2{
        *integral_1st_order.ptr<double>(kernel_vertices.top,
                                        kernel_vertices.left)};
      const softdouble i1i3{
        *integral_1st_order.ptr<double>(kernel_vertices.bottom,
                                        kernel_vertices.left)};
      const softdouble i1i4{
        *integral_1st_order.ptr<double>(kernel_vertices.top,
                                        kernel_vertices.right)};

      const auto mean = (i1i1 + i1i2 - i1i3 - i1i4) / N;

      if (local_contrast < ct) {
        pixel = mean < gt ? binary_colors.object : binary_colors.background;
      }
      else {
        pixel = softdouble{pixel} < mean ? binary_colors.object
                                         : binary_colors.background;
      }
    });

  output.convertTo(output, CV_8U);
}

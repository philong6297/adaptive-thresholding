// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#include "imgproc/binarization/sauvola.hpp"

#include <opencv2/core/softfloat.hpp>

#include "imgproc/common/constant.hpp"
#include "imgproc/common/integral_image_calculator.hpp"

namespace {
  using longlp::imgproc::IntegralImageCalculator;
  using longlp::imgproc::Sauvola;
  using KernelVertices = IntegralImageCalculator::KernelVertices;
  using IntegralImages = IntegralImageCalculator::IntegralImages<2>;

  using cv::softdouble;
  using ErrorCode = cv::Error::Code;
}   // namespace

auto Sauvola::InvalidateParams([[maybe_unused]] const cv::Mat& input,
                               const Params& params) const -> void {
  if (params.kernel_size.empty()) {
    CV_Error(ErrorCode::StsBadArg, "kernel size is empty");
  }

  if (const softdouble r{params.r};
      !(r >= softdouble::zero() && r <= softdouble{255.0})) {
    CV_Error(ErrorCode::StsBadArg,
             "dynamic range of standard deviation(r) is not in range [0-255]");
  }
}

// https://sci-hub.se/https://doi.org/10.1016/S0031-3203(99)00055-2
void Sauvola::BinarizeUnsafe(const cv::Mat& input,
                             cv::Mat& output,
                             const bool use_background_white_color,
                             const Params& params) const {
  output = input.clone();
  output.convertTo(output, CV_64F);

  const auto binary_colors = use_background_white_color
                               ? BinaryColorPair::Get()
                               : BinaryColorPair::GetInverse();

  IntegralImageCalculator::ConstructIntegralAndIterate<double, 2>(
    output,
    params.kernel_size,
    [&binary_colors,
     N = softdouble{params.kernel_size.area()},
     k = softdouble{params.k},
     r = softdouble{params.r}](double& pixel,
                               [[maybe_unused]] const int* position,
                               const IntegralImages& integral_images,
                               const KernelVertices& kernel_vertices) {
      const auto& [integral_1st_order, integral_2nd_order] = integral_images;

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

      const auto local_mean = (i1i1 + i1i2 - i1i3 - i1i4) / N;

      const softdouble i2i1{
        *integral_2nd_order.ptr<double>(kernel_vertices.bottom,
                                        kernel_vertices.right)};
      const softdouble i2i2{
        *integral_2nd_order.ptr<double>(kernel_vertices.top,
                                        kernel_vertices.left)};
      const softdouble i2i3{
        *integral_2nd_order.ptr<double>(kernel_vertices.bottom,
                                        kernel_vertices.left)};
      const softdouble i2i4{
        *integral_2nd_order.ptr<double>(kernel_vertices.top,
                                        kernel_vertices.right)};

      const auto local_stddev =
        cv::sqrt((i2i1 + i2i2 - i2i3 - i2i4) / N - local_mean * local_mean);

      const auto thresh_hold =
        local_mean *
        (softdouble::one() + k * (local_stddev / r - softdouble::one()));

      pixel = softdouble{pixel} > thresh_hold ? binary_colors.background
                                              : binary_colors.object;
    });

  output.convertTo(output, CV_8U);
}

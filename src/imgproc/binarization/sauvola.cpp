// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#include "imgproc/binarization/sauvola.hpp"

#include <utility>
#include <vector>

#include "imgproc/common/constant.hpp"
#include "imgproc/common/integral_image_calculator.hpp"

namespace {
  using longlp::imgproc::IntegralImageCalculator;
  using longlp::imgproc::Sauvola;
  using KernelVertices = IntegralImageCalculator::KernelVertices;
  using IntegralImages = IntegralImageCalculator::IntegralImages<2>;
}   // namespace

// https://sci-hub.se/https://doi.org/10.1016/S0031-3203(99)00055-2
void Sauvola::BinarizeImpl(const cv::Mat& input,
                           cv::Mat& output,
                           const bool use_background_white_color) const {
  output = input.clone();
  output.convertTo(output, CV_64F);

  const auto binary_colors = use_background_white_color
                               ? BinaryColorPair::Get()
                               : BinaryColorPair::GetInverse();

  IntegralImageCalculator::ConstructIntegralAndIterate<double, 2>(
    output,
    kernel_size_,
    [this, &binary_colors](double& pixel,
                           [[maybe_unused]] const int* position,
                           const IntegralImages& integral_images,
                           const KernelVertices& kernel_vertices) {
      const auto& [integral_1st_order, integral_2nd_order] = integral_images;

      const auto i1i1 = *integral_1st_order.ptr<double>(kernel_vertices.bottom,
                                                        kernel_vertices.right);
      const auto i1i2 = *integral_1st_order.ptr<double>(kernel_vertices.top,
                                                        kernel_vertices.left);
      const auto i1i3 = *integral_1st_order.ptr<double>(kernel_vertices.bottom,
                                                        kernel_vertices.left);
      const auto i1i4 = *integral_1st_order.ptr<double>(kernel_vertices.top,
                                                        kernel_vertices.right);

      const auto N = kernel_size_.area();

      const auto s1         = i1i1 + i1i2 - i1i3 - i1i4;
      const auto local_mean = s1 * 1.0 / static_cast<double>(N);

      const auto i2i1 = *integral_2nd_order.ptr<double>(kernel_vertices.bottom,
                                                        kernel_vertices.right);
      const auto i2i2 = *integral_2nd_order.ptr<double>(kernel_vertices.top,
                                                        kernel_vertices.left);
      const auto i2i3 = *integral_2nd_order.ptr<double>(kernel_vertices.bottom,
                                                        kernel_vertices.left);
      const auto i2i4 = *integral_2nd_order.ptr<double>(kernel_vertices.top,
                                                        kernel_vertices.right);

      const auto s2 = i2i1 + i2i2 - i2i3 - i2i4;
      const auto local_stddev =
        std::sqrt(s2 * 1.0 / static_cast<double>(N) - local_mean * local_mean);

      const auto thresh_hold =
        local_mean * (1.0 + k_ * (local_stddev / r_ - 1.0));

      pixel =
        pixel > thresh_hold ? binary_colors.background : binary_colors.object;
    });

  output.convertTo(output, CV_8U);
}

// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_COMMON_INTEGRAL_IMAGE_CALCULATOR_HPP_
#define IMGPROC_COMMON_INTEGRAL_IMAGE_CALCULATOR_HPP_

#include <array>   // IntegralImages
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>   // std::as_const

#include <opencv2/imgproc.hpp>

namespace longlp::imgproc {
  class IntegralImageCalculator {
   public:
    struct KernelVertices {
      int top;
      int bottom;
      int left;
      int right;
    };

    template <size_t Order>
    using IntegralImages = std::array<cv::Mat, Order>;

    // https://en.wikipedia.org/wiki/Summed-area_table
    template <class PixelType, size_t Order, class Processor>
    requires requires {
      requires std::is_same_v<PixelType, uint8_t> ||
        std::is_same_v<PixelType, double> || std::is_same_v<PixelType, float>;

      requires requires(Processor && processor,
                        PixelType & pixel,
                        const int* position,
                        const IntegralImages<Order>& integral_images,
                        const KernelVertices& kernel_vertices) {
        {
          processor(pixel, position, integral_images, kernel_vertices)
          } -> std::same_as<void>;
      };
    }
    static void ConstructIntegralAndIterate(cv::Mat& input_output,
                                            const cv::Size& kernel_size,
                                            Processor&& processor) noexcept {
      // pre-conditions
      if (kernel_size.empty()) {
        CV_Error(cv::Error::Code::StsBadArg, "kernel size is empty");
      }
      if (!(input_output.dims == 2 &&
            (input_output.type() == CV_8U || input_output.type() == CV_32F ||
             input_output.type() == CV_64F))) {
        CV_Error(cv::Error::Code::StsBadArg,
                 "input_output must be 2D image, 8-bit or floating point "
                 "(32-bit, 64-bit)");
      }

      const auto delta_x = (kernel_size.width - 1) / 2;
      const auto delta_y = (kernel_size.height - 1) / 2;

      const cv::Mat& padded_input =
        MakePaddedInputForIntegral(input_output,
                                   delta_y /* top */,
                                   delta_y /* bottom */,
                                   delta_x /* left */,
                                   delta_x /* right */);

      const auto integral_images = MakeIntegralImage<Order>(padded_input);

      input_output.forEach<PixelType>(
        [&delta_x, &delta_y, &integral_images, &processor](
          PixelType& pixel,
          const int* position) {
          // map index from input to integral matrices
          const auto iy = position[0] + delta_y + 1;
          const auto ix = position[1] + delta_x + 1;

          const auto top    = iy - delta_y;
          const auto bottom = iy + delta_y;
          const auto left   = ix - delta_x;
          const auto right  = ix + delta_x;

          std::invoke(processor,
                      pixel,
                      position,
                      std::as_const(integral_images),
                      KernelVertices{top, bottom, left, right});
        });
    }

   private:
    static auto MakePaddedInputForIntegral(const cv::Mat& input,
                                           int top_padding_size,
                                           int bottom_padding_size,
                                           int left_padding_size,
                                           int right_padding_size) noexcept
      -> cv::Mat;

    template <size_t Order>
    static auto MakeIntegralImage(const cv::Mat& input) noexcept
      -> IntegralImages<Order>;
  };

  // static
  template <>
  inline auto IntegralImageCalculator::MakeIntegralImage<1>(
    const cv::Mat& padded_input) noexcept -> IntegralImages<1> {
    // Calculate integral image 1st
    cv::Mat integral_1st_order;
    cv::integral(padded_input,
                 integral_1st_order /* sum */,
                 CV_64F /* force to store double in sum */);
    return {integral_1st_order};
  }

  // static
  template <>
  inline auto IntegralImageCalculator::MakeIntegralImage<2>(
    const cv::Mat& padded_input) noexcept -> IntegralImages<2> {
    // Calculate integral image 1st and 2nd Order
    cv::Mat integral_1st_order;
    cv::Mat integral_2nd_order;
    cv::integral(padded_input,
                 integral_1st_order /* sum */,
                 integral_2nd_order /* square sum */,
                 CV_64F /* force to store double in sum */,
                 CV_64F /* force to store double in square sum */);
    return {integral_1st_order, integral_2nd_order};
  }
}   // namespace longlp::imgproc

#endif   // IMGPROC_COMMON_INTEGRAL_IMAGE_CALCULATOR_HPP_

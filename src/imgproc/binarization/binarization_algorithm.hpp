// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_BINARIZATION_ALGORITHM_HPP_
#define IMGPROC_BINARIZATION_BINARIZATION_ALGORITHM_HPP_

#include <concepts>
#include <memory>   // method_
#include <type_traits>

#include <fmt/format.h>   // name
#include <nameof.hpp>     // name
#include <opencv2/core.hpp>

#include "imgproc/binarization/binarization_validator.hpp"

namespace longlp::imgproc {

  template <class T>
  concept BinarizationMethodInterface = requires {
    requires std::is_class_v<T> && std::semiregular<T>;

    requires std::is_class_v<typename T::Params> &&
      std::semiregular<typename T::Params>;

    // function requirements
    requires requires(const T& t,
                      const cv::Mat& input,
                      cv::Mat& output,
                      const bool use_background_white_color,
                      const typename T::Params& params) {
      {
        t.BinarizeUnsafe(input, output, use_background_white_color, params)
        } -> std::same_as<void>;
    };
  };

  template <BinarizationMethodInterface MethodType>
  class BinarizationAlgorithm {
   public:
    using Params = typename MethodType::Params;

    void Binarize(const cv::Mat& input,
                  cv::Mat& output,
                  const bool use_background_white_color,
                  const Params& params) const {
      // pre-conditions
      // NOLINTNEXTLINE(hicpp-signed-bitwise)
      CV_Assert(input.type() == CV_8UC1 && input.dims == 2);
      BinarizationValidator<MethodType>::ValidateInput(*method_, input);

      BinarizationValidator<MethodType>::ValidateParams(*method_,
                                                        input,
                                                        params);

      method_->BinarizeUnsafe(input,
                              output,
                              use_background_white_color,
                              params);

      // post-conditions
      CV_Assert(output.type() == input.type() && output.dims == input.dims &&
                input.size() == output.size());
      BinarizationValidator<MethodType>::ValidateOutput(*method_,
                                                        input,
                                                        output);
    }

    [[nodiscard]] auto name() const noexcept -> std::string {
      return fmt::format(
        "{algo}_{impl}",
        fmt::arg("algo", nameof::nameof_short_type<BinarizationAlgorithm>()),
        fmt::arg("impl", nameof::nameof_short_type<MethodType>()));
    }

   private:
    std::unique_ptr<MethodType> method_ = std::make_unique<MethodType>();
  };

}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_BINARIZATION_ALGORITHM_HPP_

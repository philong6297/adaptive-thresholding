// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_BINARIZATION_VALIDATOR_HPP_
#define IMGPROC_BINARIZATION_BINARIZATION_VALIDATOR_HPP_

#include <concepts>
#include <type_traits>

#include <opencv2/core.hpp>

namespace longlp::imgproc {

  template <class T>
  requires std::is_class_v<T> && std::semiregular<T>
  struct BinarizationValidator {
    static auto ValidateInput(const T& t, const cv::Mat& input) -> void {
      static constexpr auto has_input_validator = requires {
        { t.ValidateInput(input) } -> std::same_as<void>;
      };

      if constexpr (has_input_validator) {
        t.ValidateInput(input);
      }
    }

    static auto ValidateOutput(const T& t,
                               const cv::Mat& input,
                               const cv::Mat& output) -> void {
      static constexpr auto has_output_validator = requires {
        { t.ValidateInput(input, output) } -> std::same_as<void>;
      };

      if constexpr (has_output_validator) {
        t.ValidateOutput(input, output);
      }
    }
  };
}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_BINARIZATION_VALIDATOR_HPP_

// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_BINARIZATION_BINARIZATION_BASE_HPP_
#define IMGPROC_BINARIZATION_BINARIZATION_BASE_HPP_

#include <memory>
#include <nameof.hpp>
#include <opencv2/core.hpp>
#include <type_traits>

namespace longlp::imgproc {

  template <class T>
  concept BinarizationInterface = requires {
    requires std::is_class_v<T>;
    requires std::semiregular<T>;

    // function requirements
    requires requires(const T& t) {
      {
        t.BinarizeImpl(std::declval<const cv::Mat&>(),   // input
                       std::declval<cv::Mat&>(),         // output
                       std::declval<bool>()   // use_background_white_color
        )
        } -> std::same_as<void>;

      {
        t.AdditionalInputConstraints(std::declval<const cv::Mat&>()   // input
        )
        } -> std::same_as<void>;

      {
        t.AdditionalOutputConstraints(std::declval<const cv::Mat&>(),   // input
                                      std::declval<const cv::Mat&>()   // output
        )
        } -> std::same_as<void>;
    };
  };

  template <BinarizationInterface T>
  class BinarizationAlgorithm : public cv::Algorithm {
   public:
    // cv::Algorithm overrides
    [[nodiscard]] auto getDefaultName() const noexcept -> cv::String final {
      return cv::String{nameof::nameof_short_type<BinarizationAlgorithm>()} +
             cv::String{nameof::nameof_short_type<T>()};
    }

    void Binarize(const cv::Mat& input,
                  cv::Mat& output,
                  const bool use_background_white_color) const {
      // pre-conditions
      // NOLINTNEXTLINE(hicpp-signed-bitwise)
      CV_Assert(input.type() == CV_8UC1 && input.dims == 2);
      method_->AdditionalInputConstraints(input);

      method_->BinarizeImpl(input, output, use_background_white_color);

      // post-conditions
      CV_Assert(output.type() == input.type() && output.dims == input.dims &&
                input.size() == output.size());
      method_->AdditionalOutputConstraints(input, output);
    }

   private:
    static const cv::String name_;
    std::unique_ptr<T> method_ = std::make_unique<T>();
  };

}   // namespace longlp::imgproc

#endif   // IMGPROC_BINARIZATION_BINARIZATION_BASE_HPP_

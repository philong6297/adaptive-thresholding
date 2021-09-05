// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef IMGPROC_CONSTANT_HPP_
#define IMGPROC_CONSTANT_HPP_

#include <cstdint>
namespace longlp::imgproc {
  using GrayscalePixel = uint8_t;

  constexpr GrayscalePixel kGrayscaleBlack{0};
  constexpr GrayscalePixel kGrayscaleWhite{255};
  constexpr auto kGrayscaleMin{kGrayscaleBlack};
  constexpr auto kGrayscaleMax{kGrayscaleWhite};

  struct BinaryColorPair {
    GrayscalePixel object;
    GrayscalePixel background;

    static inline constexpr auto Get() noexcept -> BinaryColorPair {
      return {kGrayscaleBlack, kGrayscaleWhite};
    }
    static inline constexpr auto GetInverse() noexcept -> BinaryColorPair {
      return {kGrayscaleWhite, kGrayscaleBlack};
    }
  };

}   // namespace longlp::imgproc

#endif   // IMGPROC_CONSTANT_HPP_

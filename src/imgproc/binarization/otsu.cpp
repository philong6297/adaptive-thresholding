// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#include "imgproc/binarization/otsu.hpp"

#include <opencv2/core/softfloat.hpp>
#include <opencv2/imgproc.hpp>

#include "imgproc/common/constant.hpp"

namespace {
  using longlp::imgproc::Otsu2D;
}   // namespace

// https://sci-hub.se/10.1109/CCPR.2009.5344078
void Otsu2D::BinarizeImpl(const cv::Mat& input,
                          cv::Mat& output,
                          const bool use_background_white_color) const {
  output = input.clone();
  output.convertTo(output, CV_64F);

  cv::Mat average_image;
  cv::blur(input,
           average_image,
           kernel_size_,
           cv::Point(-1, -1) /* anchor at kernel center */,
           cv::BORDER_REFLECT /* symmetric padding */);

  cv::Mat merged;
  {
    const std::vector<cv::Mat> merge_list({input, average_image});
    cv::merge(merge_list, merged);
  }

  // Create 2D histogram
  cv::Mat f;
  {
    const std::vector<cv::Mat> images({merged});
    const std::vector<int> channels{{0, 1}};
    const std::vector<int> histSize{{256, 256}};
    const std::vector<float> ranges{
      {kGrayscaleMin, kGrayscaleMax, kGrayscaleMin, kGrayscaleMax}};
    cv::calcHist(images,
                 channels,
                 cv::noArray() /* no mask */,
                 f,
                 histSize,
                 ranges,
                 false /* disable accumulate old value from f */);
  }
  f.convertTo(f, CV_64F);

  // P = integral(f)
  cv::Mat P;
  cv::integral(f,
               P,
               CV_64F   // Force to store double in P
  );

  // X = integral([[0], [1], [2], ... [255]] * f)
  cv::Mat X;
  {
    cv::Mat temp = f.clone();
    temp.forEach<double>([](double& pixel, const int* position) {
      pixel *= position[0];
    });
    cv::integral(temp, X, CV_64F /*  Force to store double in X */);
  }

  // Y = integral(f * [0, 1, 2, 3, ..., 255])
  cv::Mat Y;
  {
    cv::Mat temp = f.clone();
    temp.forEach<double>([](double& pixel, const int* position) {
      pixel *= position[1];
    });
    cv::integral(temp, Y, CV_64F /* Force to store double in P */);
  }

  auto threshold_s  = cv::softdouble::zero();
  auto threshold_t  = cv::softdouble::zero();
  auto max_retrance = cv::softdouble::zero();

  for (auto s = 0; s < f.rows; ++s) {
    for (auto t = 0; t < f.cols; ++t) {
      const auto y = s + 1;
      const auto x = t + 1;

      const cv::softdouble X0{*X.ptr<double>(y, x)};
      const cv::softdouble Y0{*Y.ptr<double>(y, x)};

      const cv::softdouble X1{*X.ptr<double>(256, 256) -
                              *X.ptr<double>(256, x) - *X.ptr<double>(y, 256) +
                              X0};
      const cv::softdouble Y1{*Y.ptr<double>(256, 256) -
                              *Y.ptr<double>(256, x) - *Y.ptr<double>(y, 256) +
                              Y0};

      const cv::softdouble w0{*P.ptr<double>(y, x)};
      if (!(w0 > cv::softdouble::eps())) {
        continue;
      }

      const cv::softdouble w1{*P.ptr<double>(256, 256) -
                              *P.ptr<double>(256, x) - *P.ptr<double>(y, 256) +
                              *P.ptr<double>(y, x)};
      if (!(w1 > cv::softdouble::eps())) {
        break;
      }

      const auto u0 = (X0 + Y0) / (w0 + w0);
      const auto u1 = (X1 + Y1) / (w1 + w1);
      const auto ut = w0 * u0 + w1 * u1;

      if (const auto local_retrance =
            w0 * (ut - u0) * (ut - u0) + w1 * (ut - u1) * (ut - u1);
          local_retrance > max_retrance) {
        max_retrance = local_retrance;
        threshold_s  = cv::softdouble{s};
        threshold_t  = cv::softdouble{t};
      }
    }
  }

  //  Given an arbitrary threshold pair(s, t), the 2D histogram can be divided
  //  into four regions.Regions A and C represent object and background
  //  respectively, and regions B and D represent edge and noise respectively:
  //        g(x,y)
  //      ^
  //      |
  // L-1  +------+----------------+
  //      |      |                |
  //      |  D   |        C       |
  //      |      | (s, t)         |
  //   t  +-----------------------+
  //      |      |                |
  //      |  A   |      B         |
  //      |      |                |
  //      +------+----------------+------>
  //     0       s                L-1    f(x,y)

  const auto threshold =
    // object: A, background: edge(B) + C + noise(D) ~ threshold = max(s,t)
    edge_role_as_background_ && noise_role_as_background_
      ? cv::max(threshold_s, threshold_t)
      // object: A + noise(D), background: edge(B) + C ~ threshold = s
      : (edge_role_as_background_
           ? threshold_s
           // object: A + edge(B), background: C + noise(D) ~ threshold = t
           : (noise_role_as_background_ ? threshold_t
                                        // object: A + edge(B) + noise(D),
                                        // background: C ~ threshold = min(s, t)
                                        : cv::min(threshold_s, threshold_t)));

  cv::threshold(output,
                output,
                static_cast<double>(threshold),
                kGrayscaleMax,
                use_background_white_color
                  ? cv::ThresholdTypes::THRESH_BINARY
                  : cv::ThresholdTypes::THRESH_BINARY_INV);

  output.convertTo(output, CV_8U);
}

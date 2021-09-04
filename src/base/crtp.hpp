// Copyright 2021 Long Le Phi. All rights reserved.
// Use of this source code is governed by a MIT license that can be
// found in the LICENSE file.

#ifndef BASE_CRTP_HPP_
#define BASE_CRTP_HPP_

// Helper macro for access underlying class in CRTP class
#define IMPLEMENT_CRTP_HELPER(Derived)               \
  auto underlying() noexcept->Derived& {             \
    return static_cast<Derived&>(*this);             \
  }                                                  \
  auto underlying() const noexcept->const Derived& { \
    return static_cast<const Derived&>(*this);       \
  }

#endif   // BASE_CRTP_HPP_

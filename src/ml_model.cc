
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the base MLModel class (see the header for information about the class).
//
// -------------------------------------------------------------------------------------------------
//
// Copyright 2014 SURFsara
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//  http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================

// The corresponding header file
#include "internal/ml_model.h"

#include <vector>
#include <algorithm>

namespace cltune {
// =================================================================================================

// Simple constructor
template <typename T>
MLModel<T>::MLModel(const size_t m, const size_t n):
  ranges_(n, 1),
  means_(n, 0) {
}

// =================================================================================================

// Finds the ranges and the means for each feature
template <typename T>
void MLModel<T>::ComputeNormalizations(const std::vector<std::vector<T>> &x) {
  auto m = x.size();
  auto n = x[0].size();
  for (auto nid=size_t{0}; nid<n; ++nid) {
    auto min = std::numeric_limits<T>::max();
    auto max = -min;
    for (auto mid=size_t{0}; mid<m; ++mid) {
      auto value = x[mid][nid];
      if (value > max) { max = value; }
      if (value < min) { min = value; }
    }
    ranges_[nid] = max - min;
    means_[nid] = static_cast<T>(0); // TODO: implement this
  }
}

// Normalizes the training features based on previously calculated ranges and means
template <typename T>
void MLModel<T>::NormalizeFeatures(std::vector<std::vector<T>> &x) {
  auto m = x.size();
  auto n = x[0].size();
  for (auto nid=size_t{0}; nid<n; ++nid) {
    for (auto mid=size_t{0}; mid<m; ++mid) {
      auto value = x[mid][nid];
      x[mid][nid] = (value - means_[nid]) / ranges_[nid];
    }
  }
}

// Adds polynominal combinations of features as new features
template <typename T>
void MLModel<T>::AddPolynominalFeatures(std::vector<std::vector<T>> &x, const size_t order) {
  auto m = x.size();
  auto n = x[0].size();
  for (auto mid=size_t{0}; mid<m; ++mid) {

    // TODO: For now only 2 supported
    if (order == 2) {
      x[mid].reserve(x[mid].size() + n*n);
      for (auto nid1=size_t{0}; nid1<n; ++nid1) {
        for (auto nid2=size_t{0}; nid2<n; ++nid2) {
          x[mid].push_back(x[mid][nid1] * x[mid][nid2]);
        }
      }
    }
  }
}

// =================================================================================================

// Implements the gradient descent iterative search algorithm. This method is based upon a cost-
// function and gradient-function implemented by the derived class.
template <typename T>
void MLModel<T>::GradientDescent(const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                                 const T alpha, const size_t iterations) {
  auto m = x.size();
  auto n = x[0].size();

  // Sets the initial theta values
  // TODO: Move to a separate function
  theta_.resize(n);
  std::fill(theta_.begin(), theta_.end(), static_cast<T>(0));

  // Runs gradient descent
  for (auto iter=size_t{0}; iter<iterations; ++iter) {
    auto theta_temp = std::vector<T>(n, static_cast<T>(0));

    // Computes the cost (to monitor convergence)
    auto cost = Cost(m, n, x, y);
    if ((iter+1) % (iterations/kGradientDescentCostReportAmount) == 0) {
      printf("%s Gradient descent %lu/%lu: cost %.2e\n",
             TunerImpl::kMessageInfo.c_str(), iter+1, iterations, cost);
    }
    
    // Computes the gradients and the updated parameters
    for (auto nid=size_t{0}; nid<n; ++nid) {
      auto gradient = Gradient(m, n, x, y, nid);
      theta_temp[nid] = theta_[nid] - alpha * (1.0f/static_cast<T>(m)) * gradient;
    }

    // Sets the new values for theta
    for (auto nid=size_t{0}; nid<n; ++nid) {
      theta_[nid] = theta_temp[nid];
    }
  }
}

// =================================================================================================

// Classifies all training examples for verification
template <typename T>
float MLModel<T>::Verify(const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                         const float margin) const {
  auto m = x.size();
  auto correct = 0;
  for (auto mid=size_t{0}; mid<m; ++mid) {
    auto hypothesis = Hypothesis(x[mid]);
    auto limit_max = y[mid]*(1 + margin);
    auto limit_min = y[mid]*(1 - margin);
    if (hypothesis < limit_max && hypothesis > limit_min) { correct++; }
    printf("[ -------> ] Hypothesis: %7.3lf; Actual: %7.3lf\n", hypothesis, y[mid]);
  }
  auto success_rate = 100.0f*correct/static_cast<float>(m);
  return success_rate;
}

// =================================================================================================

// Compiles the class
template class MLModel<float>;

// =================================================================================================
} // namespace cltune

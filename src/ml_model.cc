
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
#include <cmath>

namespace cltune {
// =================================================================================================

// Simple constructor
template <typename T>
MLModel<T>::MLModel(const bool debug_display):
    debug_display_(debug_display) {
}

// =================================================================================================

// Finds the ranges and the means for each feature
template <typename T>
void MLModel<T>::ComputeNormalizations(const std::vector<std::vector<T>> &x) {
  auto m = x.size();
  auto n = x[0].size();

  // Loops over the features
  ranges_.resize(n, static_cast<T>(1));
  means_.resize(n, static_cast<T>(0));
  for (auto nid=size_t{0}; nid<n; ++nid) {

    // Finds the maximum, the minimum, and the sum
    auto min = std::numeric_limits<T>::max();
    auto max = -min;
    auto sum = static_cast<T>(0);
    for (auto mid=size_t{0}; mid<m; ++mid) {
      auto value = x[mid][nid];
      if (value > max) { max = value; }
      if (value < min) { min = value; }
      sum += value;
    }

    // Sets the range and the mean
    ranges_[nid] = max - min;
    means_[nid] = sum / static_cast<T>(m);
  }
}

// Normalizes the training features based on previously calculated ranges and means
template <typename T>
void MLModel<T>::NormalizeFeatures(std::vector<std::vector<T>> &x) const {
  for (auto mid=size_t{0}; mid<x.size(); ++mid) {
    for (auto nid=size_t{0}; nid<x[mid].size(); ++nid) {
      x[mid][nid] = (x[mid][nid] - means_[nid]) / ranges_[nid];
    }
  }
}

// Adds polynominal combinations of features as new features. This is implemented using recursion
// and allows any order larger than 1.
template <typename T>
void MLModel<T>::AddPolynomialFeatures(std::vector<std::vector<T>> &x,
                                       const std::vector<size_t> &orders) const {
  for (auto mid=size_t{0}; mid<x.size(); ++mid) {
    auto n = x[mid].size();
    for (auto &order: orders) {
      if (order > 1) {
        x[mid].reserve(x[mid].size() + static_cast<size_t>(pow(n, order)));
        AddPolynomialRecursive(x[mid], order, 1UL, n);
      }
    }
  }
}
template <typename T>
void MLModel<T>::AddPolynomialRecursive(std::vector<T> &xi, const size_t order, const T value,
                                        const size_t n) const {
  if (order == 0) {
    xi.push_back(value);
  }
  else {
    for (auto nid=size_t{0}; nid<n; ++nid) {
      AddPolynomialRecursive(xi, order-1, value*xi[nid], n);
    }
  }
}

// =================================================================================================

// Implements the gradient descent iterative search algorithm. This method is based upon a cost-
// function and gradient-function implemented by the derived class.
template <typename T>
void MLModel<T>::GradientDescent(const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                                 const T alpha, const T lambda, const size_t iterations) {
  auto m = x.size();
  auto n = x[0].size();

  // Sets the initial theta values
  theta_.resize(n);
  InitializeTheta();

  // Runs gradient descent
  for (auto iter=size_t{0}; iter<iterations; ++iter) {
    auto theta_temp = std::vector<T>(n, static_cast<T>(0));

    // Computes the cost (to monitor convergence)
    auto cost = Cost(m, n, lambda, x, y);
    if ((iter+1) % (iterations/kGradientDescentCostReportAmount) == 0) {
      printf("%s Gradient descent %lu/%lu: cost %.2e\n",
             TunerImpl::kMessageInfo.c_str(), iter+1, iterations, cost);
    }
    
    // Computes the gradients and the updated parameters
    for (auto nid=size_t{0}; nid<n; ++nid) {
      auto gradient = Gradient(m, n, lambda, x, y, nid);
      theta_temp[nid] = theta_[nid] - alpha * gradient;
    }

    // Sets the new values for theta
    for (auto nid=size_t{0}; nid<n; ++nid) {
      theta_[nid] = theta_temp[nid];
    }
  }
}

// =================================================================================================

// Verifies training examples: computes the success rate within a specified margin
template <typename T>
float MLModel<T>::SuccessRate(const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                              const float margin) const {
  auto m = x.size();
  auto correct = 0;
  for (auto mid=size_t{0}; mid<m; ++mid) {
    auto hypothesis = PostProcessExecutionTime(Hypothesis(x[mid]));
    auto reference = PostProcessExecutionTime(y[mid]);
    auto limit_max = reference*(1 + margin);
    auto limit_min = reference*(1 - margin);
    if (hypothesis < limit_max && hypothesis > limit_min) { correct++; }
    printf("[ -------> ] Hypothesis: %7.3lf; Actual: %7.3lf\n", hypothesis, reference);
  }
  auto success_rate = 100.0f*correct/static_cast<float>(m);
  return success_rate;
}

// Verifies training examples: computes the cost function
template <typename T>
float MLModel<T>::Verify(const std::vector<std::vector<T>> &x, const std::vector<T> &y) const {
  auto m = x.size();
  auto n = x[0].size();

  // Displays the data
  if (debug_display_) {
    printf("hypothesis; actual; error\n");
    for (auto mid=size_t{0}; mid<m; ++mid) {
      auto hypothesis = PostProcessExecutionTime(Hypothesis(x[mid]));
      auto reference = PostProcessExecutionTime(y[mid]);
      auto relative_error = (reference - hypothesis) / (reference);
      printf("%.3lf;%.3lf;%.2lf%%\n", hypothesis, reference, 100.0f*relative_error);
    }
  }

  // Computes the cost
  return Cost(m, n, 0, x, y);
}
// =================================================================================================

// Compiles the class
template class MLModel<float>;

// =================================================================================================
} // namespace cltune

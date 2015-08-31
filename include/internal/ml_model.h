
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains a base class which implements machine learning models. Actual models are
// derived from this class, such as linear regression or a neural network.
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

#ifndef CLTUNE_ML_MODEL_H_
#define CLTUNE_ML_MODEL_H_

#include <vector>
#include <string>
#include <functional>

// For output formatting messages
#include "internal/tuner_impl.h"

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class MLModel {
 public:

  // Constants
  static constexpr auto kGradientDescentCostReportAmount = 10;

  // Constructor
  MLModel(const size_t m, const size_t n);

  // Trains and validates the model
  virtual void Train(const std::vector<std::vector<T>> &x, const std::vector<T> &y) = 0;
  virtual void Validate(const std::vector<std::vector<T>> &x, const std::vector<T> &y) = 0;

 protected:

  // Process the training data
  void ComputeNormalizations(const std::vector<std::vector<T>> &x);
  void NormalizeFeatures(std::vector<std::vector<T>> &x);
  void AddPolynomialFeatures(std::vector<std::vector<T>> &x, const std::vector<size_t> &orders);
  void AddPolynomialRecursive(std::vector<T> &xi, const size_t order, const T value, const size_t n);

  // Methods to minimize an unconstrained function
  void GradientDescent(const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                       const T alpha, const T lambda, const size_t iterations);

  // Verification methods
  float SuccessRate(const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                    const float margin) const;
  float Verify(const std::vector<std::vector<T>> &x, const std::vector<T> &y) const;

  // Pure virtual hypothesis, cost and gradient functions
  virtual T Hypothesis(const std::vector<T> &x) const = 0;
  virtual T Cost(const size_t m, const size_t n, const T lambda,
                 const std::vector<std::vector<T>> &x, const std::vector<T> &y) const = 0;
  virtual T Gradient(const size_t m, const size_t n, const T lambda,
                     const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                     const size_t gradient_id) const = 0;

  // The learned weights
  std::vector<T> theta_;

  // Information for normalization
  std::vector<T> ranges_;
  std::vector<T> means_;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_ML_MODEL_H_
#endif

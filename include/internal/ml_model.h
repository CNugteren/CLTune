
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains a base class which implements machine learning models. Actual models are
// derived from this class, such as linear regression or a neural network. This class contains
// common functionality, such as gradient descent and feature normalization.
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
  MLModel(const bool debug_display);

  // Trains and validates the model
  virtual void Train(const std::vector<std::vector<T>> &x, const std::vector<T> &y) = 0;
  virtual void Validate(const std::vector<std::vector<T>> &x, const std::vector<T> &y) = 0;

  // Pure virtual prediction function: predicts 'y' based on 'x' and the learning parameters 'theta'
  virtual T Predict(const std::vector<T> &x) const = 0;

 protected:
  // Process the training data in various ways
  void ComputeNormalizations(const std::vector<std::vector<T>> &x);
  void NormalizeFeatures(std::vector<std::vector<T>> &x) const;
  void AddPolynomialFeatures(std::vector<std::vector<T>> &x, const std::vector<size_t> &orders) const;
  void AddPolynomialRecursive(std::vector<T> &xi, const size_t order, const T value,
                              const size_t n) const;

  // Methods to minimize an unconstrained function
  void GradientDescent(const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                       const T alpha, const T lambda, const size_t iterations);

  // Verification methods
  float SuccessRate(const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                    const float margin) const;
  float Verify(const std::vector<std::vector<T>> &x, const std::vector<T> &y) const;

  // Pre and post-processing of data
  virtual T PostProcessExecutionTime(T value) const = 0;

  // Pure virtual function for weights initialization
  virtual void InitializeTheta() = 0;

  // Pure virtual hypothesis, cost and gradient functions: to be implemented by derived classes
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

  // Settings
  const bool debug_display_;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_ML_MODEL_H_
#endif

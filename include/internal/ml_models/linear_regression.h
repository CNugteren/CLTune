
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains a linear regression model, derived from the base machine learning class.
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

#ifndef CLTUNE_ML_MODELS_LINEAR_REGRESSION_H_
#define CLTUNE_ML_MODELS_LINEAR_REGRESSION_H_

#include <vector>

// Machine learning base class
#include "internal/ml_model.h"

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class LinearRegression: public MLModel<T> {
 public:

  // Methods from the base class
  using MLModel<T>::ComputeNormalizations;
  using MLModel<T>::NormalizeFeatures;
  using MLModel<T>::AddPolynomialFeatures;
  using MLModel<T>::GradientDescent;
  using MLModel<T>::Verify;

  // Variables from the base class
  using MLModel<T>::theta_;
  using MLModel<T>::means_;
  using MLModel<T>::ranges_;

  // Constructor
  LinearRegression(const size_t learning_iterations, const T learning_rate, const T lambda,
                   const bool debug_display);

  // Trains and validates the model
  virtual void Train(const std::vector<std::vector<T>> &x, const std::vector<T> &y) override;
  virtual void Validate(const std::vector<std::vector<T>> &x, const std::vector<T> &y) override;

  // Prediction
  virtual T Predict(const std::vector<T> &x) const override;

 private:
  // Pre and post-processing of data
  void PreProcessFeatures(std::vector<std::vector<T>> &x) const;
  void PreProcessExecutionTimes(std::vector<T> &y) const;
  virtual T PostProcessExecutionTime(T value) const override;

  // Initializes the weights
  virtual void InitializeTheta() override;

  // Hypothesis, cost and gradient functions
  virtual T Hypothesis(const std::vector<T> &x) const override;
  virtual T Cost(const size_t m, const size_t n, const T lambda,
                 const std::vector<std::vector<T>> &x, const std::vector<T> &y) const override;
  virtual T Gradient(const size_t m, const size_t n, const T lambda,
                     const std::vector<std::vector<T>> &x, const std::vector<T> &y,
                     const size_t gradient_id) const override;

  size_t learning_iterations_;
  T learning_rate_;
  T lambda_; // Regularization parameter
};

// =================================================================================================
} // namespace cltune

// CLTUNE_ML_MODELS_LINEAR_REGRESSION_H_
#endif

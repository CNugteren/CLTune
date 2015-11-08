
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the NeuralNetwork class (see the header for information about the class).
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
#include "internal/ml_models/neural_network.h"

#include <vector>
#include <cmath>
#include <random>
#include <exception>
#include <chrono>

namespace cltune {
// =================================================================================================

// Calls the base-class constructor
template <typename T>
NeuralNetwork<T>::NeuralNetwork(const size_t learning_iterations, const T learning_rate,
                                const T lambda, const std::vector<size_t> &layer_sizes,
                                const bool debug_display):
    MLModel<T>(debug_display),
    num_layers_(layer_sizes.size()),
    layer_sizes_(layer_sizes),
    learning_iterations_(learning_iterations),
    learning_rate_(learning_rate),
    lambda_(lambda) {
  if (num_layers_ != 3) { throw std::runtime_error("Only supporting networks with 3 layers"); }
}

// =================================================================================================

// Trains the model
template <typename T>
void NeuralNetwork<T>::Train(const std::vector<std::vector<T>> &x, const std::vector<T> &y) {
  auto x_temp = x;
  auto y_temp = y;

  // Modifies data to get a better model
  ComputeNormalizations(x_temp);
  PreProcessFeatures(x_temp);
  PreProcessExecutionTimes(y_temp);

  // Runs gradient descent to train the model
  GradientDescent(x_temp, y_temp, learning_rate_, lambda_, learning_iterations_);

  // Verifies and displays the trained results
  auto cost = Verify(x_temp, y_temp);
  printf("%s Training cost: %.2e\n", TunerImpl::kMessageResult.c_str(), cost);
}

// Validates the model
template <typename T>
void NeuralNetwork<T>::Validate(const std::vector<std::vector<T>> &x, const std::vector<T> &y) {
  auto x_temp = x;
  auto y_temp = y;

  // Modifies validation data in the same way as the training data
  PreProcessFeatures(x_temp);
  PreProcessExecutionTimes(y_temp);

  // Verifies and displays the trained results
  auto cost = Verify(x_temp, y_temp);
  printf("%s Validation cost: %.2e\n", TunerImpl::kMessageResult.c_str(), cost);
}

// Prediction: pre-processe a single sample and pass it through the model
template <typename T>
T NeuralNetwork<T>::Predict(const std::vector<T> &x) const {
  auto x_preprocessed = std::vector<std::vector<T>>{x};
  PreProcessFeatures(x_preprocessed);
  return PostProcessExecutionTime(Hypothesis(x_preprocessed[0]));
}

// =================================================================================================

// Pre-processes the features based on normalization data
template <typename T>
void NeuralNetwork<T>::PreProcessFeatures(std::vector<std::vector<T>> &x) const {
  NormalizeFeatures(x);
}

// Pre-processes the execution times using a logarithmic function
template <typename T>
void NeuralNetwork<T>::PreProcessExecutionTimes(std::vector<T> &y) const {
  for (auto &value: y) { value = static_cast<T>(log(static_cast<double>(value))); }
}

// Post-processes an execution time using an exponent function (inverse of the logarithm)
template <typename T>
T NeuralNetwork<T>::PostProcessExecutionTime(T value) const {
  return static_cast<T>(exp(static_cast<double>(value)));
}

// =================================================================================================

// Initialization-function: sets the initial weights theta
template <typename T>
void NeuralNetwork<T>::InitializeTheta(const size_t n) {

  // Resizes the weight matrices theta
  if (layer_sizes_[0] != n) { throw std::runtime_error("Invalid size of the first layer"); }
  if (layer_sizes_[2] != 1) { throw std::runtime_error("Invalid size of the third layer"); }
  theta1_.resize((layer_sizes_[0]+1)*layer_sizes_[1]);
  theta2_.resize((layer_sizes_[1]+1)*layer_sizes_[2]);

  // Calculates the random-initialization range
  auto epsilon1 = static_cast<T>(sqrt(static_cast<T>(6))/sqrt(static_cast<T>(layer_sizes_[0]+layer_sizes_[1])));
  auto epsilon2 = static_cast<T>(sqrt(static_cast<T>(6))/sqrt(static_cast<T>(layer_sizes_[1]+layer_sizes_[2])));
  
  // Creates a random number generator
  const auto random_seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(static_cast<unsigned int>(random_seed));
  std::uniform_real_distribution<T> distribution1(-epsilon1, epsilon1);
  std::uniform_real_distribution<T> distribution2(-epsilon2, epsilon2);

  // Fills the weights with random values
  for (auto &weight: theta1_) { weight = distribution1(generator); }
  for (auto &weight: theta2_) { weight = distribution2(generator); }
}

// =================================================================================================

// Hypothesis-function: pass a single sample through the model and returns its hypothesis
// TODO: Memory usage and performance can be improved
template <typename T>
T NeuralNetwork<T>::Hypothesis(const std::vector<T> &x) const {

  // Adds a bias to the input data
  auto a0 = FeedForward0(x);
  
  // Performs the activations of the first hidden layer
  auto a1 = FeedForward1(a0, true);
  
  // Performs the activations of the output layer
  auto a2 = FeedForward2(a1);

  // Only one output supported
  if (layer_sizes_[2] != 1) { throw std::runtime_error("Invalid size of the third layer"); }
  auto hypothesis = a2[0];
  return hypothesis;
}

// Cost-function: computes the sum of squared differences
template <typename T>
T NeuralNetwork<T>::Cost(const size_t m, const size_t, const T lambda,
                         const std::vector<std::vector<T>> &x, const std::vector<T> &y) const {

  // Computes the sum of squared differences
  auto cost = static_cast<T>(0);
  for (auto mid=size_t{0}; mid<m; ++mid) {
    auto difference = Hypothesis(x[mid]) - y[mid];
    cost += difference * difference;
  }
  cost /= static_cast<T>(m);

  // Computes the squared sum of theta's (not counting theta-zero) for the regularization term
  auto theta_squared_sum = static_cast<T>(0);
  for (auto id1=size_t{0}; id1<layer_sizes_[1]; ++id1) {
    for (auto id0=size_t{1}; id0<layer_sizes_[0]+1; ++id0) {
      auto value = theta1_[id1*(layer_sizes_[0]+1) + id0];
      theta_squared_sum += value * value;
    }
  }
  for (auto id2=size_t{0}; id2<layer_sizes_[2]; ++id2) {
    for (auto id1=size_t{1}; id1<layer_sizes_[1]+1; ++id1) {
      auto value = theta2_[id2*(layer_sizes_[1]+1) + id1];
      theta_squared_sum += value * value;
    }
  }

  // Computes the final cost
  return cost + (lambda*theta_squared_sum) / (static_cast<T>(2 * m));
}

// Gradient-function: computes the gradient of the cost-function
template <typename T>
void NeuralNetwork<T>::Gradient(const size_t m, const size_t, const T lambda, const T alpha,
                                const std::vector<std::vector<T>> &x, const std::vector<T> &y) {

  // Temporary data storage for the gradients
  auto gradient1 = std::vector<T>(theta1_.size(), static_cast<T>(0));
  auto gradient2 = std::vector<T>(theta2_.size(), static_cast<T>(0));

  // Loops over all samples of the training data
  for (auto mid=size_t{0}; mid<m; ++mid) {

    // Performs the feed-forward computations (with and without sigmoid)
    auto a0 = FeedForward0(x[mid]);
    auto z1 = FeedForward1(a0, false);
    auto a1 = FeedForward1(a0, true);
    auto a2 = FeedForward2(a1);

    // Computes the error at the last layer
    auto d2 = std::vector<T>(layer_sizes_[2]);
    if (layer_sizes_[2] != 1) { throw std::runtime_error("Invalid size of the third layer"); }
    d2[0] = a2[0] - y[mid];

    // Propagates the error back (backpropagation) to compute the delta at the hidden layer
    auto d1 = std::vector<T>(layer_sizes_[1]);
    for (auto id1=size_t{1}; id1<layer_sizes_[1]+1; ++id1) {
      auto value = static_cast<T>(0);
      for (auto id2=size_t{0}; id2<layer_sizes_[2]; ++id2) {
        value += d2[id2] * theta2_[id2*(layer_sizes_[1]+1) + id1];
      }
      d1[id1-1] = value * SigmoidGradient(z1[id1]);
    }

    // Accumulates the partial gradients
    for (auto id0=size_t{0}; id0<layer_sizes_[0]+1; ++id0) {
      for (auto id1=size_t{0}; id1<layer_sizes_[1]; ++id1) {
        gradient1[id1*(layer_sizes_[0]+1) + id0] += d1[id1] * a0[id0];
      }
    }
    for (auto id1=size_t{0}; id1<layer_sizes_[1]+1; ++id1) {
      for (auto id2=size_t{0}; id2<layer_sizes_[2]; ++id2) {
        gradient2[id2*(layer_sizes_[1]+1) + id1] += d2[id2] * a1[id1];
      }
    }
  }

  // Computes the final gradients, adding regularization
  for (auto id0=size_t{0}; id0<layer_sizes_[0]+1; ++id0) {
    for (auto id1=size_t{0}; id1<layer_sizes_[1]; ++id1) {
      auto index = id1*(layer_sizes_[0]+1) + id0;
      if (id0 != 0) { // Don't add regularization for the bias term
        gradient1[index] += lambda * theta1_[index];
      }
      gradient1[index] /= static_cast<T>(m);
    }
  }
  for (auto id1=size_t{0}; id1<layer_sizes_[1]+1; ++id1) {
    for (auto id2=size_t{0}; id2<layer_sizes_[2]; ++id2) {
      auto index = id2*(layer_sizes_[1]+1) + id1;
      if (id1 != 0) { // Don't add regularization for the bias term
        gradient2[index] += lambda * theta2_[index];
      }
      gradient2[index] /= static_cast<T>(m);
    }
  }

  // Sets the new values of theta
  for (auto id0=size_t{0}; id0<layer_sizes_[0]+1; ++id0) {
    for (auto id1=size_t{0}; id1<layer_sizes_[1]; ++id1) {
      auto index = id1*(layer_sizes_[0]+1) + id0;
      theta1_[index] = theta1_[index] - alpha * gradient1[index];
    }
  }
  for (auto id1=size_t{0}; id1<layer_sizes_[1]+1; ++id1) {
    for (auto id2=size_t{0}; id2<layer_sizes_[2]; ++id2) {
      auto index = id2*(layer_sizes_[1]+1) + id1;
      theta2_[index] = theta2_[index] - alpha * gradient2[index];
    }
  }
}

// =================================================================================================

// Feed-forward function: input layer (with bias unit)
template <typename T>
std::vector<T> NeuralNetwork<T>::FeedForward0(const std::vector<T> &x) const {
  auto a0 = std::vector<T>(layer_sizes_[0]+1);
  a0[0] = static_cast<T>(1);
  for (auto id0=size_t{0}; id0<layer_sizes_[0]; ++id0) {
    a0[id0 + 1] = x[id0];
  }
  return a0;
}

// Feed-forward function: hidden layer (with bias unit and optionally a sigmoid activation function)
template <typename T>
std::vector<T> NeuralNetwork<T>::FeedForward1(const std::vector<T> &a0, const bool sigmoid) const {
  auto a1 = std::vector<T>(layer_sizes_[1]+1);
  a1[0] = static_cast<T>(1);
  for (auto id1=size_t{0}; id1<layer_sizes_[1]; ++id1) {
    auto z1 = static_cast<T>(0);
    for (auto id0=size_t{0}; id0<layer_sizes_[0]+1; ++id0) {
      z1 += a0[id0] * theta1_[id1*(layer_sizes_[0]+1) + id0];
    }
    a1[id1 + 1] = (sigmoid) ? Sigmoid(z1) : z1;
  }
  return a1;
}

// Feed-forward function: output layer
template <typename T>
std::vector<T> NeuralNetwork<T>::FeedForward2(const std::vector<T> &a1) const {
  auto a2 = std::vector<T>(layer_sizes_[2]);
  for (auto id2=size_t{0}; id2<layer_sizes_[2]; ++id2) {
    auto z2 = static_cast<T>(0);
    for (auto id1=size_t{0}; id1<layer_sizes_[1]+1; ++id1) {
      z2 += a1[id1] * theta2_[id2*(layer_sizes_[1]+1) + id1];
    }
    a2[id2] = z2; // No sigmoid activation function in the output layer
  }
  return a2;
}

// =================================================================================================

// Compiles the class
template class NeuralNetwork<float>;

// =================================================================================================
} // namespace cltune

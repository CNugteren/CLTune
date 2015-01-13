
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file tests public methods of the Tuner class.
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

#include "tuner/tuner.h"

#include "gtest/gtest.h"

// =================================================================================================

// Initializes a Tuner test fixture
class TunerTest : public testing::Test {
 protected:

  // Helper structure for testing
  struct CLKernel {
    std::string filename;
    std::string kernel_name;
  };

  // Test parameters
  const int kNumKernelAdditions = 1;
  const int kNumParameters = 2;
  const int kNumParameterAdditions = 3;

  // Test kernels (taken from the samples folder)
  const int kNumKernels = 2;
  const std::vector<CLKernel> kKernelFiles = {
    {"../samples/simple_reference.opencl","matvec_reference"},
    {"../samples/simple_unroll.opencl","matvec_unroll"}
  };

  // Test matrix sizes
  const int kSizeM = 128;
  const int kSizeN = 512;
  const int kSizeK = 256;

  // Constructor
  explicit TunerTest() :
    tuner_(new cltune::Tuner(0, 0)) {
  }

  // Initializes the tester
  virtual void SetUp() {
    tuner_->SuppressOutput();

    // Creates example global/local thread-sizes
    global_ = cl::NDRange(kSizeM, kSizeN);
    local_ = cl::NDRange(8, 1);

    // Adds example parameters
    for (int k=0; k<kNumParameters; k++) {
      std::string parameter = "TEST_PARAM_"+std::to_string(static_cast<long long>(k));
      std::initializer_list<int> values = {5, 1, 999};
      cltune::StringRange string_range = {parameter, parameter};
      parameter_list_.push_back(parameter);
      values_list_.push_back(values);
      string_ranges_.push_back(string_range);
    }
  }

  virtual void TearDown() {
  }

  // Member variables
  std::shared_ptr<cltune::Tuner> tuner_;
  cl::NDRange global_;
  cl::NDRange local_;
  std::vector<std::string> parameter_list_;
  std::vector<std::initializer_list<int>> values_list_;
  std::vector<cltune::StringRange> string_ranges_;
};

// =================================================================================================

// Tests the initialization of OpenCL
TEST_F(TunerTest, InitOpenCL) {

  // Normal behaviour
  cltune::Tuner tuner0(0, 0);
  tuner0.SuppressOutput();

  // Invalid behaviour, expect an exception
  ASSERT_THROW(new cltune::Tuner(0, 99), std::runtime_error);
  ASSERT_THROW(new cltune::Tuner(99, 0), std::runtime_error);
  ASSERT_THROW(new cltune::Tuner(99, 99), std::runtime_error);
}

// =================================================================================================

// Checks whether AddKernel returns an incrementing ID
TEST_F(TunerTest, AddKernel) {
  int counter = 0;
  for (auto &kernel_file: kKernelFiles) {
    for (int i=0; i<kNumKernelAdditions; i++) {
      int id = tuner_->AddKernel(kernel_file.filename, kernel_file.kernel_name, global_, local_);
      EXPECT_EQ(counter, id);
      counter++;
    }
  }
}

// Tests the AddParameter and AddKernel functions
TEST_F(TunerTest, AddParameter) {

  // Adds parameters for invalid kernels, expecting a crash
  for (int k=0; k<kNumParameters; k++) {
    ASSERT_THROW(tuner_->AddParameter(k, parameter_list_[k], values_list_[k]),
                 cltune::Tuner::TunerException);
  }

  // Adds a new kernel and then adds parameters
  for (auto &kernel_file: kKernelFiles) {
    for (int i=0; i<kNumKernelAdditions; i++) {
      int id = tuner_->AddKernel(kernel_file.filename, kernel_file.kernel_name, global_, local_);
      for (int k=0; k<kNumParameters; k++) {
        for (int j=0; j<kNumParameterAdditions; j++) {
          if (j == 0) {
            tuner_->AddParameter(id, parameter_list_[k], values_list_[k]);
          }
          else {
            ASSERT_THROW(tuner_->AddParameter(id, parameter_list_[k], values_list_[k]),
                         cltune::Tuner::TunerException);
          }
        }
      }
    }
  }
}

// Tests whether the {Mul/Div}{Global/Local}Size functions return assertions when used with invalid
// kernel IDs. Also check if they don't crash for valid kernels
TEST_F(TunerTest, ModifyThreadSize) {

  // Modifies parameters for invalid kernels, expecting a crash
  for (int k=0; k<kNumParameters; k++) {
    ASSERT_THROW(tuner_->MulGlobalSize(k, string_ranges_[k]), cltune::Tuner::TunerException);
    ASSERT_THROW(tuner_->DivGlobalSize(k, string_ranges_[k]), cltune::Tuner::TunerException);
    ASSERT_THROW(tuner_->MulLocalSize(k, string_ranges_[k]), cltune::Tuner::TunerException);
    ASSERT_THROW(tuner_->DivLocalSize(k, string_ranges_[k]), cltune::Tuner::TunerException);
  }

  // Adds a new kernel and then modifies the thread-sizes
  for (auto &kernel_file: kKernelFiles) {
    for (int i=0; i<kNumKernelAdditions; i++) {
      int id = tuner_->AddKernel(kernel_file.filename, kernel_file.kernel_name, global_, local_);
      for (int k=0; k<kNumParameters; k++) {
        tuner_->MulGlobalSize(id, string_ranges_[k]);
        tuner_->DivGlobalSize(id, string_ranges_[k]);
        tuner_->MulLocalSize(id, string_ranges_[k]);
        tuner_->DivLocalSize(id, string_ranges_[k]);
      }
    }
  }
}

// =================================================================================================

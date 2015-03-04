
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file tests public methods of the KernelInfo class.
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

#include "tuner/internal/kernel_info.h"

#include <memory>

// Includes the Google Test framework
#include "gtest/gtest.h"

// =================================================================================================

// Initializes a KernelInfo test fixture
class KernelInfoTest : public testing::Test {
 protected:
  static constexpr auto kNumParameters = 8;
  static constexpr auto kNumRanges = 8;

  // Constructor
  explicit KernelInfoTest() :
    kernel_(new cltune::KernelInfo("name", "source")) {
  }

  // Initializes the tester
  virtual void SetUp() {

    // Sets a bunch of parameters to test
    for (auto i=0; i<kNumParameters; ++i) {

       // Creates a pseudo-random name and values
      std::string name = "TEST_PARAM_" + std::to_string(static_cast<long long>(i));
      std::vector<int> values = {1, 6+i, 9, 1*i, 2000};
      for (auto j=0; j<i; ++j) { values.push_back((j+3)*i); }

      // Sets the name and value
      values_list_.push_back(values);
      names_.push_back(name);
    }

    // Creates some example NDRanges and StringRanges
    for (auto i=0; i<kNumRanges; ++i) {
      cl::NDRange range;
      cltune::StringRange string_range;

      // Sets some example values
      auto v1 = static_cast<long long>(i*i);
      auto v2 = static_cast<long long>(i+3);
      auto v3 = static_cast<long long>(8);

      // Creates ranges different lengths (x,y,z)
      if (i%4 == 0) {
        range = cl::NDRange();
        string_range = cltune::StringRange();
      }
      if (i%4 == 1) {
        range = cl::NDRange(v1);
        string_range = cltune::StringRange(std::to_string(v1));
      }
      if (i%4 == 2) {
        range = cl::NDRange(v1, v2);
        string_range = cltune::StringRange(std::to_string(v1), std::to_string(v2));
      }
      if (i%4 == 3) {
        range = cl::NDRange(v1, v2, v3);
        string_range = cltune::StringRange(std::to_string(v1), std::to_string(v2),
                                           std::to_string(v3));
      }

      // Stores the ranges
      ranges_.push_back(range);
      string_ranges_.push_back(string_range);
    }
  }

  virtual void TearDown() {
  }

  // Member variables
  std::shared_ptr<cltune::KernelInfo> kernel_;
  std::vector<std::string> names_;
  std::vector<std::vector<int>> values_list_;
  std::vector<cl::NDRange> ranges_;
  std::vector<cltune::StringRange> string_ranges_;
};

// =================================================================================================

// Tests set_global_base for a number of example NDRange values
TEST_F(KernelInfoTest, SetGlobalBase) {
  for (auto i=0; i<kNumRanges; ++i) {
    kernel_->set_global_base(ranges_[i]);
    ASSERT_EQ(ranges_[i].dimensions(), kernel_->global_base().dimensions());
    for (auto j=0; j<kernel_->global_base().dimensions(); ++j) {
      EXPECT_EQ(ranges_[i][j], kernel_->global_base()[j]);
    }
  }
}

// Tests set_local_base for a number of example NDRange values
TEST_F(KernelInfoTest, SetLocalBase) {
  for (auto i=0; i<kNumRanges; ++i) {
    kernel_->set_local_base(ranges_[i]);
    ASSERT_EQ(ranges_[i].dimensions(), kernel_->local_base().dimensions());
    for (auto j=0; j<kernel_->local_base().dimensions(); ++j) {
      EXPECT_EQ(ranges_[i][j], kernel_->local_base()[j]);
    }
  }
}

// Adds a number of parameter and then tests whether they are all set correctly
TEST_F(KernelInfoTest, AddParameter) {

  // Adds several parameters
  for (auto i=0; i<kNumParameters; ++i) {
    kernel_->AddParameter(names_[i], values_list_[i]);
  }

  // Tests each parameter
  for (auto i=0; i<kNumParameters; ++i) {
    ASSERT_EQ(values_list_[i].size(), kernel_->parameters()[i].values.size());
    EXPECT_EQ(names_[i], kernel_->parameters()[i].name);
    for (auto j=0; j<kernel_->parameters()[i].values.size(); ++j) {
      EXPECT_EQ(values_list_[i][j], kernel_->parameters()[i].values[j]);
    }
  }
}

// Tests CreateLocalRange and SetLocalString
TEST_F(KernelInfoTest, CreateLocalRange) {

  // Sets an example configuration
  cltune::KernelInfo::Configuration config;
  config.push_back(cltune::KernelInfo::Setting({"PARAM", 32}));

  // Tests a couple of different ranges against this configuration
  for (auto i=0; i<kNumRanges; ++i) {
    kernel_->set_global_base(ranges_[i]);
    kernel_->set_local_base(ranges_[i]);
    kernel_->ComputeRanges(config);
    ASSERT_EQ(ranges_[i].dimensions(), kernel_->local_base().dimensions());
    for (auto j=0; j<kernel_->local_base().dimensions(); ++j) {
      EXPECT_EQ(ranges_[i][j], kernel_->local_base()[j]);
    }
  }
}

// =================================================================================================

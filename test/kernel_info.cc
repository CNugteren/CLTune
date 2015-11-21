
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file tests public methods of the KernelInfo class.
//
// =================================================================================================

#include "catch.hpp"

#include "internal/kernel_info.h"

// Settings
const size_t kPlatformID = 0;
const size_t kDeviceID = 0;

// =================================================================================================

SCENARIO("kernel info objects can be used", "[KernelInfo]") {
  GIVEN("An example kernel info object") {

    auto platform = cltune::Platform(kPlatformID);
    auto device = cltune::Device(platform, kDeviceID);
    cltune::KernelInfo kernel("name", "source", device);

    // Example data
    const std::vector<cltune::IntRange> kExampleRanges = {
      cltune::IntRange(8, 16),
      cltune::IntRange(4, 1),
      cltune::IntRange(17, 3)
    };
    const std::vector<std::pair<std::string,std::vector<size_t>>> kExampleParameters = {
      {"example_param_0", {8, 16}},
      {"example_param_1", {4, 1}},
      {"example_param_2", {17, 3}}
    };

    WHEN("the global range is set") {
      for (auto i=size_t{0}; i<kExampleRanges.size(); ++i) {
        kernel.set_global_base(kExampleRanges[i]);
        THEN("the values are stored correctly #" + std::to_string(i)) {
          REQUIRE(kExampleRanges[i].size() == kernel.global_base().size());
          for (auto j=size_t{0}; j<kernel.global_base().size(); ++j) {
            REQUIRE(kExampleRanges[i][j] == kernel.global_base()[j]);
          }
        }
      }
    }

    WHEN("the local range is set") {
      for (auto i=size_t{0}; i<kExampleRanges.size(); ++i) {
        kernel.set_local_base(kExampleRanges[i]);
        THEN("the values are stored correctly #" + std::to_string(i)) {
          REQUIRE(kExampleRanges[i].size() == kernel.local_base().size());
          for (auto j=size_t{0}; j<kernel.local_base().size(); ++j) {
            REQUIRE(kExampleRanges[i][j] == kernel.local_base()[j]);
          }
        }
      }
    }

    WHEN("parameters are added") {
      for (auto i=size_t{0}; i<kExampleParameters.size(); ++i) {
        kernel.AddParameter(kExampleParameters[i].first, kExampleParameters[i].second);
        THEN("the values are stored correctly #" + std::to_string(i)) {
          REQUIRE(kExampleParameters[i].first == kernel.parameters()[i].name);
          REQUIRE(kExampleParameters[i].second.size() == kernel.parameters()[i].values.size());
          for (auto j=size_t{0}; j<kernel.local_base().size(); ++j) {
            REQUIRE(kExampleParameters[i].first[j] == kernel.parameters()[i].values[j]);
          }
        }
      }
    }

    WHEN("a configuration is set") {
      cltune::KernelInfo::Configuration config;
      config.push_back(cltune::KernelInfo::Setting({"example_param", 32}));
      for (auto i=size_t{0}; i<kExampleParameters.size(); ++i) {
        kernel.set_global_base(kExampleRanges[i]);
        kernel.set_local_base(kExampleRanges[i]);
        kernel.ComputeRanges(config);
        THEN("the values are stored correctly #" + std::to_string(i)) {
          REQUIRE(kExampleRanges[i].size() == kernel.local_base().size());
          for (auto j=size_t{0}; j<kernel.local_base().size(); ++j) {
            REQUIRE(kExampleRanges[i][j] == kernel.local_base()[j]);
          }
        }
      }
    }

  }
}

// =================================================================================================

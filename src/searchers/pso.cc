
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the PSO class (see the header for information about the class).
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
#include "internal/searchers/pso.h"

#include <algorithm>
#include <limits>

namespace cltune {
// =================================================================================================

// Initializes the PSO searcher
PSO::PSO(const Configurations &configurations, const Parameters &parameters,
         const double fraction, const size_t swarm_size, const double influence_global,
         const double influence_local, const double influence_random):
    Searcher(configurations),
    fraction_(fraction),
    swarm_size_(swarm_size),
    influence_global_(influence_global),
    influence_local_(influence_local),
    influence_random_(influence_random),
    particle_index_(0),
    particle_positions_(swarm_size_),
    global_best_time_(std::numeric_limits<double>::max()),
    local_best_times_(swarm_size_, std::numeric_limits<double>::max()),
    global_best_config_(),
    local_best_configs_(swarm_size_),
    parameters_(parameters),
    generator_(RandomSeed()),
    int_distribution_(0, static_cast<int>(configurations_.size())),
    probability_distribution_(0.0, 1.0) {
  for (auto &position: particle_positions_) {
    position = static_cast<size_t>(int_distribution_(generator_));
  }
  index_ = particle_positions_[particle_index_];
}

// =================================================================================================

// Returns the next configuration. This is similar to other searchers.
KernelInfo::Configuration PSO::GetConfiguration() {
  return configurations_[index_];
}

// Computes the next position of the current particle in the swarm. This is based on probabilities.
void PSO::CalculateNextIndex() {

  // Calculates the next state of the current swarm. This next state could be an invalid
  // configuration, so the next block is put in a do-while loop and only ends when a valid next
  // state is found. The next state is computed for each dimension separately and can depend on:
  // 1) the global best, 2) the particle's best so far, 3) a random location, and 4) its previous
  // location.
  auto new_index = index_;
  do {
    auto next_configuration = configurations_[index_];
    for (auto i=size_t{0}; i<next_configuration.size(); ++i) {
      //printf("%s = %d\n", next_configuration[i].name.c_str(), next_configuration[i].value);

      // Move towards best known globally (swarm)
      if (probability_distribution_(generator_) <= influence_global_) {
        next_configuration[i].value = global_best_config_[i].value;
      }
      // Move towards best known locally (particle)
      else if (probability_distribution_(generator_) <= influence_local_) {
        next_configuration[i].value = local_best_configs_[particle_index_][i].value;
      }
      // Move in a random direction
      else if (probability_distribution_(generator_) <= influence_random_) {
        std::uniform_int_distribution<size_t> distribution(0, parameters_[i].values.size());
        next_configuration[i].value = parameters_[i].values[distribution(generator_)];
      }
      // Else: stay at current location
    }
    new_index = IndexFromConfiguration(next_configuration);
  } while (new_index >= configurations_.size());
  particle_positions_[particle_index_] = new_index;

  // Calculates the next index --> move to the next particle in the swarm
  ++particle_index_;
  if (particle_index_ == swarm_size_) { particle_index_ = 0; }
  index_ = particle_positions_[particle_index_];
}

// The number of configurations is equal to all possible configurations
size_t PSO::NumConfigurations() {
  return std::max(size_t{1}, static_cast<size_t>(configurations_.size()*fraction_));
}

// =================================================================================================

// Adds the resulting execution time to the back of the execution times vector. Also updates the
// swarm best and global best configurations and execution times.
void PSO::PushExecutionTime(const double execution_time) {
  explored_indices_.push_back(index_);
  execution_times_[index_] = execution_time;
  if (execution_time < local_best_times_[particle_index_]) {
    local_best_times_[particle_index_] = execution_time;
    local_best_configs_[particle_index_] = configurations_[index_];
  }
  if (execution_time < global_best_time_) {
    global_best_time_ = execution_time;
    global_best_config_ = configurations_[index_];
  }
}

// =================================================================================================

// Searches all configuration to find which configuration is the 'target' (argument to this
// function). The target's index in the total configuration vector is returned.
size_t PSO::IndexFromConfiguration(const KernelInfo::Configuration target) const {
  auto config_index = size_t{0};
  for (auto &configuration: configurations_) {
    auto num_matches = size_t{0};
    for (auto i=size_t{0}; i<configuration.size(); ++i) {
      if (configuration[i].value == target[i].value) { num_matches++; }
    }
    if (num_matches == configuration.size()) { return config_index; }
    ++config_index;
  }

  // No match is found: this is an invalid configuration
  return config_index;
}

// =================================================================================================
} // namespace cltune

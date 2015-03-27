
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the Annealing class (see the header for information about the class).
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

#include "tuner/internal/searchers/annealing.h"

#include <limits>
#include <cmath>

namespace cltune {
// =================================================================================================

// Initializes the simulated annealing searcher by specifying the fraction of the total search space
// to consider and the maximum annealing 'temperature'.
Annealing::Annealing(const Configurations &configurations,
                     const float fraction, const double max_temperature):
    Searcher(configurations),
    fraction_(fraction),
    max_temperature_(max_temperature),
    num_visited_states_(0),
    current_state_(0),
    neighbour_state_(0),
    num_already_visisted_states_(0),
    generator_(rd_()),
    int_distribution_(0, configurations_.size()),
    probability_distribution_(0.0, 1.0) {
}

// =================================================================================================

// Returns the next configuration. This is similar to other searchers, but now also keeps track of
// the number of visited states to be able to compute the temperature.
KernelInfo::Configuration Annealing::GetConfiguration() {
  ++num_visited_states_;
  return configurations_[index_];
}

// Computes the new temperate, the new state (based on the acceptance probability function), and
// a random neighbour of the new state. If the newly calculated neighbour is already visited, this
// function is called recursively until some maximum number of calls has been reached.
void Annealing::CalculateNextIndex() {

  // Computes the new temperature
  auto progress = num_visited_states_ / static_cast<double>(NumConfigurations());
  auto temperature = max_temperature_ * (1.0 - progress);

  // Determines whether to continue with the neighbour or with the current ID
  auto acceptance_probability = AcceptanceProbability(execution_times_[current_state_],
                                                      execution_times_[neighbour_state_],
                                                      temperature);
  auto random_probability = probability_distribution_(generator_);
  if (acceptance_probability > random_probability) {
    current_state_ = neighbour_state_;
  }

  // Computes the new neighbour state
  auto neighbours = GetNeighboursOf(current_state_);
  neighbour_state_ = neighbours[int_distribution_(generator_) % neighbours.size()];

  // Checks whether this neighbour was already visited. If so, calculate a new neighbour instead.
  // This continues up to a maximum number, because all neighbours might already be visited. In
  // that case, the algorithm terminates.
  if (execution_times_[neighbour_state_] != std::numeric_limits<double>::max()) {
    if (num_already_visisted_states_ < kMaxAlreadyVisitedStates) {
      ++num_already_visisted_states_;
      CalculateNextIndex();
      return;
    }
  }
  num_already_visisted_states_ = 0;

  // Sets the next index
  index_ = neighbour_state_;
}

// The number of configurations is equal to all possible configurations
size_t Annealing::NumConfigurations() {
  return configurations_.size()*fraction_;
}

// =================================================================================================

// Adds the resulting execution time to the back of the execution times vector. Also stores the
// index value (to keep track of which indices are explored).
void Annealing::PushExecutionTime(const double execution_time) {
  explored_indices_.push_back(current_state_);
  execution_times_[index_] = execution_time;
}

// =================================================================================================

// Retrieves the neighbours IDs of a configuration identified by a reference ID. This searches
// through all configurations and checks how many values are different. This function returns a
// vector with IDs of which there is only one difference with the reference configuration.
// TODO: Is there a smarter way to compute this? This can become quite slow if the number of
// configurations is large.
std::vector<size_t> Annealing::GetNeighboursOf(const size_t reference_id) const {
  auto neighbours = std::vector<size_t>{};
  auto other_id = 0;
  for (auto &configuration: configurations_) {

    // Count the number of different settings for this configuration
    auto differences = 0;
    auto setting_id = 0;
    for (auto &setting: configuration) {
      if (setting.value != configurations_[reference_id][setting_id].value) { ++differences; }
      ++setting_id;
    }

    // Consider this configuration a neighbour if there is exactly one difference
    if (differences == 1) {
      neighbours.push_back(other_id);
    }
    ++other_id;
  }
  return neighbours;
}

// Computes the acceptance probablity P(e_current, e_neighbour, T) based on the Kirkpatrick et al.
// method: if the new (neighbouring) energy is lower, always accept it. If it is higher, there is
// a chance to accept it based on the energy difference and the current temperature (decreasing
// over time).
double Annealing::AcceptanceProbability(const double current_energy,
                                        const double neighbour_energy,
                                        const double temperature) const {
  if (neighbour_energy < current_energy) { return 1.0; }
  return exp( - (neighbour_energy - current_energy) / temperature);
}

// =================================================================================================
} // namespace cltune

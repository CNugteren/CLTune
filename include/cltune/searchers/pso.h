
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements a variant of particle swarm optimisation (PSO). It is adapted from PSO
// because of the higly dimensional discrete (or even boolean) search space. Therefore, there is no
// continous position nor velocity calculation. In fact, velocity is completely absent, following
// the principles of accelerated PSO (or APSO). Parameters to this form of PSO are the swarm size,
// the fraction of search space to explore, and the influences of the global best position, the
// local (particle's) best position, and the random influence.
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

#ifndef CLTUNE_SEARCHERS_PSO_H_
#define CLTUNE_SEARCHERS_PSO_H_

#include <vector>
#include <random>

#include "cltune/searcher.h"
#include "cltune/kernel_info.h"

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
class PSO: public Searcher {
 public:

  // Shorthand
  using Parameters = std::vector<KernelInfo::Parameter>;

  // Takes additionally a fraction of configurations to consider
  PSO(const Configurations &configurations, const Parameters &parameters,
      const double fraction, const size_t swarm_size, const double influence_global,
      const double influence_local, const double influence_random);

  // Retrieves the next configuration to test
  virtual KernelInfo::Configuration GetConfiguration() override;

  // Calculates the next index
  virtual void CalculateNextIndex() override;

  // Retrieves the total number of configurations to try
  virtual size_t NumConfigurations() override;

  // Pushes feedback (in the form of execution time) from the tuner to the search algorithm
  virtual void PushExecutionTime(const double execution_time) override;

 private:

  // Returns the index of the target configuration in the whole configuration list
  size_t IndexFromConfiguration(const KernelInfo::Configuration target) const;

  // Configuration parameters
  double fraction_;
  size_t swarm_size_;

  // Percentages of influence on the whole swarm's best (global), the particle's best (local), and
  // the random values. The remainder fraction is the chance of staying in the current position.
  float influence_global_;
  float influence_local_;
  float influence_random_;

  // Locations of the particles in the swarm
  size_t particle_index_;
  std::vector<size_t> particle_positions_;

  // Best cases found so far
  double global_best_time_;
  std::vector<double> local_best_times_;
  KernelInfo::Configuration global_best_config_;
  std::vector<KernelInfo::Configuration> local_best_configs_;

  // Allowed parameters
  Parameters parameters_;

  // Random number generation
  std::default_random_engine generator_;
  std::uniform_int_distribution<int> int_distribution_;
  std::uniform_real_distribution<double> probability_distribution_;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_SEARCHERS_PSO_H_
#endif

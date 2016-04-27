CLTune: API reference
================

This file describes the API of the CLTune auto-tuner for OpenCL and CUDA kernels. Everything is in the `cltune` namespace.


Constructors
-------------

* `Tuner()`:
Initializes a new tuner on platform 0 and device 0.

* `Tuner(size_t platform_id, size_t device_id)`:
Initializes a new tuner on platform `platform_id` and device `device_id`. For CUDA `platform_id` should be set to 0.


Auto-tuning
-------------

* `size_t AddKernel(const std::vector<std::string> &filenames, const std::string &kernel_name, const IntRange &global, const IntRange &local)`:
Adds a new kernel to the list of tuning-kernels and returns a unique ID (to be used when adding tuning parameters). This loads one or more kernel files given by a vector of `filenames`. The string `kernel_name` gives the entry point of the kernel. The parameters `global` and `local` represent the base 1D, 2D, or 3D thread configuration with `local` being the size of a work-group/thread-block, and `global` being the total amount of threads in each dimension.

* `size_t AddKernelFromString(const std::string &source, const std::string &kernel_name, const IntRange &global, const IntRange &local)`:
As above, but now the kernel is loaded from a string instead of from a file.

* `void AddParameter(const size_t id, const std::string &parameter_name, const std::initializer_list<size_t> &values)`:
Adds a new tuning parameter for the kernel with the given `id`. The parameter has as a name `parameter_name`, and a list of tuneable integer values.

* `void MulGlobalSize(const size_t id, const StringRange range)`:
Multiplies the global thread configuration for kernel `id` by one of the specified tuning parameters given as a 1D, 2D, or 3D `range`.

* `void DivGlobalSize(const size_t id, const StringRange range)`:
As above, but global thread division instead.

* `void MulLocalSize(const size_t id, const StringRange range)`:
As above, but local thread multiplication instead.

* `void DivLocalSize(const size_t id, const StringRange range)`:
As above, but local thread division instead.

* `template <typename T> void AddArgumentInput(const std::vector<T> &source)` and `template <typename T> void AddArgumentOutput(const std::vector<T> &source)` and `template <typename T> void AddArgumentScalar(const T argument)`:
Functions to add kernel-arguments for input or output buffers (given as `std::vector` CPU arrays) and scalars. These should be called in the order in which the arguments appear in the kernel.

* `void Tune()`:
Starts the tuning process after everything is set-up. This compiles all kernels and runs them for each permutation of the tuning-parameters.


Constraints
-------------

* `void AddConstraint(const size_t id, ConstraintFunction valid_if, const std::vector<std::string> &parameters)`:
Adds a new constraint (e.g. must be equal or larger than) to the set of parameters of kernel `id`. The constraint `valid_if` comes in the form of a function object which takes a number of tuning parameters, given as a vector of tuning-parameters (`parameters`). Their names are later substituted by actual values.

* `void SetLocalMemoryUsage(const size_t id, LocalMemoryFunction amount, const std::vector<std::string> &parameters)`:
As above, but for local memory usage. If this method is not called, it is assumed that the local memory usage is zero: no configurations will be excluded because of too much local memory.


Verification
-------------

* `void SetReference(const std::vector<std::string> &filenames, const std::string &kernel_name, const IntRange &global, const IntRange &local)`:
Sets the reference kernel for automatic verification purposes. Same arguments as the `AddKernel()` method, but in this case there can be only one reference kernel so no ID is returned. Calling this method again will overwrite the previous reference kernel.

* `void SetReferenceFromString(const std::string &source, const std::string &kernel_name, const IntRange &global, const IntRange &local)`:
As above, but now the reference kernel is loaded from a string instead of from a file.

* `void AddParameterReference(const std::string &parameter_name, const size_t value)`:
For convenience, a tuning 'parameter' `parameter_name` with a single value `value` can be added to the reference kernel as well. This can be useful in case the same kernel is used for tuning and as reference and certain values are not defined. It is not necessary to call this function in case a separate fully functional OpenCL or CUDA kernel is supplied.


Search methods
-------------

* `void UseFullSearch()`:
Call this method before calling the `Tune()` method. This will use full-search, i.e. all configurations will be tested on the device and the best-result will be found by the tuner. This is the default behaviour: it is not necessary to call this method except to override a previously set search method.

* `void UseRandomSearch(const double fraction)`:
Call this method before calling the `Tune()` method. This will make the tuner explore only a random subset of all configurations. The size of the subset is given as the fraction `fraction`. For example, passing `0.01` will explore 1% of the search-space.

* `void UseAnnealing(const double fraction, const double max_temperature)`:
Call this method before calling the `Tune()` method. This will make the tuner explore only a subset (size determined by `fraction`) of all configurations according to the simulated annealing algorithm with a maximum 'temperature' of `max_temperature`. Annealing uses randomly generated numbers, so behaviour will change from run to run.

* `void UsePSO(const double fraction, const size_t swarm_size, const double influence_global, const double influence_local, const double influence_random)`:
Call this method before calling the `Tune()` method. This will make the tuner explore only a subset (size determined by `fraction`) of all configurations according to the particle swarm optimisation (PSO) algorithm with a swarm size of `swarm_size` and fractional influence values for the global, local, and random search directions. PSO uses randomly generated numbers, so behaviour will change from run to run.

* `void ModelPrediction(const Model model_type, const float validation_fraction, const size_t test_top_x_configurations)`:
Call this method *after* calling the `Tune()` method. Trains a machine learning model of type `model_type` (`kLinearRegression` or `kNeuralNetwork`) based on the search space explored so far. Then, all the missing data-points are estimated based on this model. Following, the top `test_top_x_configurations` configurations are tested on the actual device. Training a model is only useful if a fraction of the search space is explored, as is the case when doing for example random-search.

Output
-------------

* `void OutputSearchLog(const std::string &filename)`:
Outputs the search process to the file `filename`.

* `double PrintToScreen() const`:
Prints the results of the tuning to screen (stdout). Returns the best-case execution time in milliseconds.

* `void PrintFormatted() const`:
Prints the results of the tuning to screen as a formatted table (stdout).

* `void PrintJSON(const std::string &filename, const std::vector<std::pair<std::string,std::string>> &descriptions) const`:
Prints the results of the tuning to the file `filename` in JSON format. Additional key-value input can be given as a vector of pairs through the `descriptions` argument.

* `void PrintToFile(const std::string &filename) const`:
Prints the results of the tuning to the file `filename` in plain text format.

* `void SuppressOutput()`:
Disables all further printing to screen (stdout).

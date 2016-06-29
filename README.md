
CLTune: Automatic OpenCL kernel tuning
================

| | master | development |
|-----|-----|-----|
| Linux/OS X | [![Build Status](https://travis-ci.org/CNugteren/CLTune.svg?branch=master)](https://travis-ci.org/CNugteren/CLTune/branches) | [![Build Status](https://travis-ci.org/CNugteren/CLTune.svg?branch=development)](https://travis-ci.org/CNugteren/CLTune/branches) |
| Windows | [![Build Status](https://ci.appveyor.com/api/projects/status/github/cnugteren/cltune?branch=master&svg=true)](https://ci.appveyor.com/project/CNugteren/cltune) | [![Build Status](https://ci.appveyor.com/api/projects/status/github/cnugteren/cltune?branch=development&svg=true)](https://ci.appveyor.com/project/CNugteren/cltune) |

CLTune is a C++ library which can be used to automatically tune your OpenCL and CUDA kernels. The only thing you'll need to provide is a tuneable kernel and a list of allowed parameters and values.

For example, if you would perform loop unrolling or local memory tiling through a pre-processor define, just remove the define from your kernel code, pass the kernel to CLTune and tell it what the name of your parameter(s) are and what values you want to try. CLTune will take care of the rest: it will iterate over all possible permutations, test them, and report the best combination.


Compilation
-------------

CLTune can be compiled as a shared library using CMake. The pre-requisites are:

* CMake version 2.8.10 or higher
* A C++11 compiler, for example:
  - GCC 4.7.0 or newer
  - Clang 3.3 or newer
  - AppleClang 5.0 or newer
  - ICC 14.0 or newer
  - MSVC (Visual Studio) 2015 or newer
* An OpenCL library. CLTune has been tested with:
  - Apple OpenCL
  - NVIDIA CUDA SDK (requires version 7.5 or newer for the CUDA back-end)
  - AMD APP SDK
  - Intel OpenCL
  - Beignet

An example of an out-of-source build (starting from the root of the CLTune folder):

    mkdir build
    cd build
    cmake ..
    make
    sudo make install

A custom installation folder can be specified when calling CMake:

    cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/directory ..

You can then link your own programs against the CLTune library. An example for a Linux-system with OpenCL:

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libcltune.so
    g++ example.cc -o example -L/path/to/libcltune.so -lcltune -lOpenCL


Example of using the tuner
-------------

Before we start using the tuner, we'll have to create one. The constructor takes two arguments: the first specifying the OpenCL platform number, and the second the device ID on that platform:

    cltune::Tuner my_tuner(0, 1); // Tuner on device 1 of OpenCL platform 0

For the CUDA back-end use 0 as the platform ID. Now that we have a tuner, we can add a tuning kernel. This is done by providing a list of paths to kernel files (first argument), the name of the kernel (second argument), a list of global thread dimensions (third argument), and a list of local thread or workgroup dimensions (fourth argument). Note that the thread configuration can be dynamic as well, see the included samples. Here is an example of a more basic usage using a static configuration:

    size_t id = my_tuner.AddKernel({"path/to/kernel.opencl"}, "my_kernel", {1024,512}, {16,8});

Notice that the AddKernel function returns an integer: it is the ID of the added kernel. We'll need this ID when we want to add tuning parameters to this kernel. Let's say that our kernel has two pre-processor parameters named `PARAM_1` with allowed values 16 and 24 and `PARAM_2` with allowed values 0 through 4:

    my_tuner.AddParameter(id, "PARAM_1", {16, 24});
    my_tuner.AddParameter(id, "PARAM_2", {0, 1, 2, 3, 4});

Now that we've added a kernel and its parameters, we can add another one if we wish. When we're satisfied with the kernels and their parameters, there are a couple of things left to be done. Let's start by adding a reference kernel. This reference kernel can provide the tuner with the ground-truth and is optional - the tuner will only perform verification checks to ensure correctness when it is provided.

    my_tuner.SetReference({"path/to/reference.opencl"}, "my_reference", {8192}, {128});

The tuner also needs to know which arguments the kernels take. Scalar arguments can be provided as-is and are passed-by-value, whereas arrays have to be provided as C++ `std::vector`s. That's right, you don't have to create device buffers yourself, CLTune will handle that! Here is an example:

    int my_variable = 900;
    std::vector<float> input_vector(8192);
    std::vector<float> output_vector(8192);
    my_tuner.AddArgumentScalar(my_variable);
    my_tuner.AddArgumentScalar(3.7);
    my_tuner.AddArgumentInput(input_vector);
    my_tuner.AddArgumentOutput(output_vector);

Now that we've configured the tuner, it is time to start it and ask it to report the results:

    my_tuner.Tune(); // Starts the tuner
    my_tuner.PrintToScreen(); // Prints the results


Other examples
-------------

Several examples are included as part of the CLTune distribution. They illustrate based and more advanced features, such as modifying the thread dimensions based on the parameters and adding user-defined parameter constraints. The examples are compiled when setting `ENABLE_SAMPLES` to `ON` in CMake (default option). All examples have both CUDA and OpenCL kernels. The included examples are:

* `simple.cc` The simplest possible example: tuning the work-group/thread-block size of a vector-addition kernel.
* `conv_simple.cc` A simple example of a 2D convolution kernel.
* `multiple_kernels.cc` A simple example with two different matrix-vector multiplication kernels, also showing the verification of output data.
* `gemm.cc` An advanced and heavily tunable implementation of matrix-matrix multiplication (GEMM).
* `conv.cc` An advanced and heavily tunable implementation of 2D convolution. This also demonstrates advanced search strategies including machine learning models.

The latter two optionally take command-line arguments. The first argument is an integer to select the platform (NVIDIA, AMD, etc.), the second argument is an integer for the device to run on, the third argument is an integer to select a search strategy (0=random, 1=annealing, 2=PSO, 3=fullsearch), and the fourth an optional search-strategy parameter.

Other examples are found in the [CLTuneDemos repository](https://github.com/williamjshipman/CLTuneDemos). CLTune is also used in the [CLBlast library](https://github.com/CNugteren/CLBlast).


Search strategies and machine-learning
-------------

The GEMM and 2D convolution examples are additionally configured to use one of the four supported search strategies. More details can be found in the corresponding CLTune paper (see below). These search-strategies can be used for any example as follows:

    tuner.UseFullSearch(); // Default
    tuner.UseRandomSearch(double fraction);
    tuner.UseAnnealing(double fraction, double max_temperature);
    tuner.UsePSO(double fraction, size_t swarm_size, double influence_global, double influence_local, double influence_random);

The 2D convolution example is additionally configured to use machine-learning to predict the quality of parameters based on a limited set of 'training' data. The supported models are linear regression and a 3-layer neural network. These machine-learning models are still experimental, but can be used as follows:

    // Trains a machine learning model based on the search space explored so far. Then, all the
    // missing data-points are estimated based on this model. This is only useful if a fraction of
    // the search space is explored, as is the case when doing random-search.
    tuner.ModelPrediction(Model model_type, float validation_fraction, size_t test_top_x_configurations);


Experimental CUDA support
-------------

CLTune was originally developed for OpenCL kernels, but since it uses the high-level C++ API `CLCudaAPI`, it can also work with CUDA kernels. To compile CLTune with CUDA as a back-end, set the `USE_OPENCL` CMake flag to `OFF`, for example as follows:

    cmake -DUSE_OPENCL=OFF ..

The samples ship with a basic header to convert the included OpenCL samples to CUDA (`cl_to_cuda.h`). This header file is automatically included when CLTune is built with CUDA as a back-end. It has been tested with the `simple` example, but doesn't work with the more advanced kernels. Nevertheless, CLTune should work with any proper CUDA kernel.


Development and tests
-------------

The CLTune project follows the Google C++ styleguide (with some exceptions) and uses a tab-size of two spaces and a max-width of 100 characters per line. It is furthermore based on practises from the third edition of Effective C++ and the first edition of Effective Modern C++. The project is licensed under the APACHE 2.0 license by SURFsara, (c) 2014.

CLTune is packaged with Catch 1.2.1 and a custom test suite. No external dependencies are needed. The tests will be compiled when providing the `TESTS=ON` option to CMake. Running the tests goes as follows:

    ./unit_tests

However, the more useful tests are the provided examples, since they include a verification kernel. Running the examples on device Y on platform X goes as follows:

    ./sample_conv X Y
    ./sample_gemm X Y


More information
-------------

Further information on CLTune is available below:

* The full [CLTune API reference](doc/api.md) is available in the current repository.

* A 19-minute presentation of CLTune was given at the GPU Technology Conference in April 2016. A recording is available on [the GTC on-demand website](http://on-demand.gputechconf.com/gtc/2016/video/S6206.html) and a full slideset is [also available as PDF](http://www.cedricnugteren.nl/downloads/handouts2016a.pdf).

* A how-to-use CLTune tutorial written by William J Shipman is available on [his blog](https://williamjshipman.wordpress.com/2016/01/31/autotuning-opencl-kernels-cltune-on-windows-7/).

* More in-depth information and experimental results are also available in a scientific paper. If you refer to this work in a scientific publication, please cite the corresponding CLTune paper published in MCSoC '15:

> Cedric Nugteren and Valeriu Codreanu. CLTune: A Generic Auto-Tuner for OpenCL Kernels. In: MCSoC: 9th International Symposium on Embedded Multicore/Many-core Systems-on-Chip. IEEE, 2015.


Related projects
-------------

A simpler but perhaps easier-to-use Python-based OpenCL auto-tuner was made by Ben van Werkhoven and is [also available on GitHub](https://github.com/benvanwerkhoven/kernel_tuner).

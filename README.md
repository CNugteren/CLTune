
CLTune: Automatic OpenCL kernel tuning
================

CLTune is a C++ library which can be used to automatically tune your OpenCL kernels. The only thing you'll need to provide is a tuneable kernel and a list of allowed parameters and values.

For example, if you would perform loop unrolling or local memory tiling through a pre-processor define, just remove the define from your kernel code, pass the kernel to CLTune and tell it what the name of your parameter(s) are and what values you want to try. CLTune will take care of the rest: it will iterate over all possible permutations, test them, and report the best combination.


Compilation
-------------

CLTune can be compiled as a shared library using CMake. The pre-requisites are:

* CMake version 2.8 or higher
* A C++11 compiler, for example:
  - GCC 4.9.0 or newer
  - Clang 3.3 or newer
  - ICC 14.0 or newer
* An OpenCL library. CLTune has been tested with:
  - Apple OpenCL
  - NVIDIA CUDA SDK
  - AMD APP SDK

An example of an out-of-source build (starting from the root of the cltune folder):

    mkdir build
    cd build
    cmake ..
    make
    sudo make install

A custom installation folder can be specified when calling CMake:

    cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/directory ..

You can then link your own programs against the CLTune library. An example for a Linux-system:

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libcltune.so
    g++ example.cc -o example -L/path/to/libcltune.so -lcltune -lOpenCL


Example of using the tuner
-------------

Before we start using the tuner, we'll have to create one. The constructor takes two arguments: the first specifying the OpenCL platform number, and the second the device ID on that platform:

    cltune::Tuner my_tuner(0, 1); // Tuner on device 1 of OpenCL platform 0

Now that we have a tuner, we can add a tuning kernel. This is done by providing the path to an OpenCL kernel (first argument), the name of the kernel (second argument), a list of global thread dimensions (third argument), and a list of local thread or workgroup dimensions (fourth argument). Here is an example:

    int id = my_tuner.AddKernel("path/to/kernel.opencl", "my_kernel", {1024,512}, {16,8});

Notice that the AddKernel function returns an integer: it is the ID of the added kernel. We'll need this ID when we want to add tuning parameters to this kernel. Let's say that our kernel has two pre-processor parameters named `PARAM_1` and `PARAM_2`:

    my_tuner.AddParameter(id, "PARAM_1", {16, 24});
    my_tuner.AddParameter(id, "PARAM_2", {0, 1, 2, 3, 4});

Now that we've added a kernel and its parameters, we can add another one if we wish. When we're done, there are a couple of things left to be done. Let's start with adding an reference kernel. This reference kernel can provide the tuner with the ground-truth and is optional - only when it is provided will the tuner perform verification checks to ensure correctness.

    my_tuner.SetReference("path/to/reference.opencl", "my_reference", {8192}, {128});

The tuner also needs to know which arguments the kernels take. Scalar arguments can be provided as-is and are passed-by-value, whereas arrays have to be provided as C++ `std::vector`s. That's right, we won't have to create OpenCL buffers, CLTune will handle that for us! Here is an example:

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

Examples are included as part of the CLTune distribution. They illustrate some more advanced features, such as modifying the thread dimensions based on the parameters and adding user-defined parameter constraints. The examples are compiled when providing `-ENABLE_SAMPLES=ON` to CMake (default option). The included examples are:

* `simple.cc` providing a basic example of matrix-vector multiplication
* `gemm.cc` providing an advanced and heavily tuned implementation of matrix-matrix
  multiplication (GEMM)
* `conv.cc` providing an advanced and heavily tuned implementation of 2D convolution

The latter two take optionally command-line arguments. The first argument is an integer for the device to run on, the second argument is an integer to select a search strategy (0=random, 1=annealing, 2=PSO, 3=fullsearch), and the third an optional search-strategy parameter.

Development and tests
-------------

The CLTune project follows the Google C++ styleguide (with some exceptions) and uses a tab-size of
two spaces and a max-width of 100 characters per line. It is furthermore based on practises from the
third edition of Effective C++ and the first edition of Effective Modern C++. The project is
licensed under the MIT license by SURFsara, (c) 2014. The contributing authors so far are:

* Cedric Nugteren

CLTune is packaged with Google Test 1.7.0 and a custom test suite. The tests will be compiled when
providing the `-TESTS=ON` option to CMake. Running the tests goes as follows:

    ./unit_tests

Other useful tests are the provided examples, since they include a verification kernel:

    ./sample_conv 0 0
    ./sample_gemm 0 0

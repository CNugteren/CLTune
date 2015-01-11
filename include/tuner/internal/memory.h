
// =================================================================================================
// This file is part of the CLTune project. The project is licensed under the MIT license by
// SURFsara, (c) 2014.
//
// The CLTune project follows the Google C++ styleguide and uses a tab-size of two spaces and a
// max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the OpenCL Memory class, a container for both host and device data. The
// device data is based on the OpenCL C++ API and the cl::Buffer class, while the host data is based
// on the std::vector class. The Memory class is templated to support different types.
//
// =================================================================================================

#ifndef CLBLAS_TUNER_MEMORY_H_
#define CLBLAS_TUNER_MEMORY_H_

#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

// The C++ OpenCL wrapper
#include "cl.hpp"

namespace cltune {
// =================================================================================================

// Enumeration of currently supported data-types by this class
enum MemType { kInt, kFloat, kDouble };

// OpenCL-related exception, prints not only a message but also an OpenCL error code. This class is
// added to this file because it is only used by the Memory class.
class OpenCLException : public std::runtime_error {
 public:
  OpenCLException(const std::string &message, cl_int status)
                  : std::runtime_error(message+
                  " [code: "+std::to_string(static_cast<long long>(status))+"]") { };
};

// See comment at top of file for a description of the class
template <typename T>
class Memory {
 public:

  // Static variable to get the memory type based on a template argument
  const static MemType type;

  // Initializes the host and device data (with zeroes or based on a source-vector)
  explicit Memory(const size_t size, cl::Context context, cl::CommandQueue queue);
  explicit Memory(const size_t size, cl::Context context, cl::CommandQueue queue,
                  std::vector<T> &source);

  // Accessors to the host/device data
  std::vector<T> host() const { return host_; }
  std::shared_ptr<cl::Buffer> device() const { return device_; }

  // Downloads the device data onto the host
  void UploadToDevice();
  void DownloadFromDevice();

 private:

  // The data (both host and device)
  const size_t size_;
  std::vector<T> host_;
  std::shared_ptr<cl::Buffer> device_;

  // Pointers to the memory's context and command queue
  // TODO: Pass these objects by reference instead of creating copies
  cl::Context context_;
  cl::CommandQueue queue_;
};


// =================================================================================================
} // namespace cltune

// CLBLAS_TUNER_MEMORY_H_
#endif

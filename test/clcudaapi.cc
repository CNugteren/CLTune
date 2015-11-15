
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the basic OpenCL/CUDA tests based on the CLCudaAPI.
//
// =================================================================================================

#include "catch.hpp"

// Runs with either OpenCL or CUDA as a back-end
#if USE_OPENCL
  #include "internal/clpp11.h"
#else
  #include "internal/cupp11.h"
#endif

// Settings
const size_t kPlatformID = 0;
const size_t kDeviceID = 0;
const size_t kBufferSize = 10;

// =================================================================================================

SCENARIO("events can be created and used", "[Event]") {
  GIVEN("An example event") {
    #if !USE_OPENCL
    auto platform = cltune::Platform(kPlatformID);
    auto device = cltune::Device(platform, kDeviceID);
    auto context = cltune::Context(device);
    auto queue = cltune::Queue(context, device);
    #endif
    auto event = cltune::Event();

    #if USE_OPENCL // Not available for the CUDA version
    WHEN("its underlying data-structure is retrieved") {
      auto raw_event = event();
      THEN("a copy of this event can be created") {
        auto event_copy = cltune::Event(raw_event);
        REQUIRE(event_copy() == event());
      }
    }
    #else // Not available for the OpenCL version
    WHEN("its underlying data-structures are retrieved") {
      auto raw_start = event.start();
      auto raw_end = event.end();
      THEN("their underlying data-structures are not null") {
        REQUIRE(raw_start != nullptr);
        REQUIRE(raw_end != nullptr);
      }
    }
    #endif

    WHEN("a copy is created using the copy constructor") {
      auto event_copy = cltune::Event(event);
      THEN("its underlying data-structure is unchanged") {
        #if USE_OPENCL
          REQUIRE(event_copy() == event());
        #else
          REQUIRE(event_copy.start() == event.start());
          REQUIRE(event_copy.end() == event.end());
        #endif
      }
    }
  }
}

// =================================================================================================

SCENARIO("platforms can be created and used", "[Platform]") {
  GIVEN("An example platform") {
    auto platform = cltune::Platform(kPlatformID);
    auto num_devices = platform.NumDevices();

    #if USE_OPENCL // Not available for the CUDA version
    WHEN("its underlying data-structure is retrieved") {
      auto raw_platform = platform();
      THEN("a copy of this platform can be created") {
        auto platform_copy = cltune::Platform(raw_platform);
        REQUIRE(platform_copy.NumDevices() == num_devices);
      }
    }
    #endif

    WHEN("a copy is created using the copy constructor") {
      auto platform_copy = cltune::Platform(platform);
      THEN("the platform's properties remain unchanged") {
        REQUIRE(platform_copy.NumDevices() == num_devices);
      }
    }
  }
}

// =================================================================================================

SCENARIO("devices can be created and used", "[Device][Platform]") {
  GIVEN("An example device on a platform") {
    auto platform = cltune::Platform(kPlatformID);
    auto device = cltune::Device(platform, kDeviceID);

    GIVEN("...and device properties") {
      auto device_version = device.Version();
      auto device_vendor = device.Vendor();
      auto device_name = device.Name();
      auto device_type = device.Type();
      auto device_max_work_group_size = device.MaxWorkGroupSize();
      auto device_max_work_item_dimensions = device.MaxWorkItemDimensions();
      auto device_max_work_item_sizes = device.MaxWorkItemSizes();
      auto device_local_mem_size = device.LocalMemSize();
      auto device_capabilities = device.Capabilities();
      auto device_core_clock = device.CoreClock();
      auto device_compute_units = device.ComputeUnits();
      auto device_memory_size = device.MemorySize();
      auto device_max_alloc_size = device.MaxAllocSize();
      auto device_memory_clock = device.MemoryClock();
      auto device_memory_bus_width = device.MemoryBusWidth();

      // TODO: test for valid device properties

      WHEN("its underlying data-structure is retrieved") {
        auto raw_device = device();
        THEN("a copy of this device can be created") {
          auto device_copy = cltune::Device(raw_device);
          REQUIRE(device_copy.Name() == device_name); // Only verifying device name
        }
      }

      WHEN("a copy is created using the copy constructor") {
        auto device_copy = cltune::Device(device);
        THEN("the device's properties remain unchanged") {
          REQUIRE(device_copy.Name() == device_name); // Only verifying device name
        }
      }

      WHEN("the local memory size is tested") {
        THEN("the maximum local memory size should be considered valid") {
          REQUIRE(device.IsLocalMemoryValid(device_local_mem_size) == true);
        }
        THEN("more than the maximum local memory size should be considered invalid") {
          REQUIRE(device.IsLocalMemoryValid(device_local_mem_size+1) == false);
        }
      }

      WHEN("the local thread configuration is tested") {
        THEN("equal to the maximum size in one dimension should be considered valid") {
          REQUIRE(device.IsThreadConfigValid({device_max_work_item_sizes[0],1,1}) == true);
          REQUIRE(device.IsThreadConfigValid({1,device_max_work_item_sizes[1],1}) == true);
          REQUIRE(device.IsThreadConfigValid({1,1,device_max_work_item_sizes[2]}) == true);
        }
        THEN("more than the maximum size in one dimension should be considered invalid") {
          REQUIRE(device.IsThreadConfigValid({device_max_work_item_sizes[0]+1,1,1}) == false);
          REQUIRE(device.IsThreadConfigValid({1,device_max_work_item_sizes[1]+1,1}) == false);
          REQUIRE(device.IsThreadConfigValid({1,1,device_max_work_item_sizes[2]+1}) == false);
        }
      }
    }
  }
}

// =================================================================================================

SCENARIO("contexts can be created and used", "[Context][Device][Platform]") {
  GIVEN("An example context on a device") {
    auto platform = cltune::Platform(kPlatformID);
    auto device = cltune::Device(platform, kDeviceID);
    auto context = cltune::Context(device);

    WHEN("its underlying data-structure is retrieved") {
      auto raw_context = context();
      THEN("a copy of this context can be created") {
        auto context_copy = cltune::Context(raw_context);
        REQUIRE(context_copy() != nullptr);
      }
    }

    WHEN("a copy is created using the copy constructor") {
      auto context_copy = cltune::Context(context);
      THEN("its underlying data-structure is not null") {
        REQUIRE(context_copy() != nullptr);
      }
    }
  }
}

// =================================================================================================

SCENARIO("programs can be created and used", "[Program][Context][Device][Platform]") {
  GIVEN("An example program for a specific context and device") {
    auto platform = cltune::Platform(kPlatformID);
    auto device = cltune::Device(platform, kDeviceID);
    auto context = cltune::Context(device);
    auto source = std::string{""};
    auto program = cltune::Program(context, source);
  }
}

// =================================================================================================

SCENARIO("queues can be created and used", "[Queue][Context][Device][Platform][Event]") {
  GIVEN("An example queue associated to a context and device") {
    auto platform = cltune::Platform(kPlatformID);
    auto device = cltune::Device(platform, kDeviceID);
    auto context = cltune::Context(device);
    auto queue = cltune::Queue(context, device);

    #if USE_OPENCL // Not available for the CUDA version
    WHEN("its underlying data-structure is retrieved") {
      auto raw_queue = queue();
      THEN("a copy of this queue can be created") {
        auto queue_copy = cltune::Queue(raw_queue);
        REQUIRE(queue_copy() != nullptr);
      }
    }
    #endif

    WHEN("a copy is created using the copy constructor") {
      auto queue_copy = cltune::Queue(queue);
      THEN("its underlying data-structure is not null") {
        REQUIRE(queue_copy() != nullptr);
      }
    }

    WHEN("the associated context is retrieved") {
      auto context_copy = queue.GetContext();
      THEN("their underlying data-structures match") {
        REQUIRE(context_copy() == context());
      }
    }
    WHEN("the associated device is retrieved") {
      auto device_copy = queue.GetDevice();
      THEN("their underlying data-structures match") {
        REQUIRE(device_copy() == device());
      }
    }

    WHEN("the queue is synchronised") {
      queue.Finish();
      THEN("its underlying data-structure is not null") {
        REQUIRE(queue() != nullptr);
      }
    }
    WHEN("the queue is synchronised using an event") {
      auto event = cltune::Event();
      queue.Finish(event);
      THEN("its underlying data-structure is not null") {
        REQUIRE(queue() != nullptr);
      }
    }
  }
}

// =================================================================================================

SCENARIO("host buffers can be created and used", "[BufferHost][Context][Device][Platform]") {
  GIVEN("An example host buffer for a specific context and device") {
    auto platform = cltune::Platform(kPlatformID);
    auto device = cltune::Device(platform, kDeviceID);
    auto context = cltune::Context(device);
    auto size = static_cast<size_t>(kBufferSize);
    auto buffer_host = cltune::BufferHost<float>(context, size);
  }
}

// =================================================================================================

SCENARIO("device buffers can be created and used", "[Buffer][Context][Device][Platform]") {
  GIVEN("An example device buffer for a specific context and device") {
    auto platform = cltune::Platform(kPlatformID);
    auto device = cltune::Device(platform, kDeviceID);
    auto context = cltune::Context(device);
    auto size = static_cast<size_t>(kBufferSize);
    auto buffer = cltune::Buffer<float>(context, size);
  }
}

// =================================================================================================

SCENARIO("kernels can be created and used", "[Kernel][Program][Context][Device][Platform]") {
  GIVEN("An example device buffer for a specific context and device") {
    auto platform = cltune::Platform(kPlatformID);
    auto device = cltune::Device(platform, kDeviceID);
    auto context = cltune::Context(device);
    auto source = std::string{""};
    auto program = cltune::Program(context, source);
    auto name = std::string{""};
    //auto kernel = cltune::Kernel(program, name);
  }
}

// =================================================================================================

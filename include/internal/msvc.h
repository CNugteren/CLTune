
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides macro's and definitions to make compilation work on Microsoft Visual Studio,
// in particular for versions older than 2015 with limited C++11 support.
// MSVC++ 14.0 _MSC_VER == 1900 (Visual Studio 2015)
// MSVC++ 12.0 _MSC_VER == 1800 (Visual Studio 2013)
// MSVC++ 11.0 _MSC_VER == 1700 (Visual Studio 2012)
// MSVC++ 10.0 _MSC_VER == 1600 (Visual Studio 2010)
// MSVC++ 9.0  _MSC_VER == 1500 (Visual Studio 2008)
//
// =================================================================================================

#ifndef CLTUNE_MSVC_H_
#define CLTUNE_MSVC_H_

namespace cltune {
// =================================================================================================
#ifdef _MSC_VER

// No support for constexpr prior to 2015. Note that this only works with constants, not with
// constexpr functions (unused in this project).
#if _MSC_VER < 1900
#define constexpr const
#endif

// _MSC_VER
#endif
// =================================================================================================
} // namespace cltune

// CLTUNE_MSVC_H_
#endif

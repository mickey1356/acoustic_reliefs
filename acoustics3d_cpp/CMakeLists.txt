cmake_minimum_required(VERSION 3.16)
project(acoustics3d_impl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

find_package(OpenMP)

include_directories(include
                    extern/eigen
)

set(SRC_FILES
    src/aca.cpp
    src/adjoint.cpp
    src/clustertree.cpp
    src/derivatives.cpp
    src/helpers.cpp
    src/hmatrix.cpp
    src/integrate.cpp
    src/read_obj.cpp
)

# for nanobind
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
  OPTIONAL_COMPONENTS Development.SABIModule
)  
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(acoustics3d_impl NOMINSIZE ${SRC_FILES} src/acoustics3d.cpp)
target_link_libraries(acoustics3d_impl PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(acoustics3d_impl PRIVATE include extern/eigen)
install(TARGETS acoustics3d_impl LIBRARY DESTINATION acoustics3d)
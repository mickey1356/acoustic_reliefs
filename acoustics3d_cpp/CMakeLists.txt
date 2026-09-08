cmake_minimum_required(VERSION 3.16)
project(acoustics3d_impl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optional CUDA GPU backend. Default OFF -> pure-CPU build is completely unchanged.
# Enable with:
#   pip install --no-build-isolation -ve . \
#     -C cmake.define.BEM_USE_CUDA=ON \
#     -C cmake.define.CUDAToolkit_ROOT=/usr/local/cuda-12.4 \
#     -C cmake.define.CMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
#     -C cmake.define.CMAKE_CUDA_ARCHITECTURES=89
# (wipe build/ when toggling these; scikit-build-core caches CMakeCache)
option(BEM_USE_CUDA "Build CUDA GPU kernels" OFF)
if (BEM_USE_CUDA)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

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
    src/profiler.cpp
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

# CUDA sources are only compiled when BEM_USE_CUDA is ON.
set(CUDA_SRC_FILES
    src/cuda/device_hello.cu
    src/cuda/device_kernels.cu
)

nanobind_add_module(acoustics3d_impl NOMINSIZE
    ${SRC_FILES} src/acoustics3d.cpp
    $<$<BOOL:${BEM_USE_CUDA}>:${CUDA_SRC_FILES}>)
target_link_libraries(acoustics3d_impl PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(acoustics3d_impl PRIVATE include extern/eigen)

# Performance flags. -O3/-funroll-loops/-fno-math-errno are value-preserving;
# -march=native enables SIMD (and FMA). To keep results bit-for-bit identical to
# the previous (no-march) build we disable FP contraction by default via
# BEM_STRICT_FP. Turn it OFF to allow FMA contraction (faster, tiny last-bit drift):
#   pip install --no-build-isolation -ve . -C cmake.define.BEM_STRICT_FP=OFF
# NOTE: -march=native bakes in the build host's ISA; fine for local editable
# installs, not for redistributed wheels.
# The flags are gated on COMPILE_LANGUAGE:CXX so nvcc never receives -march=native.
option(BEM_STRICT_FP "Disable FP contraction so results bit-match the reference build" ON)
if (NOT MSVC)
  target_compile_options(acoustics3d_impl PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -march=native -funroll-loops -fno-math-errno>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-O3 -march=native -funroll-loops -fno-math-errno>)
  if (BEM_STRICT_FP)
    target_compile_options(acoustics3d_impl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endif()

if (BEM_USE_CUDA)
  target_link_libraries(acoustics3d_impl PRIVATE CUDA::cudart CUDA::cublas)
  target_compile_definitions(acoustics3d_impl PRIVATE BEM_USE_CUDA=1)
  set_target_properties(acoustics3d_impl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # Prefer the 12.4 toolkit's cudart over the conda env's 12.1 at load time.
  set_target_properties(acoustics3d_impl PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")
  target_compile_options(acoustics3d_impl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
endif()

install(TARGETS acoustics3d_impl LIBRARY DESTINATION acoustics3d)
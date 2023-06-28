cmake_minimum_required(VERSION 3.14)
project(laplace LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

# Add source files for laplace-cuda excecutable
set(SOURCE_FILES
   laplace-cuda.cu
   reduce-max.cu
   main-cuda.cu)


add_executable(laplace-gpu.x ${SOURCE_FILES})

target_compile_features(laplace-gpu.x PRIVATE cxx_std_17)
set_target_properties(laplace-gpu.x PROPERTIES CXX_EXTENSIONS OFF)

target_link_libraries(laplace-gpu.x PRIVATE OpenMP::OpenMP_CXX)

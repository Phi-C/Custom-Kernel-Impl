cmake_minimum_required(VERSION 3.10)
project(CustomKernelLib LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/utils)

set(SOURCE_DIRS
    prefix_sum
    non_zero_copy
    sort
)

# Collect all CUDA source files from the specified directories
set(CUDA_SOURCES)
foreach(dir ${SOURCE_DIRS})
    file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/${dir}/*.cu")
    list(APPEND CUDA_SOURCES ${SOURCES})
endforeach()


# Helper function to add a CUDA executable
# This function takes a source file and creates an executable
function(add_cuda_executable source_file)
    # source_file: /path/to/source/file.cu
    # executable_name: file
    # source_dir: /path/to/source
    if(NOT EXISTS ${source_file})
        message(FATAL_ERROR "Source file does not exist: ${source_file}")
    endif()
    get_filename_component(executable_name ${source_file} NAME_WE)
    get_filename_component(source_dir ${source_file} DIRECTORY)
    add_executalbe(${executable_name} ${source_file})

    target_include_directories(${executable_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/utils
        ${source_dir}
    )

    set_target_properties(${executable_name} PROPERTIES
        CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )

    target_compile_options(${executable_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        >
    )

endfunction()


foreach(cuda_source ${CUDA_SOURCES})
    add_cuda_executable(${cuda_source})
endforeach()


cmake_minimum_required(VERSION 3.20)
project(mpi_workload LANGUAGES C CXX)

find_package(MPI REQUIRED COMPONENTS C)

# handle tpls
set(LOGURU_STACKTRACES OFF)
add_subdirectory(tpl/loguru)

add_subdirectory(tpl/cxxopts)

set(FMT_INSTALL OFF)
add_subdirectory(tpl/fmt)

add_executable(mpi_workload main.cpp)
target_link_libraries(mpi_workload
 PRIVATE
    MPI::MPI_C
    loguru
    cxxopts
    fmt::fmt)

install(TARGETS mpi_workload
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static)

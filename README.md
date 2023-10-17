# MPI-Workload: Sample MPI Workload

This is a sample MPI workload that can be used to test the MPI cluster or emulate
a real workload.

## Building

The project uses CMake to generate the build files. The following commands can be
used to build the project:

```bash
# requires CMake 3.20 or newer

cd mpi-workload

# create a build directory and generate the build files
cmake -b build

# build the project
cmake --build build
```

## Usage

```bash
cd build 

# supported options
./mpi-workload --help
# >output
MPI Workload
Usage:
  mpi_workload [OPTION...]

  -h, --help          Print help
  -t, --threads arg   Number of threads to spawn on each rank (default: 1)
  -b, --buffer arg    Buffer size in KBs (default: 1)
  -i, --interval arg  Interval in seconds (default: 10)
```

Currently only `MPI_Alltoall` based workload is supported. This may be extended
in the future as needed.

## Third-party libraries

The codebase uses the following open source libraries (each added as submodules):

* [Loguru](https://github.com/emilk/loguru)
* [libfmt](https://github.com/fmtlib/fmt)
* [cxxopts](https://github.com/jarro2783/cxxopts)

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT License](./LICENSE)

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

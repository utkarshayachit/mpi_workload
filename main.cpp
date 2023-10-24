// Copyright (c) Microsoft Corporation
// Licensed under the MIT License.

// #define LOGURU_USE_FMTLIB 1

#include <chrono>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include <loguru.hpp>
#include <mpi.h>
#include <thread>
#include <vector>

namespace
{
void thread_func(const int rank, const int thread_id, const int num_ranks, const int num_threads,
  const size_t buffer_size, const int interval, MPI_Comm comm)
{
  loguru::set_thread_name(fmt::format("rank {:03d} [{:03d}]", rank, thread_id).c_str());
  LOG_F(1, "Thread %d:%d started", rank, thread_id);

  size_t num_ints = buffer_size / sizeof(int);
  LOG_F(1, "buffer size %zu KB", buffer_size / 1024);
  LOG_F(1, "num ints %zu", num_ints);
  const auto start_time = std::chrono::system_clock::now();
  auto last_progress_time = std::chrono::system_clock::now();

  // do some work
  while (true)
  {
    /* code */
    const auto end_time = std::chrono::system_clock::now();
    const auto elapsed_time =
      std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    if (elapsed_time.count() > interval)
    {
      break;
    }

    // send to all other ranks
    std::vector<int> send_buffer(num_ints * num_ranks, rank);
    std::vector<int> recv_buffer(num_ints * num_ranks, -1);
    const auto status =
      MPI_Alltoall(send_buffer.data(), num_ints, MPI_INT, recv_buffer.data(), num_ints, MPI_INT, comm);
    if (status != MPI_SUCCESS)
    {
      LOG_F(ERROR, "MPI_Alltoall failed with status %d", status);
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (rank == 0 && thread_id == 0)
    {
      const auto progress_elapsed_time =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - last_progress_time);
      if (progress_elapsed_time.count() >= 1)
      {
        last_progress_time = end_time;
        fmt::print(".");
        std::fflush(stdout);
      }
    }
  }
  fmt::print("\n");
  LOG_F(1, "Thread %d:%d finished", rank, thread_id);
  MPI_Barrier(comm);
}
}

int main(int argc, char* argv[])
{
  loguru::g_internal_verbosity = loguru::Verbosity_1;
  loguru::g_preamble_header = false;
  loguru::init(argc, argv);

  // parse command line arguments
  cxxopts::Options options("mpi_workload", "MPI Workload");

  options.add_options()("h,help", "Print help")("t,threads",
    "Number of threads to spawn on each rank", cxxopts::value<int>()->default_value("1"))(
    "b,buffer", "Buffer size in KBs", cxxopts::value<size_t>()->default_value("1"))(
    "i,interval", "Interval in seconds", cxxopts::value<int>()->default_value("10"));

  auto result = options.parse(argc, argv);
  const int num_threads = result["threads"].as<int>();
  const size_t buffer_size = result["buffer"].as<size_t>() * 1024;
  const int interval = result["interval"].as<int>();

  if (num_threads > 1)
  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
      LOG_F(ERROR, "MPI_THREAD_MULTIPLE not supported");
      return EXIT_FAILURE;
    }
  }
  else
  {
    MPI_Init(&argc, &argv);
  }

  // get rank and number of processes
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  loguru::set_thread_name(fmt::format("rank {:03d}", rank).c_str());

  if (result.count("help"))
  {
    if (rank == 0)
    {
      fmt::print(options.help());
    }
    return EXIT_SUCCESS;
  }

  if (rank == 0)
  {
    fmt::print("num_ranks: {}\n", size);
    fmt::print("buffer_size: {} KB\n", buffer_size / 1024);
    fmt::print("num_threads: {}\n", num_threads);
    fmt::print("interval: {} seconds\n", interval);
  }

  std::vector<std::thread> threads;
  std::vector<MPI_Comm> comms;
  for (int cc = 0; cc < num_threads; cc++)
  {
    // each thread will have its own communicator
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, /*color*/ 0, /*key*/ rank, &comm);
    threads.emplace_back(thread_func, rank, cc, size, num_threads, buffer_size, interval, comm);
    comms.push_back(std::move(comm));
  }

  for (auto& thread : threads)
  {
    thread.join();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  for (auto& comm : comms)
  {
    MPI_Comm_free(&comm);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return EXIT_SUCCESS;
}

// Copyright (c) Microsoft Corporation
// Licensed under the MIT License.

// #define LOGURU_USE_FMTLIB 1

#include <chrono>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include <limits>
#include <loguru.hpp>
#include <mpi.h>
#include <numeric>
#include <thread>
#include <vector>

namespace
{

void call_safe(int mpi_status)
{
  if (mpi_status != MPI_SUCCESS)
  {
    char error_string[MPI_MAX_ERROR_STRING];
    int length_of_error_string;
    MPI_Error_string(mpi_status, error_string, &length_of_error_string);
    LOG_F(ERROR, "MPI error: %s", error_string);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    abort();
  }
}

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
  auto last_summary_time = std::chrono::system_clock::now();

  double t = 0, t_min = std::numeric_limits<double>::max(), t_max = 0;
  std::uint64_t count = 0;

  auto print_summary = [&]()
  {
    if (rank == 0 && thread_id == 0)
    {
      fmt::print("\n");
      LOG_F(INFO, "Summary :");
      LOG_F(INFO, "  Average time per MPI_Alltoall: %f ms", t / count * 1000);
      LOG_F(INFO, "  Min time per MPI_Alltoall: %f ms", t_min * 1000);
      LOG_F(INFO, "  Max time per MPI_Alltoall: %f ms", t_max * 1000);
    }
  };

  // do some work
  while (true)
  {
    /* code */
    const auto end_time = std::chrono::system_clock::now();
    const auto elapsed_time =
      std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    int done = (rank == 0 && elapsed_time.count() > interval) ? 1 : 0;
    call_safe(MPI_Bcast(&done, 1, MPI_INT, 0, comm));
    if (done == 1)
    {
      break;
    }

    // send to all other ranks
    std::vector<int> send_buffer(num_ints * num_ranks, rank);
    std::vector<int> recv_buffer(num_ints * num_ranks, -1);

    const auto t_start = MPI_Wtime();
    const auto status =
      MPI_Alltoall(send_buffer.data(), num_ints, MPI_INT, recv_buffer.data(), num_ints, MPI_INT, comm);
    const auto t_end = MPI_Wtime();
    t += t_end - t_start;
    t_min = std::min(t_min, t_end - t_start);
    t_max = std::max(t_max, t_end - t_start);
    ++count;

    call_safe(status);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
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

      const auto summary_elapsed_time =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - last_summary_time);
      if (summary_elapsed_time.count() >= 60) // summarize every minute
      {
        last_summary_time = end_time;
        print_summary();
      }
    }
  }
  print_summary();

  call_safe(MPI_Barrier(comm));
  LOG_F(1, "Thread %d:%d finished", rank, thread_id);
}

class Options
{
public:
  Options(int argc, char* argv[], bool allow_unrecognised = false)
    : options_("mpi_workload", "Synthetic MPI Workload / Benchmark")
  {
    // clang-format off
    if (allow_unrecognised)
    {
      options_.allow_unrecognised_options();
    }
    options_.add_options()
      ("h,help", "Print help")
      ("t,threads", "Number of threads to spawn on each rank", cxxopts::value<int>()->default_value("1"))
      ("b,buffer", "Buffer size in KBs", cxxopts::value<int>()->default_value("1"))
      ("d,duration", "Test duration in seconds", cxxopts::value<int>()->default_value("10"));
    // clang-format on

    auto result = options_.parse(argc, argv);
    num_threads_ = std::max(1, result["threads"].as<int>());
    buffer_size_ = std::max(1, result["buffer"].as<int>()) * 1024;
    duration_ = std::max(1, result["duration"].as<int>());
    do_help_ = result.count("help") > 0;
  }

  int num_threads() const { return num_threads_; }
  size_t buffer_size() const { return buffer_size_; }
  int duration() const { return duration_; }
  bool do_help() const { return do_help_; }
  std::string help() const { return options_.help(); }
  void print() const
  {
    LOG_F(INFO, "Options:");
    LOG_F(INFO, "  num_threads: %d", num_threads_);
    LOG_F(INFO, "  buffer_size: %zu KB", buffer_size_ / 1024);
    LOG_F(INFO, "  duration: %d seconds", duration_);
  }

private:
  cxxopts::Options options_;
  int num_threads_;
  size_t buffer_size_;
  int duration_;
  bool do_help_;
};

} // end of anonymous namespace

int main(int argc, char* argv[])
{
  // initialize MPI
  Options options_tmp(argc, argv, true);
  if (options_tmp.num_threads() > 1)
  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
      ABORT_F("MPI_THREAD_MULTIPLE not supported");
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

  if (options_tmp.do_help())
  {
    if (rank == 0)
    {
      fmt::print(options_tmp.help());
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  // initalize loguru
  loguru::set_thread_name(fmt::format("rank {:03d}/main", rank).c_str());
  loguru::g_internal_verbosity = rank == 0 ? loguru::Verbosity_INFO : loguru::Verbosity_9;
  loguru::g_preamble_header = (rank == 0);
  loguru::Options loptions;
  loptions.main_thread_name = nullptr;
  loguru::init(argc, argv, loptions);

  // parse command line arguments
  Options options(argc, argv);

  if (rank == 0)
  {
    options.print();
  }

  std::vector<std::thread> threads;
  std::vector<MPI_Comm> comms;
  for (int cc = 0; cc < options.num_threads(); cc++)
  {
    // each thread will have its own communicator
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, /*color*/ 0, /*key*/ rank, &comm);
    threads.emplace_back(thread_func, rank, cc, size, options.num_threads(), options.buffer_size(),
      options.duration(), comm);
    comms.push_back(std::move(comm));
  }

  for (auto& thread : threads)
  {
    thread.join();
  }
  call_safe(MPI_Barrier(MPI_COMM_WORLD));

  for (auto& comm : comms)
  {
    call_safe(MPI_Comm_free(&comm));
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}

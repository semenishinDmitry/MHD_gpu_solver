#pragma once

// Lightweight persistent thread pool used by the CPU kernels.
//
// * No OpenMP dependency (Apple Clang ships without libomp; MSVC's OpenMP is 2.0).
// * Dynamic chunk scheduling: idle threads grab the next chunk from an atomic counter, which
//   balances work across heterogeneous cores (Apple P/E cores, Intel P/E cores).
// * Kernels partition work by grid rows or by vector-width-aligned index blocks, so results are
//   bitwise identical for any thread count.
//
// Configuration (process-wide):
//   mhd::set_num_threads(n)       — 0 = hardware_concurrency (default, or $MHD_NUM_THREADS)
//   mhd::set_simd_enabled(bool)   — false forces the scalar instantiation of the kernels
//   mhd::set_parallel_min_cells(n) — grids smaller than this run single-threaded

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace mhd {

class ThreadPool {
  public:
    explicit ThreadPool(int n_threads) : n_(std::max(1, n_threads))
    {
        workers_.reserve(static_cast<std::size_t>(n_ - 1));
        for (int tid = 1; tid < n_; ++tid) {
            workers_.emplace_back([this, tid] { worker_loop(tid); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            stop_ = true;
            generation_.fetch_add(1, std::memory_order_release);
        }
        start_cv_.notify_all();
        for (auto& t : workers_) {
            t.join();
        }
    }

    int size() const
    {
        return n_;
    }

    // Calls f(chunk_begin, chunk_end, thread_id) over [begin, end) in chunks of `grain`.
    // thread_id is in [0, size()). The calling thread participates as thread 0.
    template <class F>
    void parallel_for(int begin, int end, int grain, F&& f)
    {
        if (end <= begin) {
            return;
        }
        grain = std::max(1, grain);
        if (n_ == 1 || end - begin <= grain || in_worker()) {
            f(begin, end, 0);
            return;
        }

        // One job at a time; concurrent callers (e.g. two solvers in two Python threads) queue.
        std::lock_guard<std::mutex> submit(submit_mutex_);

        using Fn = std::remove_reference_t<F>;
        Job job;
        job.end = end;
        job.grain = grain;
        job.ctx = static_cast<void*>(&f);
        job.call = [](void* ctx, int b, int e, int tid) { (*static_cast<Fn*>(ctx))(b, e, tid); };
        next_.store(begin, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(mutex_);
            job_ = &job;
            pending_ = n_ - 1;
            generation_.fetch_add(1, std::memory_order_release);
        }
        start_cv_.notify_all();

        in_worker() = true;
        run_chunks(job, 0);
        in_worker() = false;

        {
            std::unique_lock<std::mutex> lk(mutex_);
            done_cv_.wait(lk, [this] { return pending_ == 0; });
            job_ = nullptr;
        }
        if (job.error) {
            std::rethrow_exception(job.error);
        }
    }

  private:
    struct Job {
        int end = 0;
        int grain = 1;
        void* ctx = nullptr;
        void (*call)(void*, int, int, int) = nullptr;
        std::exception_ptr error;
        std::mutex error_mutex;
    };

    static bool& in_worker()
    {
        thread_local bool flag = false;
        return flag;
    }

    void run_chunks(Job& job, int tid)
    {
        for (;;) {
            const int b = next_.fetch_add(job.grain, std::memory_order_relaxed);
            if (b >= job.end) {
                return;
            }
            const int e = std::min(b + job.grain, job.end);
            try {
                job.call(job.ctx, b, e, tid);
            } catch (...) {
                std::lock_guard<std::mutex> lk(job.error_mutex);
                if (!job.error) {
                    job.error = std::current_exception();
                }
            }
        }
    }

    void worker_loop(int tid)
    {
        in_worker() = true;
        std::uint64_t seen = 0;
        for (;;) {
            // Short spin before sleeping: RK stages launch several jobs back to back, and a
            // condition-variable wakeup costs tens of microseconds on some systems.
            for (int spin = 0; spin < 4000; ++spin) {
                if (generation_.load(std::memory_order_acquire) != seen) {
                    break;
                }
                std::this_thread::yield();
            }
            Job* job = nullptr;
            {
                std::unique_lock<std::mutex> lk(mutex_);
                start_cv_.wait(lk,
                               [&] { return generation_.load(std::memory_order_acquire) != seen; });
                seen = generation_.load(std::memory_order_acquire);
                if (stop_) {
                    return;
                }
                job = job_;
            }
            if (job != nullptr) {
                run_chunks(*job, tid);
            }
            {
                std::lock_guard<std::mutex> lk(mutex_);
                if (--pending_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    int n_ = 1;
    std::vector<std::thread> workers_;
    std::mutex submit_mutex_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<int> next_{0};
    Job* job_ = nullptr;
    int pending_ = 0;
    bool stop_ = false;
};

// --- Process-wide execution settings -------------------------------------------------------

struct ExecutionSettings {
    int threads = 0; // 0 = auto
    bool simd = true;
    long long parallel_min_cells = 128LL * 128LL;
    std::unique_ptr<ThreadPool> pool;
    std::mutex mutex;
};

inline int default_thread_count()
{
    if (const char* env = std::getenv("MHD_NUM_THREADS")) {
        const int n = std::atoi(env);
        if (n > 0) {
            return n;
        }
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<int>(hc);
}

inline ExecutionSettings& execution_settings()
{
    static ExecutionSettings s;
    return s;
}

inline ThreadPool& thread_pool()
{
    ExecutionSettings& s = execution_settings();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (!s.pool) {
        const int n = s.threads > 0 ? s.threads : default_thread_count();
        s.pool = std::make_unique<ThreadPool>(n);
    }
    return *s.pool;
}

// Number of worker threads (including the caller). n <= 0 restores the default.
inline void set_num_threads(int n)
{
    ExecutionSettings& s = execution_settings();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.threads = std::max(0, n);
    const int target = s.threads > 0 ? s.threads : default_thread_count();
    if (!s.pool || s.pool->size() != target) {
        s.pool.reset();
        s.pool = std::make_unique<ThreadPool>(target);
    }
}

inline int num_threads()
{
    return thread_pool().size();
}

inline void set_simd_enabled(bool enabled)
{
    execution_settings().simd = enabled;
}

inline bool simd_enabled()
{
    return execution_settings().simd;
}

inline void set_parallel_min_cells(long long cells)
{
    execution_settings().parallel_min_cells = std::max(0LL, cells);
}

inline long long parallel_min_cells()
{
    return execution_settings().parallel_min_cells;
}

// Row-parallel loop over [begin, end). Falls back to a serial call for small problems.
// `cells` is the amount of work used for the threshold (usually nx * ny).
template <class F>
void parallel_rows(int begin, int end, long long cells, F&& f)
{
    if (cells < parallel_min_cells()) {
        f(begin, end, 0);
        return;
    }
    ThreadPool& pool = thread_pool();
    const int rows = end - begin;
    // ~8 chunks per thread for load balance on heterogeneous cores, at least 2 rows each.
    const int grain = std::max(2, rows / (8 * pool.size()));
    pool.parallel_for(begin, end, grain, std::forward<F>(f));
}

// Flat-index parallel loop; chunk boundaries are multiples of `align` so that the split between
// vector body and scalar tail does not depend on the thread count.
template <class F>
void parallel_range(std::size_t n, int align, F&& f)
{
    if (static_cast<long long>(n) < parallel_min_cells()) {
        f(std::size_t{0}, n, 0);
        return;
    }
    ThreadPool& pool = thread_pool();
    const std::size_t block = static_cast<std::size_t>(std::max(1, align)) * 512;
    const int nblocks = static_cast<int>((n + block - 1) / block);
    const int grain = std::max(1, nblocks / (8 * pool.size()));
    pool.parallel_for(0, nblocks, grain, [&](int b0, int b1, int tid) {
        const std::size_t lo = static_cast<std::size_t>(b0) * block;
        const std::size_t hi = std::min(n, static_cast<std::size_t>(b1) * block);
        f(lo, hi, tid);
    });
}

} // namespace mhd

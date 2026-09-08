#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace bem3d {

// Lightweight, env-gated profiler for the BEM solve phases.
// Enable at runtime by setting the environment variable BEM_PROFILE (to any
// nonempty value other than "0") BEFORE the first profiled region runs.
// When disabled, ScopedTimer/profile_metric are no-ops (single cached-bool check).
class Profiler {
public:
    // Meyers singleton (function-local static: thread-safe since C++11, SIOF-safe).
    static Profiler &instance();

    // Whether profiling is enabled. Reads BEM_PROFILE once, then caches.
    static bool enabled();

    void add_time(const std::string &label, double seconds);
    void add_metric(const std::string &label, double value);
    void reset();
    std::string summary() const;

private:
    Profiler() = default;

    mutable std::mutex _mtx;
    // label -> (total_seconds, call_count)
    std::unordered_map<std::string, std::pair<double, long>> _times;
    // label -> (sum, count), for numeric metrics such as solver iterations
    std::unordered_map<std::string, std::pair<double, long>> _metrics;
};

// RAII scoped timer. Times the enclosing block and accumulates on destruction.
// No-op (aside from a cached bool load) when profiling is disabled.
class ScopedTimer {
public:
    explicit ScopedTimer(const char *label)
        : _label(label), _active(Profiler::enabled()) {
        if (_active)
            _t0 = std::chrono::steady_clock::now();
    }
    ~ScopedTimer() {
        if (!_active)
            return;
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - _t0).count();
        Profiler::instance().add_time(_label, s);
    }
    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

private:
    const char *_label;
    bool _active;
    std::chrono::steady_clock::time_point _t0;
};

// Record a numeric metric (e.g. BiCGSTAB iterations/error). Gated like the timers.
inline void profile_metric(const char *label, double value) {
    if (Profiler::enabled())
        Profiler::instance().add_metric(label, value);
}

} // namespace bem3d

// Unique-per-line scoped timer: BEM_SCOPED_TIMER("fwd.bicgstab_solve");
#define BEM_PROF_CONCAT_(a, b) a##b
#define BEM_PROF_CONCAT(a, b) BEM_PROF_CONCAT_(a, b)
#define BEM_SCOPED_TIMER(label) ::bem3d::ScopedTimer BEM_PROF_CONCAT(_bem_timer_, __LINE__)(label)

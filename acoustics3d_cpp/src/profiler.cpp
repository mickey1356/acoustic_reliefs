#include "profiler.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <vector>

namespace bem3d {

Profiler &Profiler::instance() {
    static Profiler p;
    return p;
}

bool Profiler::enabled() {
    static const bool en = [] {
        const char *v = std::getenv("BEM_PROFILE");
        return v != nullptr && v[0] != '\0' && std::string(v) != "0";
    }();
    return en;
}

void Profiler::add_time(const std::string &label, double seconds) {
    std::lock_guard<std::mutex> lk(_mtx);
    auto &e = _times[label];
    e.first += seconds;
    e.second += 1;
}

void Profiler::add_metric(const std::string &label, double value) {
    std::lock_guard<std::mutex> lk(_mtx);
    auto &e = _metrics[label];
    e.first += value;
    e.second += 1;
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lk(_mtx);
    _times.clear();
    _metrics.clear();
}

std::string Profiler::summary() const {
    std::lock_guard<std::mutex> lk(_mtx);
    std::ostringstream os;
    os << std::fixed << std::setprecision(6);

    // sort timing rows by total time descending
    std::vector<std::pair<std::string, std::pair<double, long>>> rows(_times.begin(), _times.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto &a, const auto &b) { return a.second.first > b.second.first; });

    double grand_total = 0.0;
    for (const auto &r : rows)
        grand_total += r.second.first;

    os << "=== Profiler timings ===\n";
    os << "  " << std::setw(28) << std::left << "label"
       << std::right << std::setw(12) << "total_s"
       << std::setw(8) << "calls"
       << std::setw(14) << "mean_ms"
       << std::setw(9) << "%tot" << "\n";
    for (const auto &r : rows) {
        double tot = r.second.first;
        long n = r.second.second;
        os << "  " << std::setw(28) << std::left << r.first
           << std::right << std::setw(12) << tot
           << std::setw(8) << n
           << std::setw(14) << (n ? tot / n * 1e3 : 0.0)
           << std::setw(8) << (grand_total > 0 ? tot / grand_total * 100.0 : 0.0) << "%\n";
    }
    os << "  " << std::setw(28) << std::left << "(sum of above)"
       << std::right << std::setw(12) << grand_total << "\n";

    os << "=== Metrics (mean over count) ===\n";
    for (const auto &r : _metrics) {
        double sum = r.second.first;
        long n = r.second.second;
        os << "  " << std::setw(28) << std::left << r.first
           << std::right << std::setw(14) << (n ? sum / n : 0.0)
           << "  (n=" << n << ")\n";
    }
    return os.str();
}

} // namespace bem3d

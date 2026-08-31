// timer.cpp
// TSC calibration, wall-clock helpers, and C0 busy-spin implementation.

#include "timer.hpp"
#include <immintrin.h>   // _mm_pause
#include <ctime>
#include <cstdio>
#include <cstdlib>

namespace timer {

uint64_t g_tsc_hz = 0;

uint64_t calibrate_tsc_hz(int duration_ms) {
    // Sleep briefly for OS to settle, then measure ticks vs wall clock.
    // We use CLOCK_MONOTONIC_RAW to avoid NTP distortions.

    uint64_t wall_start = now_ns();
    uint64_t tsc_start  = rdtsc();

    // Busy-spin for duration_ms to get a clean measurement window.
    uint64_t deadline_ns = wall_start + (uint64_t)duration_ms * 1'000'000ULL;
    while (now_ns() < deadline_ns) {
        _mm_pause();
    }

    uint64_t tsc_end  = rdtsc();
    uint64_t wall_end = now_ns();

    uint64_t delta_tsc  = tsc_end  - tsc_start;
    uint64_t delta_ns   = wall_end - wall_start;

    // tsc_hz = delta_tsc / (delta_ns / 1e9)
    g_tsc_hz = (uint64_t)((double)delta_tsc / ((double)delta_ns * 1e-9));

    fprintf(stderr, "[timer] TSC calibration: %lu ticks in %lu ns → %.3f GHz\n",
            delta_tsc, delta_ns, g_tsc_hz / 1e9);

    return g_tsc_hz;
}

void spin_wait_us(double duration_us) {
    if (duration_us <= 0.0) return;
    uint64_t deadline = now_ns() + (uint64_t)(duration_us * 1000.0);
    spin_until_ns(deadline);
}

void spin_until_ns(uint64_t deadline_ns) {
    while (now_ns() < deadline_ns) {
        _mm_pause();
    }
}

} // namespace timer

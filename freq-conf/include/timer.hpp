// timer.hpp
// High-resolution timing and TSC calibration for the P-state experiment.
//
// Provides:
//   - Empirical TSC frequency calibration (measures RDTSC ticks vs wall clock)
//   - Nanosecond-resolution elapsed time helpers
//   - Tight busy-wait (spin) using PAUSE to avoid deep C-state during idle phase

#pragma once

#include <cstdint>
#include <ctime>

namespace timer {

// ---------------------------------------------------------------------------
// TSC calibration
// ---------------------------------------------------------------------------

// Calibrate the TSC frequency empirically by measuring rdtsc ticks
// against CLOCK_MONOTONIC_RAW over `duration_ms` milliseconds.
// Returns TSC frequency in Hz (ticks per second).
// Called once at program start.
uint64_t calibrate_tsc_hz(int duration_ms = 100);

// Global TSC frequency (set by calibrate_tsc_hz, used by msr_reader).
extern uint64_t g_tsc_hz;

// ---------------------------------------------------------------------------
// RDTSC helpers
// ---------------------------------------------------------------------------

// Read the Time Stamp Counter (serializing with LFENCE to prevent reorder).
inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ volatile(
        "lfence\n\t"
        "rdtsc\n\t"
        : "=a"(lo), "=d"(hi)
    );
    return ((uint64_t)hi << 32) | lo;
}

// ---------------------------------------------------------------------------
// Wall clock helpers (CLOCK_MONOTONIC_RAW for immunity to NTP adjustments)
// ---------------------------------------------------------------------------

inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1'000'000'000ULL + ts.tv_nsec;
}

inline double elapsed_s(uint64_t start_ns, uint64_t end_ns) {
    return (double)(end_ns - start_ns) * 1e-9;
}

// ---------------------------------------------------------------------------
// Busy spin-wait (keeps core in C0, avoids C-state wake-up latency)
// Uses _mm_pause() to reduce power while spinning without sleeping.
// ---------------------------------------------------------------------------

// Spin for `duration_us` microseconds using a busy-wait loop.
void spin_wait_us(double duration_us);

// Spin until the monotonic wall clock reaches `deadline_ns`.
void spin_until_ns(uint64_t deadline_ns);

} // namespace timer

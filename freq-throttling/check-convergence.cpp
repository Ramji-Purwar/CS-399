// check_convergence.cpp
//
// Standalone verifier. Does NOT modify the original benchmark file.
//
// This reuses the *exact same* intrinsics-level logic as foo_mulps() and
// foo_avx512() in the original file (same operand encoding, same mul_a/mul_b
// setup, same initial value 1.234567f), but instead of looping ITERATIONS
// times silently, it prints out each step's bit pattern so we can see
// directly -- from real hardware SSE/AVX-512 instructions, not a Python
// emulation -- whether the sequence converges to a constant, diverges, or
// settles into a repeating cycle.
//
// Build (either works now -- the AVX-512 function is self-tagged with
// __attribute__((target("avx512f"))), so it no longer needs a global
// -march=native/-mavx512f flag to compile):
//   g++ -O3 -o check-convergence check-convergence.cpp
//   g++ -O2 -march=native -o check_convergence check_convergence.cpp
// Run:
//   ./check-convergence
//
// If AVX-512 isn't available on the host CPU, that section is skipped
// automatically (SSE mulps section always runs).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>

#if defined(__GNUC__)
#include <cpuid.h>
#endif

static bool cpu_has_avx512f() {
#if defined(__GNUC__)
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & (1u << 16)) != 0; // AVX512F bit
#else
    return false;
#endif
}

// ---------------------------------------------------------------------
// Same operand -> (fa, fb) mapping as the original foo_mulps/foo_avx512.
// ---------------------------------------------------------------------
static void operand_to_fafb(uint64_t data, float &fa, float &fb) {
    if (data == 0) {
        fa = 1.0f;
        fb = 1.0f;
    } else {
        fa = 1.3333334f;
        fb = 0.75f;
    }
}

static uint32_t f32_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

// ---------------------------------------------------------------------
// SSE (mulps) instrumented version of foo_mulps() -- same setup, same
// __m128 chain (just one chain here since all 8 chains in the real kernel
// are identical/independent with the same starting value), but we read
// back the value after every pair of instructions instead of running
// ITERATIONS/16 times blind.
// ---------------------------------------------------------------------
static void check_mulps(uint64_t data, int print_steps) {
    float fa, fb;
    operand_to_fafb(data, fa, fb);

    __m128 mul_a = _mm_set1_ps(fa);
    __m128 mul_b = _mm_set1_ps(fb);
    __m128 x0 = _mm_set1_ps(1.234567f);

    printf("== SSE mulps  | operand=0x%016lx | fa=%.9g (0x%08x) fb=%.9g (0x%08x) ==\n",
           (unsigned long)data, fa, f32_bits(fa), fb, f32_bits(fb));

    uint32_t prev_pair[2] = {0, 0};
    int cycle_len = -1;

    for (int i = 0; i < print_steps; ++i) {
        x0 = _mm_mul_ps(x0, mul_a);              // real SSE MULPS instruction
        float after_a = _mm_cvtss_f32(x0);
        x0 = _mm_mul_ps(x0, mul_b);              // real SSE MULPS instruction
        float after_b = _mm_cvtss_f32(x0);

        if (i < 10 || i == print_steps - 1) {
            printf("  step %2d: after*fa = %.9g (0x%08x)   after*fb = %.9g (0x%08x)\n",
                   i, after_a, f32_bits(after_a), after_b, f32_bits(after_b));
        }

        uint32_t pair[2] = {f32_bits(after_a), f32_bits(after_b)};
        if (i > 0 && pair[0] == prev_pair[0] && pair[1] == prev_pair[1] && cycle_len < 0) {
            cycle_len = 1; // repeats every iteration once it appears
            printf("  -> repeating pattern detected at step %d (period 1)\n", i);
        }
        prev_pair[0] = pair[0];
        prev_pair[1] = pair[1];
    }

    // Run a long tail silently to confirm it never drifts (Inf/NaN/0/other).
    long total_iters = 2000000;
    for (long i = 0; i < total_iters; ++i) {
        x0 = _mm_mul_ps(x0, mul_a);
        x0 = _mm_mul_ps(x0, mul_b);
    }
    float final_val = _mm_cvtss_f32(x0);
    printf("  after %ld more iterations: %.9g (0x%08x)\n\n", total_iters, final_val, f32_bits(final_val));
}

// ---------------------------------------------------------------------
// AVX-512 (vmulps) instrumented version of foo_avx512() -- same setup.
//
// Tagged with target("avx512f") so this function alone is compiled with
// AVX-512 codegen enabled, even if the whole translation unit is built
// with a plain "g++ -O3" (no -march=native/-mavx512f). This mirrors how
// the original benchmark relies on a build flag to enable AVX-512; here
// we just make the checker self-sufficient so it builds the same way
// everywhere.
// ---------------------------------------------------------------------
__attribute__((target("avx512f")))
static void check_avx512(uint64_t data, int print_steps) {
    float fa, fb;
    operand_to_fafb(data, fa, fb);

    __m512 mul_a = _mm512_set1_ps(fa);
    __m512 mul_b = _mm512_set1_ps(fb);
    __m512 x0 = _mm512_set1_ps(1.234567f);

    printf("== AVX-512 vmulps | operand=0x%016lx | fa=%.9g (0x%08x) fb=%.9g (0x%08x) ==\n",
           (unsigned long)data, fa, f32_bits(fa), fb, f32_bits(fb));

    for (int i = 0; i < print_steps; ++i) {
        x0 = _mm512_mul_ps(x0, mul_a);           // real AVX-512 VMULPS instruction
        float after_a = _mm_cvtss_f32(_mm512_extractf32x4_ps(x0, 0));
        x0 = _mm512_mul_ps(x0, mul_b);           // real AVX-512 VMULPS instruction
        float after_b = _mm_cvtss_f32(_mm512_extractf32x4_ps(x0, 0));

        if (i < 10 || i == print_steps - 1) {
            printf("  step %2d: after*fa = %.9g (0x%08x)   after*fb = %.9g (0x%08x)\n",
                   i, after_a, f32_bits(after_a), after_b, f32_bits(after_b));
        }
    }

    long total_iters = 2000000;
    for (long i = 0; i < total_iters; ++i) {
        x0 = _mm512_mul_ps(x0, mul_a);
        x0 = _mm512_mul_ps(x0, mul_b);
    }
    float final_val = _mm_cvtss_f32(_mm512_extractf32x4_ps(x0, 0));
    printf("  after %ld more iterations: %.9g (0x%08x)\n\n", total_iters, final_val, f32_bits(final_val));
}

int main() {
    printf("Verifying convergence behavior using the SAME intrinsics/operand\n");
    printf("logic as foo_mulps()/foo_avx512() in the original benchmark.\n\n");

    check_mulps(0x0000000000000000ULL, 15);
    check_mulps(0xAAAAAAAAAAAAAAABULL, 15);

    if (cpu_has_avx512f()) {
        check_avx512(0x0000000000000000ULL, 15);
        check_avx512(0xAAAAAAAAAAAAAAABULL, 15);
    } else {
        printf("AVX-512F not available on this CPU -- skipping vmulps section.\n");
        printf("(SSE mulps result above is bit-for-bit identical per-lane math,\n");
        printf(" since vmulps on this fa/fb/x0 does the same scalar float32\n");
        printf(" operation independently in every 32-bit lane.)\n");
    }

    return 0;
}
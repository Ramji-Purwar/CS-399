// workload_kernel.cpp
#include "workload_kernel.hpp"
#include <immintrin.h>
#include <cpuid.h>

namespace kernel {

void run_avx512_fma_iters(uint64_t iters) {
    if (iters == 0) return;

    __m512 acc0 = _mm512_set1_ps(1.0f);
    __m512 acc1 = _mm512_set1_ps(1.0f);
    __m512 acc2 = _mm512_set1_ps(1.0f);
    __m512 acc3 = _mm512_set1_ps(1.0f);
    __m512 acc4 = _mm512_set1_ps(1.0f);
    __m512 acc5 = _mm512_set1_ps(1.0f);
    __m512 acc6 = _mm512_set1_ps(1.0f);
    __m512 acc7 = _mm512_set1_ps(1.0f);

    __m512 mul = _mm512_set1_ps(1.0000001f);
    __m512 add = _mm512_set1_ps(0.0f);

    for (uint64_t i = 0; i < iters; ++i) {
        acc0 = _mm512_fmadd_ps(acc0, mul, add);
        acc1 = _mm512_fmadd_ps(acc1, mul, add);
        acc2 = _mm512_fmadd_ps(acc2, mul, add);
        acc3 = _mm512_fmadd_ps(acc3, mul, add);
        acc4 = _mm512_fmadd_ps(acc4, mul, add);
        acc5 = _mm512_fmadd_ps(acc5, mul, add);
        acc6 = _mm512_fmadd_ps(acc6, mul, add);
        acc7 = _mm512_fmadd_ps(acc7, mul, add);
    }

    // Defeat optimizer
    volatile float sink = _mm512_cvtss_f32(acc0) + 
                          _mm512_cvtss_f32(acc1) + 
                          _mm512_cvtss_f32(acc2) + 
                          _mm512_cvtss_f32(acc3) + 
                          _mm512_cvtss_f32(acc4) + 
                          _mm512_cvtss_f32(acc5) + 
                          _mm512_cvtss_f32(acc6) + 
                          _mm512_cvtss_f32(acc7);
    (void)sink;
}

void run_avx512_instructions(uint64_t insn_count) {
    uint64_t iters = insn_count / 8;
    if (iters > 0) {
        run_avx512_fma_iters(iters);
    }
}

bool avx512_available() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & bit_AVX512F) != 0;
    }
    return false;
}

} // namespace kernel

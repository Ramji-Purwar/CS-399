// workload_kernel.hpp
// AVX-512 FMA compute kernel for P-state throttling characterization.
//
// Design rationale (see experiment spec §4.1):
//   - Uses vfmadd (512-bit FP multiply-add) — the heaviest commonly available
//     AVX-512 instruction, targeting the 2 AVX-512 FMA execution ports.
//   - 8 independent accumulator chains (zmm0–zmm7) prevent the pipeline from
//     stalling on FMA latency (≈4 cycles); 2 ports × ~4 cycle latency means
//     ≥8 in-flight chains are needed to fully saturate throughput.
//   - All operands stay in registers (zmm0–zmm15). Zero memory/cache traffic
//     ensures measured power draw is from compute units, not memory bandwidth.
//   - Uses float (512-bit = 16 × f32 per register) for maximum throughput.
//
// Dry-run mode: compiles the kernel structure but spins doing integer NOPs
// so it can be exercised without access to an AVX-512 machine (though this
// machine does have AVX-512 confirmed from lscpu).

#pragma once

#include <cstdint>
#include <cstdbool>

namespace kernel {

// Run exactly `insn_count` AVX-512 FMA instructions (rounded to nearest 8)
void run_avx512_instructions(uint64_t insn_count);

// Underlying loop runner (each iteration executes 8 AVX-512 FMA instructions)
void run_avx512_fma_iters(uint64_t iters);

// Returns true if AVX-512F is available at runtime (CPUID check).
bool avx512_available();

} // namespace kernel

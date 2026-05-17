/**
 * gollek_metal_bridge.m — Objective-C/Metal bridge for Gollek
 * 
 * Optimized for reduced memory pressure and fixed KV cache layout.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>

// ── Globals ───────────────────────────────────────────────────────────────────

id<MTLDevice>       g_device      = nil;
id<MTLCommandQueue> g_queue       = nil;
BOOL                g_initialized = NO;
static id<MTLComputePipelineState> g_matvec_half_pipeline = nil;
static id<MTLComputePipelineState> g_matvec_t_half_pipeline = nil;
static id<MTLComputePipelineState> g_matvec_half_pair_pipeline = nil;
static id<MTLComputePipelineState> g_matvec_half_triple_mixed_pipeline = nil;
static id<MTLComputePipelineState> g_add_pipeline = nil;
static id<MTLComputePipelineState> g_silu_ffn_pipeline = nil;
static id<MTLComputePipelineState> g_gelu_ffn_pipeline = nil;
static id<MTLComputePipelineState> g_rmsnorm_pipeline = nil;
static id<MTLComputePipelineState> g_decode_attention_pipeline = nil;
static id<MTLBuffer> g_swiglu_gate_scratch = nil;
static id<MTLBuffer> g_swiglu_up_scratch = nil;
static id<MTLBuffer> g_swiglu_combined_scratch = nil;
static size_t g_swiglu_scratch_capacity = 0;

static const int DEFAULT_SMALL_MATMUL_CPU_THRESHOLD = 1024;
static int g_small_matmul_cpu_threshold = DEFAULT_SMALL_MATMUL_CPU_THRESHOLD;
static BOOL g_disable_mps_cache = NO;
static BOOL g_disable_elementwise_kernels = YES;

// MPSMatrixMultiplication and MPSMatrixDescriptor instances are immutable for a
// fixed shape/layout. Reusing them avoids rebuilding MPS state dozens of times
// per generated token while keeping Java-owned tensor memory zero-copy.
static NSMutableDictionary<NSString*, MPSMatrixMultiplication*>* g_mmul_cache = nil;
static NSMutableDictionary<NSString*, MPSMatrixDescriptor*>* g_matrix_desc_cache = nil;
static NSMutableDictionary<NSString*, MPSMatrixVectorMultiplication*>* g_mvec_cache = nil;
static NSMutableDictionary<NSString*, MPSVectorDescriptor*>* g_vector_desc_cache = nil;
static NSMutableSet<NSString*>* g_mps_matvec_validated_shapes = nil;
static NSMutableSet<NSString*>* g_mps_matvec_failed_shapes = nil;
static NSMutableSet<NSString*>* g_mps_matvec_mps_preferred_shapes = nil;
static NSMutableSet<NSString*>* g_mps_matvec_custom_preferred_shapes = nil;
static BOOL g_enable_mps_matvec = NO;
static BOOL g_enable_mps_matvec_autotune = NO;
static int g_mps_matvec_max_inner_override = -1;
static int g_mps_matvec_max_output_override = -1;
static int g_mps_matvec_autotune_max_output_override = -1;
static BOOL g_disable_mps_matvec_after_validation_failure = NO;

// Thread-local scratch buffers to avoid allocation in hot loops
static __thread float* tl_k_scratch = NULL;
static __thread float* tl_v_scratch = NULL;
static __thread float* tl_score_scratch = NULL;
static __thread size_t tl_scratch_capacity = 0;
static __thread uint16_t* tl_half_input_scratch = NULL;
static __thread uint16_t* tl_half_output_scratch = NULL;
static __thread size_t tl_half_input_capacity = 0;
static __thread size_t tl_half_output_capacity = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static id<MTLBuffer> wrap_ptr(void* ptr, size_t bytes) {
    return [g_device newBufferWithBytesNoCopy:ptr
                                       length:bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

static BOOL ensure_swiglu_scratch(size_t activation_bytes,
                                  id<MTLBuffer>* gate,
                                  id<MTLBuffer>* up,
                                  id<MTLBuffer>* combined) {
    if (activation_bytes == 0) return NO;
    if (g_swiglu_scratch_capacity < activation_bytes
            || g_swiglu_gate_scratch == nil
            || g_swiglu_up_scratch == nil
            || g_swiglu_combined_scratch == nil) {
        g_swiglu_gate_scratch = [g_device newBufferWithLength:activation_bytes options:MTLResourceStorageModePrivate];
        g_swiglu_up_scratch = [g_device newBufferWithLength:activation_bytes options:MTLResourceStorageModePrivate];
        g_swiglu_combined_scratch = [g_device newBufferWithLength:activation_bytes options:MTLResourceStorageModePrivate];
        if (g_swiglu_gate_scratch == nil || g_swiglu_up_scratch == nil || g_swiglu_combined_scratch == nil) {
            g_swiglu_gate_scratch = nil;
            g_swiglu_up_scratch = nil;
            g_swiglu_combined_scratch = nil;
            g_swiglu_scratch_capacity = 0;
            return NO;
        }
        g_swiglu_scratch_capacity = activation_bytes;
    }
    *gate = g_swiglu_gate_scratch;
    *up = g_swiglu_up_scratch;
    *combined = g_swiglu_combined_scratch;
    return YES;
}

static BOOL env_truthy(const char* name) {
    const char* value = getenv(name);
    return value != NULL
            && (strcmp(value, "1") == 0
                || strcasecmp(value, "true") == 0
                || strcasecmp(value, "yes") == 0);
}

static int env_int_or_default(const char* name, int default_value) {
    const char* value = getenv(name);
    if (value == NULL || value[0] == '\0') return default_value;
    char* end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value) return default_value;
    if (parsed < 0) return default_value;
    if (parsed > INT32_MAX) return INT32_MAX;
    return (int)parsed;
}

static float env_float_or_default(const char* name, float default_value) {
    const char* value = getenv(name);
    if (value == NULL || value[0] == '\0') return default_value;
    char* end = NULL;
    float parsed = strtof(value, &end);
    if (end == value || !isfinite(parsed)) return default_value;
    return parsed;
}

static uint64_t monotonic_nanos(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static NSString* mps_matvec_shape_key(int K, int N) {
    return [NSString stringWithFormat:@"%d:%d", K, N];
}

static float f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    int exp = (int)((bits >> 10) & 0x1fu);
    uint32_t mant = bits & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            uint32_t exp32 = (uint32_t)(exp + (127 - 15));
            out = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        uint32_t exp32 = (uint32_t)(exp + (127 - 15));
        out = sign | (exp32 << 23) | (mant << 13);
    }
    float value;
    memcpy(&value, &out, sizeof(value));
    return value;
}

static uint16_t f32_to_f16_bits(float value) {
    __fp16 half = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static float half_matvec_reference_row(const float* A, const uint16_t* B, int K, int row) {
    const uint16_t* weight = B + ((size_t)row * (size_t)K);
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[k] * f16_to_f32(weight[k]);
    }
    return sum;
}

static BOOL validate_mps_matvec_half_output(const float* C, const float* A, const uint16_t* B, int K, int N) {
    if (env_truthy("GOLLEK_METAL_MPS_MATVEC_SKIP_VALIDATE")) {
        return YES;
    }
    int samples = env_int_or_default("GOLLEK_METAL_MPS_MATVEC_VALIDATE_SAMPLES", 8);
    if (samples <= 0) {
        samples = 1;
    }
    if (samples > N) {
        samples = N;
    }
    float abs_tol = 0.08f;
    float rel_tol = 0.03f;
    const char* abs_env = getenv("GOLLEK_METAL_MPS_MATVEC_VALIDATE_ABS_TOL");
    const char* rel_env = getenv("GOLLEK_METAL_MPS_MATVEC_VALIDATE_REL_TOL");
    if (abs_env != NULL && abs_env[0] != '\0') abs_tol = strtof(abs_env, NULL);
    if (rel_env != NULL && rel_env[0] != '\0') rel_tol = strtof(rel_env, NULL);

    // Deterministic spread over the vector. The custom Metal reduction kernel
    // remains the correctness fallback, so this check is allowed to be strict.
    uint32_t state = (uint32_t)(N * 2654435761u) ^ (uint32_t)(K * 2246822519u);
    for (int i = 0; i < samples; i++) {
        int row;
        if (i == 0) {
            row = 0;
        } else if (i == 1) {
            row = N - 1;
        } else if (i == 2) {
            row = N / 2;
        } else {
            state = state * 1664525u + 1013904223u;
            row = (int)(state % (uint32_t)N);
        }
        float actual = C[row];
        float expected = half_matvec_reference_row(A, B, K, row);
        if (!isfinite(actual) || !isfinite(expected)) {
            return NO;
        }
        float diff = fabsf(actual - expected);
        float limit = fmaxf(abs_tol, rel_tol * fmaxf(1.0f, fabsf(expected)));
        if (diff > limit) {
            if (env_truthy("GOLLEK_METAL_MPS_MATVEC_DEBUG")) {
                fprintf(stderr,
                        "[gollek-metal] MPS matvec validation failed row=%d K=%d N=%d actual=%g expected=%g diff=%g limit=%g\n",
                        row, K, N, actual, expected, diff, limit);
            }
            return NO;
        }
    }
    return YES;
}

static id<MTLComputePipelineState> compile_pipeline(id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (fn == nil) {
        return nil;
    }
    return [g_device newComputePipelineStateWithFunction:fn error:&error];
}

static void compile_runtime_kernels(void) {
    NSString* source =
        @"#include <metal_stdlib>\n"
        @"using namespace metal;\n"
        @"constant uint GOLLEK_RMS_THREADS = 256;\n"
        @"kernel void gollek_add_kernel(\n"
        @"    device float* C [[buffer(0)]],\n"
        @"    device const float* A [[buffer(1)]],\n"
        @"    device const float* B [[buffer(2)]],\n"
        @"    constant uint& N [[buffer(3)]],\n"
        @"    uint gid [[thread_position_in_grid]]) {\n"
        @"    if (gid < N) C[gid] = A[gid] + B[gid];\n"
        @"}\n"
        @"kernel void gollek_silu_ffn_kernel(\n"
        @"    device float* out [[buffer(0)]],\n"
        @"    device const float* gate [[buffer(1)]],\n"
        @"    device const float* up [[buffer(2)]],\n"
        @"    constant uint& N [[buffer(3)]],\n"
        @"    uint gid [[thread_position_in_grid]]) {\n"
        @"    if (gid >= N) return;\n"
        @"    float g = gate[gid];\n"
        @"    out[gid] = (g / (1.0f + exp(-g))) * up[gid];\n"
        @"}\n"
        @"kernel void gollek_gelu_ffn_kernel(\n"
        @"    device float* out [[buffer(0)]],\n"
        @"    device const float* gate [[buffer(1)]],\n"
        @"    device const float* up [[buffer(2)]],\n"
        @"    constant uint& N [[buffer(3)]],\n"
        @"    uint gid [[thread_position_in_grid]]) {\n"
        @"    if (gid >= N) return;\n"
        @"    float g = gate[gid];\n"
        @"    float inner = 0.79788456f * (g + 0.044715f * g * g * g);\n"
        @"    float gelu = 0.5f * g * (1.0f + tanh(inner));\n"
        @"    out[gid] = gelu * up[gid];\n"
        @"}\n"
        @"kernel void gollek_rmsnorm_kernel(\n"
        @"    device float* out [[buffer(0)]],\n"
        @"    device const float* x [[buffer(1)]],\n"
        @"    device const float* weight [[buffer(2)]],\n"
        @"    constant uint& N [[buffer(3)]],\n"
        @"    constant float& eps [[buffer(4)]],\n"
        @"    constant uint& addOne [[buffer(5)]],\n"
        @"    uint tid [[thread_position_in_threadgroup]]) {\n"
        @"    threadgroup float partial[GOLLEK_RMS_THREADS];\n"
        @"    float ss = 0.0f;\n"
        @"    for (uint i = tid; i < N; i += GOLLEK_RMS_THREADS) {\n"
        @"        float v = x[i];\n"
        @"        ss += v * v;\n"
        @"    }\n"
        @"    partial[tid] = ss;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    for (uint stride = GOLLEK_RMS_THREADS >> 1; stride > 0; stride >>= 1) {\n"
        @"        if (tid < stride) partial[tid] += partial[tid + stride];\n"
        @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    }\n"
        @"    float inv = rsqrt((partial[0] / float(N)) + eps);\n"
        @"    for (uint i = tid; i < N; i += GOLLEK_RMS_THREADS) {\n"
        @"        float w = weight[i] + (addOne != 0 ? 1.0f : 0.0f);\n"
        @"        out[i] = x[i] * inv * w;\n"
        @"    }\n"
        @"}\n"
        @"kernel void gollek_matvec_tb_half_kernel(\n"
        @"    device float* C [[buffer(0)]],\n"
        @"    device const float* A [[buffer(1)]],\n"
        @"    device const half* B [[buffer(2)]],\n"
        @"    constant uint& K [[buffer(3)]],\n"
        @"    constant uint& N [[buffer(4)]],\n"
        @"    uint gid [[threadgroup_position_in_grid]],\n"
        @"    uint tid [[thread_position_in_threadgroup]]) {\n"
        @"    if (gid >= N) return;\n"
        @"    uint base = gid * K;\n"
        @"    float sum = 0.0f;\n"
        @"    for (uint k = tid; k < K; k += GOLLEK_RMS_THREADS) {\n"
        @"        sum += A[k] * float(B[base + k]);\n"
        @"    }\n"
        @"    threadgroup float partial[GOLLEK_RMS_THREADS];\n"
        @"    partial[tid] = sum;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    for (uint stride = GOLLEK_RMS_THREADS >> 1; stride > 0; stride >>= 1) {\n"
        @"        if (tid < stride) partial[tid] += partial[tid + stride];\n"
        @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    }\n"
        @"    if (tid == 0) C[gid] = partial[0];\n"
        @"}\n"
        @"kernel void gollek_matvec_tb_half_pair_kernel(\n"
        @"    device float* C0 [[buffer(0)]],\n"
        @"    device float* C1 [[buffer(1)]],\n"
        @"    device const float* A [[buffer(2)]],\n"
        @"    device const half* B0 [[buffer(3)]],\n"
        @"    device const half* B1 [[buffer(4)]],\n"
        @"    constant uint& K [[buffer(5)]],\n"
        @"    constant uint& N [[buffer(6)]],\n"
        @"    uint gid [[threadgroup_position_in_grid]],\n"
        @"    uint tid [[thread_position_in_threadgroup]]) {\n"
        @"    if (gid >= N) return;\n"
        @"    uint base = gid * K;\n"
        @"    float sum0 = 0.0f;\n"
        @"    float sum1 = 0.0f;\n"
        @"    for (uint k = tid; k < K; k += GOLLEK_RMS_THREADS) {\n"
        @"        float av = A[k];\n"
        @"        sum0 += av * float(B0[base + k]);\n"
        @"        sum1 += av * float(B1[base + k]);\n"
        @"    }\n"
        @"    threadgroup float partial0[GOLLEK_RMS_THREADS];\n"
        @"    threadgroup float partial1[GOLLEK_RMS_THREADS];\n"
        @"    partial0[tid] = sum0;\n"
        @"    partial1[tid] = sum1;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    for (uint stride = GOLLEK_RMS_THREADS >> 1; stride > 0; stride >>= 1) {\n"
        @"        if (tid < stride) {\n"
        @"            partial0[tid] += partial0[tid + stride];\n"
        @"            partial1[tid] += partial1[tid + stride];\n"
        @"        }\n"
        @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    }\n"
        @"    if (tid == 0) {\n"
        @"        C0[gid] = partial0[0];\n"
        @"        C1[gid] = partial1[0];\n"
        @"    }\n"
        @"}\n"
        @"kernel void gollek_matvec_tb_half_triple_mixed_kernel(\n"
        @"    device float* C0 [[buffer(0)]],\n"
        @"    device float* C1 [[buffer(1)]],\n"
        @"    device float* C2 [[buffer(2)]],\n"
        @"    device const float* A [[buffer(3)]],\n"
        @"    device const half* B0 [[buffer(4)]],\n"
        @"    device const half* B1 [[buffer(5)]],\n"
        @"    device const half* B2 [[buffer(6)]],\n"
        @"    constant uint& K [[buffer(7)]],\n"
        @"    constant uint& N0 [[buffer(8)]],\n"
        @"    constant uint& N1 [[buffer(9)]],\n"
        @"    constant uint& N2 [[buffer(10)]],\n"
        @"    uint gid [[threadgroup_position_in_grid]],\n"
        @"    uint tid [[thread_position_in_threadgroup]]) {\n"
        @"    uint total = N0 + N1 + N2;\n"
        @"    if (gid >= total) return;\n"
        @"    device const half* B = B0;\n"
        @"    device float* C = C0;\n"
        @"    uint local = gid;\n"
        @"    if (gid >= N0 + N1) {\n"
        @"        B = B2;\n"
        @"        C = C2;\n"
        @"        local = gid - N0 - N1;\n"
        @"    } else if (gid >= N0) {\n"
        @"        B = B1;\n"
        @"        C = C1;\n"
        @"        local = gid - N0;\n"
        @"    }\n"
        @"    uint base = local * K;\n"
        @"    float sum = 0.0f;\n"
        @"    for (uint k = tid; k < K; k += GOLLEK_RMS_THREADS) {\n"
        @"        sum += A[k] * float(B[base + k]);\n"
        @"    }\n"
        @"    threadgroup float partial[GOLLEK_RMS_THREADS];\n"
        @"    partial[tid] = sum;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    for (uint stride = GOLLEK_RMS_THREADS >> 1; stride > 0; stride >>= 1) {\n"
        @"        if (tid < stride) partial[tid] += partial[tid + stride];\n"
        @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    }\n"
        @"    if (tid == 0) C[local] = partial[0];\n"
        @"}\n"
        @"kernel void gollek_decode_attention_kernel(\n"
        @"    device float* out [[buffer(0)]],\n"
        @"    device const float* Q [[buffer(1)]],\n"
        @"    device const float* K_cache [[buffer(2)]],\n"
        @"    device const float* V_cache [[buffer(3)]],\n"
        @"    device const int* block_table [[buffer(4)]],\n"
        @"    device const int* context_lens [[buffer(5)]],\n"
        @"    constant uint& H [[buffer(6)]],\n"
        @"    constant uint& H_kv [[buffer(7)]],\n"
        @"    constant uint& D [[buffer(8)]],\n"
        @"    constant uint& block_size [[buffer(9)]],\n"
        @"    constant uint& max_blocks [[buffer(10)]],\n"
        @"    constant float& scale [[buffer(11)]],\n"
        @"    constant uint& is_causal [[buffer(12)]],\n"
        @"    constant int& query_start_pos [[buffer(13)]],\n"
        @"    constant int& sliding_window [[buffer(14)]],\n"
        @"    constant float& soft_cap [[buffer(15)]],\n"
        @"    uint group [[threadgroup_position_in_grid]],\n"
        @"    uint tid [[thread_position_in_threadgroup]]) {\n"
        @"    threadgroup float scores[4096];\n"
        @"    threadgroup float partial[256];\n"
        @"    uint b = group / H;\n"
        @"    uint h = group - b * H;\n"
        @"    int ctx_len = context_lens[b];\n"
        @"    if (ctx_len <= 0 || H_kv == 0 || D == 0) return;\n"
        @"    uint gqa_group = max(1u, H / H_kv);\n"
        @"    uint kv_h = min(H_kv - 1u, h / gqa_group);\n"
        @"    int abs_query_pos = query_start_pos;\n"
        @"    if (is_causal != 0 && sliding_window <= 0 && query_start_pos == 0) abs_query_pos = ctx_len - 1;\n"
        @"    int max_pos = (is_causal != 0) ? abs_query_pos : (ctx_len - 1);\n"
        @"    if (max_pos >= ctx_len) max_pos = ctx_len - 1;\n"
        @"    int min_pos = sliding_window > 0 ? (abs_query_pos - sliding_window + 1) : 0;\n"
        @"    if (min_pos < 0) min_pos = 0;\n"
        @"    uint valid = max_pos >= min_pos ? uint(max_pos - min_pos + 1) : 0u;\n"
        @"    if (valid == 0u || valid > 4096u) {\n"
        @"        for (uint d = tid; d < D; d += 256u) out[(b * H + h) * D + d] = 0.0f;\n"
        @"        return;\n"
        @"    }\n"
        @"    device const float* qh = Q + (b * H + h) * D;\n"
        @"    float local_max = -3.402823466e+38F;\n"
        @"    for (uint idx = tid; idx < valid; idx += 256u) {\n"
        @"        uint pos = uint(min_pos) + idx;\n"
        @"        uint logical_block = pos / block_size;\n"
        @"        uint block_offset = pos - logical_block * block_size;\n"
        @"        int phys_i = block_table[b * max_blocks + logical_block];\n"
        @"        uint phys = uint(max(phys_i, 0));\n"
        @"        device const float* kh = K_cache + (((phys * H_kv + kv_h) * block_size + block_offset) * D);\n"
        @"        float dot = 0.0f;\n"
        @"        for (uint d = 0u; d < D; d++) dot += qh[d] * kh[d];\n"
        @"        float score = dot * scale;\n"
        @"        if (soft_cap > 0.0f) score = soft_cap * tanh(score / soft_cap);\n"
        @"        scores[idx] = score;\n"
        @"        local_max = max(local_max, score);\n"
        @"    }\n"
        @"    partial[tid] = local_max;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
        @"        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);\n"
        @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    }\n"
        @"    float mx = partial[0];\n"
        @"    float local_sum = 0.0f;\n"
        @"    for (uint idx = tid; idx < valid; idx += 256u) {\n"
        @"        float w = exp(scores[idx] - mx);\n"
        @"        scores[idx] = w;\n"
        @"        local_sum += w;\n"
        @"    }\n"
        @"    partial[tid] = local_sum;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
        @"        if (tid < stride) partial[tid] += partial[tid + stride];\n"
        @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    }\n"
        @"    float inv_sum = 1.0f / (partial[0] + 1.0e-9f);\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    for (uint d = tid; d < D; d += 256u) {\n"
        @"        float acc = 0.0f;\n"
        @"        for (uint idx = 0u; idx < valid; idx++) {\n"
        @"            uint pos = uint(min_pos) + idx;\n"
        @"            uint logical_block = pos / block_size;\n"
        @"            uint block_offset = pos - logical_block * block_size;\n"
        @"            int phys_i = block_table[b * max_blocks + logical_block];\n"
        @"            uint phys = uint(max(phys_i, 0));\n"
        @"            device const float* vh = V_cache + (((phys * H_kv + kv_h) * block_size + block_offset) * D);\n"
        @"            acc += (scores[idx] * inv_sum) * vh[d];\n"
        @"        }\n"
        @"        out[(b * H + h) * D + d] = acc;\n"
        @"    }\n"
        @"}\n"
        @"kernel void gollek_matvec_t_half_kernel(\n"
        @"    device float* C [[buffer(0)]],\n"
        @"    device const float* A [[buffer(1)]],\n"
        @"    device const half* B [[buffer(2)]],\n"
        @"    constant uint& K [[buffer(3)]],\n"
        @"    constant uint& N [[buffer(4)]],\n"
        @"    uint gid [[threadgroup_position_in_grid]],\n"
        @"    uint tid [[thread_position_in_threadgroup]]) {\n"
        @"    if (gid >= N) return;\n"
        @"    float sum = 0.0f;\n"
        @"    for (uint k = tid; k < K; k += GOLLEK_RMS_THREADS) {\n"
        @"        sum += A[k] * float(B[k * N + gid]);\n"
        @"    }\n"
        @"    threadgroup float partial[GOLLEK_RMS_THREADS];\n"
        @"    partial[tid] = sum;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    for (uint stride = GOLLEK_RMS_THREADS >> 1; stride > 0; stride >>= 1) {\n"
        @"        if (tid < stride) partial[tid] += partial[tid + stride];\n"
        @"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    }\n"
        @"    if (tid == 0) C[gid] = partial[0];\n"
        @"}\n";
    NSError* error = nil;
    id<MTLLibrary> library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
        return;
    }
    g_add_pipeline = compile_pipeline(library, @"gollek_add_kernel");
    g_silu_ffn_pipeline = compile_pipeline(library, @"gollek_silu_ffn_kernel");
    g_gelu_ffn_pipeline = compile_pipeline(library, @"gollek_gelu_ffn_kernel");
    g_rmsnorm_pipeline = compile_pipeline(library, @"gollek_rmsnorm_kernel");
    g_matvec_half_pipeline = compile_pipeline(library, @"gollek_matvec_tb_half_kernel");
    g_matvec_t_half_pipeline = compile_pipeline(library, @"gollek_matvec_t_half_kernel");
    g_matvec_half_pair_pipeline = compile_pipeline(library, @"gollek_matvec_tb_half_pair_kernel");
    g_matvec_half_triple_mixed_pipeline = compile_pipeline(library, @"gollek_matvec_tb_half_triple_mixed_kernel");
    g_decode_attention_pipeline = compile_pipeline(library, @"gollek_decode_attention_kernel");
}

static MPSMatrixDescriptor* cached_matrix_descriptor(int rows, int cols, NSUInteger row_bytes, MPSDataType data_type) {
    if (g_disable_mps_cache) {
        return [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                     columns:cols
                                                    rowBytes:row_bytes
                                                    dataType:data_type];
    }
    NSString* key = [NSString stringWithFormat:@"%d:%d:%lu:%lu",
                     rows, cols, (unsigned long)row_bytes, (unsigned long)data_type];
    @synchronized([MPSMatrixDescriptor class]) {
        if (g_matrix_desc_cache == nil) {
            g_matrix_desc_cache = [[NSMutableDictionary alloc] init];
        }
        MPSMatrixDescriptor* desc = [g_matrix_desc_cache objectForKey:key];
        if (desc == nil) {
            desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                         columns:cols
                                                        rowBytes:row_bytes
                                                        dataType:data_type];
            [g_matrix_desc_cache setObject:desc forKey:key];
        }
        return desc;
    }
}

static MPSVectorDescriptor* cached_vector_descriptor(int length, MPSDataType data_type) {
    if (g_disable_mps_cache) {
        return [MPSVectorDescriptor vectorDescriptorWithLength:length dataType:data_type];
    }
    NSString* key = [NSString stringWithFormat:@"%d:%lu", length, (unsigned long)data_type];
    @synchronized([MPSVectorDescriptor class]) {
        if (g_vector_desc_cache == nil) {
            g_vector_desc_cache = [[NSMutableDictionary alloc] init];
        }
        MPSVectorDescriptor* desc = [g_vector_desc_cache objectForKey:key];
        if (desc == nil) {
            desc = [MPSVectorDescriptor vectorDescriptorWithLength:length dataType:data_type];
            [g_vector_desc_cache setObject:desc forKey:key];
        }
        return desc;
    }
}

static MPSMatrixMultiplication* cached_mmul(BOOL transpose_left, BOOL transpose_right,
        int result_rows, int result_cols, int interior_cols,
        MPSDataType left_type, MPSDataType right_type, MPSDataType result_type,
        float alpha, float beta) {
    if (g_disable_mps_cache) {
        return [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                                 transposeLeft:transpose_left
                                                transposeRight:transpose_right
                                                    resultRows:result_rows
                                                 resultColumns:result_cols
                                               interiorColumns:interior_cols
                                                         alpha:alpha
                                                          beta:beta];
    }
    NSString* key = [NSString stringWithFormat:@"%d:%d:%d:%d:%d:%lu:%lu:%lu:%.8g:%.8g",
                     transpose_left ? 1 : 0,
                     transpose_right ? 1 : 0,
                     result_rows,
                     result_cols,
                     interior_cols,
                     (unsigned long)left_type,
                     (unsigned long)right_type,
                     (unsigned long)result_type,
                     alpha,
                     beta];
    @synchronized([MPSMatrixMultiplication class]) {
        if (g_mmul_cache == nil) {
            g_mmul_cache = [[NSMutableDictionary alloc] init];
        }
        MPSMatrixMultiplication* mmul = [g_mmul_cache objectForKey:key];
        if (mmul == nil) {
            mmul = [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                                     transposeLeft:transpose_left
                                                    transposeRight:transpose_right
                                                        resultRows:result_rows
                                                     resultColumns:result_cols
                                                   interiorColumns:interior_cols
                                                             alpha:alpha
                                                              beta:beta];
            mmul.options = MPSKernelOptionsSkipAPIValidation;
            [g_mmul_cache setObject:mmul forKey:key];
        }
        return mmul;
    }
}

static MPSMatrixVectorMultiplication* cached_mvec(BOOL transpose,
        int rows, int columns, double alpha, double beta) {
    if (g_disable_mps_cache) {
        return [[MPSMatrixVectorMultiplication alloc] initWithDevice:g_device
                                                          transpose:transpose
                                                               rows:rows
                                                            columns:columns
                                                              alpha:alpha
                                                               beta:beta];
    }
    NSString* key = [NSString stringWithFormat:@"%d:%d:%d:%.8g:%.8g",
                     transpose ? 1 : 0, rows, columns, alpha, beta];
    @synchronized([MPSMatrixVectorMultiplication class]) {
        if (g_mvec_cache == nil) {
            g_mvec_cache = [[NSMutableDictionary alloc] init];
        }
        MPSMatrixVectorMultiplication* mvec = [g_mvec_cache objectForKey:key];
        if (mvec == nil) {
            mvec = [[MPSMatrixVectorMultiplication alloc] initWithDevice:g_device
                                                              transpose:transpose
                                                                   rows:rows
                                                                columns:columns
                                                                  alpha:alpha
                                                                   beta:beta];
            [g_mvec_cache setObject:mvec forKey:key];
        }
        return mvec;
    }
}

static float* aligned_float_alloc(size_t elements) {
    void* ptr = NULL;
    size_t bytes = elements * sizeof(float);
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        return NULL;
    }
    return (float*)ptr;
}

static void ensure_scratch(size_t required_elements) {
    if (tl_scratch_capacity < required_elements) {
        if (tl_k_scratch) free(tl_k_scratch);
        if (tl_v_scratch) free(tl_v_scratch);
        if (tl_score_scratch) free(tl_score_scratch);
        
        size_t new_cap = required_elements + (required_elements / 4); // +25%
        tl_k_scratch = aligned_float_alloc(new_cap);
        tl_v_scratch = aligned_float_alloc(new_cap);
        tl_score_scratch = aligned_float_alloc(new_cap);
        if (!tl_k_scratch || !tl_v_scratch || !tl_score_scratch) {
            if (tl_k_scratch) free(tl_k_scratch);
            if (tl_v_scratch) free(tl_v_scratch);
            if (tl_score_scratch) free(tl_score_scratch);
            tl_k_scratch = NULL;
            tl_v_scratch = NULL;
            tl_score_scratch = NULL;
            tl_scratch_capacity = 0;
            return;
        }
        tl_scratch_capacity = new_cap;
    }
}

static uint16_t* ensure_half_scratch(uint16_t** scratch, size_t* capacity, size_t required_elements) {
    if (required_elements == 0) return NULL;
    if (*capacity < required_elements || *scratch == NULL) {
        size_t new_cap = required_elements * 2;
        uint16_t* next = (uint16_t*)realloc(*scratch, new_cap * sizeof(uint16_t));
        if (next == NULL) {
            free(*scratch);
            *scratch = NULL;
            *capacity = 0;
            return NULL;
        }
        *scratch = next;
        *capacity = new_cap;
    }
    return *scratch;
}

// ── Public C API ─────────────────────────────────────────────────────────────

int gollek_metal_init(void) {
    if (g_initialized) return 0;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return -1;
    g_queue = [g_device newCommandQueue];
    const char* threshold_env = getenv("GOLLEK_METAL_SMALL_MATMUL_CPU_THRESHOLD");
    if (threshold_env != NULL && threshold_env[0] != '\0') {
        int parsed = atoi(threshold_env);
        if (parsed >= 0) {
            g_small_matmul_cpu_threshold = parsed;
        }
    }
    g_disable_mps_cache = env_truthy("GOLLEK_METAL_DISABLE_MPS_CACHE");
    g_disable_mps_matvec_after_validation_failure = NO;
    BOOL enable_elementwise_kernels = env_truthy("GOLLEK_METAL_ENABLE_ELEMENTWISE_KERNELS");
    BOOL explicit_disable_elementwise = env_truthy("GOLLEK_METAL_DISABLE_ELEMENTWISE_KERNELS");
    g_disable_elementwise_kernels = !enable_elementwise_kernels || explicit_disable_elementwise;
    compile_runtime_kernels();
    g_mmul_cache = [[NSMutableDictionary alloc] init];
    g_matrix_desc_cache = [[NSMutableDictionary alloc] init];
    g_mvec_cache = [[NSMutableDictionary alloc] init];
    g_vector_desc_cache = [[NSMutableDictionary alloc] init];
    g_mps_matvec_validated_shapes = [[NSMutableSet alloc] init];
    g_mps_matvec_failed_shapes = [[NSMutableSet alloc] init];
    g_mps_matvec_mps_preferred_shapes = [[NSMutableSet alloc] init];
    g_mps_matvec_custom_preferred_shapes = [[NSMutableSet alloc] init];
    g_initialized = YES;
    return 0;
}

long gollek_metal_available_memory(void) {
    if (!g_device) return 0;
    return (long)[g_device recommendedMaxWorkingSetSize];
}

int gollek_metal_set_mps_matvec_enabled(int enabled) {
    g_enable_mps_matvec = enabled != 0;
    return 0;
}

int gollek_metal_set_mps_matvec_autotune_enabled(int enabled) {
    g_enable_mps_matvec_autotune = enabled != 0;
    @synchronized([MPSMatrixVectorMultiplication class]) {
        [g_mps_matvec_mps_preferred_shapes removeAllObjects];
        [g_mps_matvec_custom_preferred_shapes removeAllObjects];
    }
    return 0;
}

int gollek_metal_set_mps_matvec_max_inner(int max_inner) {
    g_mps_matvec_max_inner_override = max_inner;
    return 0;
}

int gollek_metal_set_mps_matvec_max_output(int max_output) {
    g_mps_matvec_max_output_override = max_output;
    return 0;
}

int gollek_metal_set_mps_matvec_autotune_max_output(int max_output) {
    g_mps_matvec_autotune_max_output_override = max_output;
    return 0;
}

void* gollek_metal_alloc(size_t bytes, size_t align) {
    if (!g_device) return NULL;
    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return buf ? [buf contents] : NULL;
}

int gollek_metal_matmul(void* C, const void* A, const void* B,
                         int M, int K, int N,
                         float alpha, float beta) {
    if (!g_initialized) return -1;
    
    // Fallback to Accelerate for small matrices
    if (M * N < g_small_matmul_cpu_threshold) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)M * K * sizeof(float));
        id<MTLBuffer> bufB = wrap_ptr((void*)B, (size_t)K * N * sizeof(float));

        MPSMatrixDescriptor* descA = cached_matrix_descriptor(M, K, K*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descB = cached_matrix_descriptor(K, N, N*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descC = cached_matrix_descriptor(M, N, N*sizeof(float), MPSDataTypeFloat32);

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication* mmul = cached_mmul(NO, NO, M, N, K,
                MPSDataTypeFloat32, MPSDataTypeFloat32, MPSDataTypeFloat32, alpha, beta);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_matmul_tb(void* C, const void* A, const void* B,
                           int M, int K, int N,
                           float alpha, float beta) {
    if (!g_initialized) return -1;

    // B is stored as [N, K] and consumed as B^T => [K, N]
    if (M * N < g_small_matmul_cpu_threshold) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, alpha, A, K, B, K, beta, C, N);
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)M * K * sizeof(float));
        id<MTLBuffer> bufB = wrap_ptr((void*)B, (size_t)N * K * sizeof(float));

        MPSMatrixDescriptor* descA = cached_matrix_descriptor(M, K, K*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descB = cached_matrix_descriptor(N, K, K*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descC = cached_matrix_descriptor(M, N, N*sizeof(float), MPSDataTypeFloat32);

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication* mmul = cached_mmul(NO, YES, M, N, K,
                MPSDataTypeFloat32, MPSDataTypeFloat32, MPSDataTypeFloat32, alpha, beta);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_matmul_tb_half(void* C, const void* A, const void* B,
                           int M, int K, int N,
                           float alpha, float beta, int is_bf16) {
    if (!g_initialized) return -1;

    @autoreleasepool {
        int b_elem_size = 2;
        MPSDataType b_mps_type;
        if (is_bf16) {
#if defined(__MAC_14_0) || defined(__IPHONE_17_0)
            if (@available(macOS 14.0, iOS 17.0, *)) {
                b_mps_type = MPSDataTypeBFloat16;
            } else {
                return -2; // BF16 not supported
            }
#else
            return -2; // BF16 not supported
#endif
        } else {
            b_mps_type = MPSDataTypeFloat16;
        }

        id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)M * K * sizeof(float));
        id<MTLBuffer> bufB = wrap_ptr((void*)B, (size_t)N * K * b_elem_size);

        MPSMatrixDescriptor* descA = cached_matrix_descriptor(M, K, K*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descB = cached_matrix_descriptor(N, K, K*b_elem_size, b_mps_type);
        MPSMatrixDescriptor* descC = cached_matrix_descriptor(M, N, N*sizeof(float), MPSDataTypeFloat32);

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication* mmul = cached_mmul(NO, YES, M, N, K,
                MPSDataTypeFloat32, b_mps_type, MPSDataTypeFloat32, alpha, beta);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_matmul_tb_half_pair(void* C0, void* C1,
                           const void* A,
                           const void* B0, const void* B1,
                           int M, int K, int N,
                           float alpha, float beta, int is_bf16) {
    if (!g_initialized) return -1;

    @autoreleasepool {
        int b_elem_size = 2;
        MPSDataType b_mps_type;
        if (is_bf16) {
#if defined(__MAC_14_0) || defined(__IPHONE_17_0)
            if (@available(macOS 14.0, iOS 17.0, *)) {
                b_mps_type = MPSDataTypeBFloat16;
            } else {
                return -2;
            }
#else
            return -2;
#endif
        } else {
            b_mps_type = MPSDataTypeFloat16;
        }

        id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)M * N * sizeof(float));
        id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)M * N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)M * K * sizeof(float));
        id<MTLBuffer> bufB0 = wrap_ptr((void*)B0, (size_t)N * K * b_elem_size);
        id<MTLBuffer> bufB1 = wrap_ptr((void*)B1, (size_t)N * K * b_elem_size);

        MPSMatrixDescriptor* descA = cached_matrix_descriptor(M, K, K*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descB = cached_matrix_descriptor(N, K, K*b_elem_size, b_mps_type);
        MPSMatrixDescriptor* descC = cached_matrix_descriptor(M, N, N*sizeof(float), MPSDataTypeFloat32);

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix* matB0 = [[MPSMatrix alloc] initWithBuffer:bufB0 descriptor:descB];
        MPSMatrix* matB1 = [[MPSMatrix alloc] initWithBuffer:bufB1 descriptor:descB];
        MPSMatrix* matC0 = [[MPSMatrix alloc] initWithBuffer:bufC0 descriptor:descC];
        MPSMatrix* matC1 = [[MPSMatrix alloc] initWithBuffer:bufC1 descriptor:descC];

        MPSMatrixMultiplication* mmul = cached_mmul(NO, YES, M, N, K,
                MPSDataTypeFloat32, b_mps_type, MPSDataTypeFloat32, alpha, beta);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB0 resultMatrix:matC0];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB1 resultMatrix:matC1];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_matmul_tb_half_pair_mixed(void* C0, void* C1,
                           const void* A,
                           const void* B0, const void* B1,
                           int M, int K, int N0, int N1,
                           float alpha, float beta, int is_bf16) {
    if (!g_initialized) return -1;
    if (M <= 0 || K <= 0 || N0 <= 0 || N1 <= 0) return -2;

    @autoreleasepool {
        int b_elem_size = 2;
        MPSDataType b_mps_type;
        if (is_bf16) {
#if defined(__MAC_14_0) || defined(__IPHONE_17_0)
            if (@available(macOS 14.0, iOS 17.0, *)) {
                b_mps_type = MPSDataTypeBFloat16;
            } else {
                return -2;
            }
#else
            return -2;
#endif
        } else {
            b_mps_type = MPSDataTypeFloat16;
        }

        id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)M * N0 * sizeof(float));
        id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)M * N1 * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)M * K * sizeof(float));
        id<MTLBuffer> bufB0 = wrap_ptr((void*)B0, (size_t)N0 * K * b_elem_size);
        id<MTLBuffer> bufB1 = wrap_ptr((void*)B1, (size_t)N1 * K * b_elem_size);
        if (bufC0 == nil || bufC1 == nil || bufA == nil || bufB0 == nil || bufB1 == nil) {
            return -3;
        }

        MPSMatrixDescriptor* descA = cached_matrix_descriptor(M, K, K*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descB0 = cached_matrix_descriptor(N0, K, K*b_elem_size, b_mps_type);
        MPSMatrixDescriptor* descB1 = cached_matrix_descriptor(N1, K, K*b_elem_size, b_mps_type);
        MPSMatrixDescriptor* descC0 = cached_matrix_descriptor(M, N0, N0*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descC1 = cached_matrix_descriptor(M, N1, N1*sizeof(float), MPSDataTypeFloat32);

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix* matB0 = [[MPSMatrix alloc] initWithBuffer:bufB0 descriptor:descB0];
        MPSMatrix* matB1 = [[MPSMatrix alloc] initWithBuffer:bufB1 descriptor:descB1];
        MPSMatrix* matC0 = [[MPSMatrix alloc] initWithBuffer:bufC0 descriptor:descC0];
        MPSMatrix* matC1 = [[MPSMatrix alloc] initWithBuffer:bufC1 descriptor:descC1];

        MPSMatrixMultiplication* mmul0 = cached_mmul(NO, YES, M, N0, K,
                MPSDataTypeFloat32, b_mps_type, MPSDataTypeFloat32, alpha, beta);
        MPSMatrixMultiplication* mmul1 = cached_mmul(NO, YES, M, N1, K,
                MPSDataTypeFloat32, b_mps_type, MPSDataTypeFloat32, alpha, beta);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [mmul0 encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB0 resultMatrix:matC0];
        [mmul1 encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB1 resultMatrix:matC1];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_matmul_tb_half_triple_mixed(void* C0, void* C1, void* C2,
                           const void* A,
                           const void* B0, const void* B1, const void* B2,
                           int M, int K, int N0, int N1, int N2,
                           float alpha, float beta, int is_bf16) {
    if (!g_initialized) return -1;
    if (M <= 0 || K <= 0 || N0 <= 0 || N1 <= 0 || N2 <= 0) return -2;

    @autoreleasepool {
        int b_elem_size = 2;
        MPSDataType b_mps_type;
        if (is_bf16) {
#if defined(__MAC_14_0) || defined(__IPHONE_17_0)
            if (@available(macOS 14.0, iOS 17.0, *)) {
                b_mps_type = MPSDataTypeBFloat16;
            } else {
                return -2;
            }
#else
            return -2;
#endif
        } else {
            b_mps_type = MPSDataTypeFloat16;
        }

        id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)M * N0 * sizeof(float));
        id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)M * N1 * sizeof(float));
        id<MTLBuffer> bufC2 = wrap_ptr(C2, (size_t)M * N2 * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)M * K * sizeof(float));
        id<MTLBuffer> bufB0 = wrap_ptr((void*)B0, (size_t)N0 * K * b_elem_size);
        id<MTLBuffer> bufB1 = wrap_ptr((void*)B1, (size_t)N1 * K * b_elem_size);
        id<MTLBuffer> bufB2 = wrap_ptr((void*)B2, (size_t)N2 * K * b_elem_size);
        if (bufC0 == nil || bufC1 == nil || bufC2 == nil || bufA == nil
                || bufB0 == nil || bufB1 == nil || bufB2 == nil) {
            return -3;
        }

        MPSMatrixDescriptor* descA = cached_matrix_descriptor(M, K, K*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descB0 = cached_matrix_descriptor(N0, K, K*b_elem_size, b_mps_type);
        MPSMatrixDescriptor* descB1 = cached_matrix_descriptor(N1, K, K*b_elem_size, b_mps_type);
        MPSMatrixDescriptor* descB2 = cached_matrix_descriptor(N2, K, K*b_elem_size, b_mps_type);
        MPSMatrixDescriptor* descC0 = cached_matrix_descriptor(M, N0, N0*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descC1 = cached_matrix_descriptor(M, N1, N1*sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descC2 = cached_matrix_descriptor(M, N2, N2*sizeof(float), MPSDataTypeFloat32);

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix* matB0 = [[MPSMatrix alloc] initWithBuffer:bufB0 descriptor:descB0];
        MPSMatrix* matB1 = [[MPSMatrix alloc] initWithBuffer:bufB1 descriptor:descB1];
        MPSMatrix* matB2 = [[MPSMatrix alloc] initWithBuffer:bufB2 descriptor:descB2];
        MPSMatrix* matC0 = [[MPSMatrix alloc] initWithBuffer:bufC0 descriptor:descC0];
        MPSMatrix* matC1 = [[MPSMatrix alloc] initWithBuffer:bufC1 descriptor:descC1];
        MPSMatrix* matC2 = [[MPSMatrix alloc] initWithBuffer:bufC2 descriptor:descC2];

        MPSMatrixMultiplication* mmul0 = cached_mmul(NO, YES, M, N0, K,
                MPSDataTypeFloat32, b_mps_type, MPSDataTypeFloat32, alpha, beta);
        MPSMatrixMultiplication* mmul1 = cached_mmul(NO, YES, M, N1, K,
                MPSDataTypeFloat32, b_mps_type, MPSDataTypeFloat32, alpha, beta);
        MPSMatrixMultiplication* mmul2 = cached_mmul(NO, YES, M, N2, K,
                MPSDataTypeFloat32, b_mps_type, MPSDataTypeFloat32, alpha, beta);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [mmul0 encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB0 resultMatrix:matC0];
        [mmul1 encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB1 resultMatrix:matC1];
        [mmul2 encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB2 resultMatrix:matC2];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

static int gollek_metal_gated_ffn_half_impl(void* C,
                           const void* A,
                           const void* gateW,
                           const void* upW,
                           const void* downW,
                           int M, int input_dim, int intermediate_dim, int output_dim,
                           int is_bf16,
                           id<MTLComputePipelineState> activation_pipeline) {
    if (!g_initialized) return -1;
    if (activation_pipeline == nil) return -3;
    if (M <= 0 || input_dim <= 0 || intermediate_dim <= 0 || output_dim <= 0) return -2;

    @autoreleasepool {
        int b_elem_size = 2;
        MPSDataType b_mps_type;
        if (is_bf16) {
#if defined(__MAC_14_0) || defined(__IPHONE_17_0)
            if (@available(macOS 14.0, iOS 17.0, *)) {
                b_mps_type = MPSDataTypeBFloat16;
            } else {
                return -2;
            }
#else
            return -2;
#endif
        } else {
            b_mps_type = MPSDataTypeFloat16;
        }

        size_t activation_bytes = (size_t)M * intermediate_dim * sizeof(float);
        id<MTLBuffer> bufC = wrap_ptr(C, (size_t)M * output_dim * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)M * input_dim * sizeof(float));
        id<MTLBuffer> bufGateW = wrap_ptr((void*)gateW, (size_t)intermediate_dim * input_dim * b_elem_size);
        id<MTLBuffer> bufUpW = wrap_ptr((void*)upW, (size_t)intermediate_dim * input_dim * b_elem_size);
        id<MTLBuffer> bufDownW = wrap_ptr((void*)downW, (size_t)output_dim * intermediate_dim * b_elem_size);
        id<MTLBuffer> bufGate = nil;
        id<MTLBuffer> bufUp = nil;
        id<MTLBuffer> bufCombined = nil;
        if (!ensure_swiglu_scratch(activation_bytes, &bufGate, &bufUp, &bufCombined)) {
            return -4;
        }
        if (bufC == nil || bufA == nil || bufGateW == nil || bufUpW == nil || bufDownW == nil
                || bufGate == nil || bufUp == nil || bufCombined == nil) {
            return -4;
        }

        MPSMatrixDescriptor* descInput = cached_matrix_descriptor(M, input_dim,
                input_dim * sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descGateWeight = cached_matrix_descriptor(intermediate_dim, input_dim,
                input_dim * b_elem_size, b_mps_type);
        MPSMatrixDescriptor* descActivation = cached_matrix_descriptor(M, intermediate_dim,
                intermediate_dim * sizeof(float), MPSDataTypeFloat32);
        MPSMatrixDescriptor* descDownWeight = cached_matrix_descriptor(output_dim, intermediate_dim,
                intermediate_dim * b_elem_size, b_mps_type);
        MPSMatrixDescriptor* descOutput = cached_matrix_descriptor(M, output_dim,
                output_dim * sizeof(float), MPSDataTypeFloat32);

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descInput];
        MPSMatrix* matGateW = [[MPSMatrix alloc] initWithBuffer:bufGateW descriptor:descGateWeight];
        MPSMatrix* matUpW = [[MPSMatrix alloc] initWithBuffer:bufUpW descriptor:descGateWeight];
        MPSMatrix* matGate = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descActivation];
        MPSMatrix* matUp = [[MPSMatrix alloc] initWithBuffer:bufUp descriptor:descActivation];
        MPSMatrix* matCombined = [[MPSMatrix alloc] initWithBuffer:bufCombined descriptor:descActivation];
        MPSMatrix* matDownW = [[MPSMatrix alloc] initWithBuffer:bufDownW descriptor:descDownWeight];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descOutput];

        MPSMatrixMultiplication* upProj = cached_mmul(NO, YES, M, intermediate_dim, input_dim,
                MPSDataTypeFloat32, b_mps_type, MPSDataTypeFloat32, 1.0f, 0.0f);
        MPSMatrixMultiplication* downProj = cached_mmul(NO, YES, M, output_dim, intermediate_dim,
                MPSDataTypeFloat32, b_mps_type, MPSDataTypeFloat32, 1.0f, 0.0f);

        uint32_t activation_count = (uint32_t)(M * intermediate_dim);
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [upProj encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matGateW resultMatrix:matGate];
        [upProj encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matUpW resultMatrix:matUp];

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:activation_pipeline];
        [enc setBuffer:bufCombined offset:0 atIndex:0];
        [enc setBuffer:bufGate offset:0 atIndex:1];
        [enc setBuffer:bufUp offset:0 atIndex:2];
        [enc setBytes:&activation_count length:sizeof(activation_count) atIndex:3];
        NSUInteger threads = MIN((NSUInteger)256, activation_pipeline.maxTotalThreadsPerThreadgroup);
        if (threads < 1) threads = 1;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)activation_count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];

        [downProj encodeToCommandBuffer:cmd leftMatrix:matCombined rightMatrix:matDownW resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_swiglu_ffn_half(void* C,
                           const void* A,
                           const void* gateW,
                           const void* upW,
                           const void* downW,
                           int M, int input_dim, int intermediate_dim, int output_dim,
                           int is_bf16) {
    return gollek_metal_gated_ffn_half_impl(C, A, gateW, upW, downW,
            M, input_dim, intermediate_dim, output_dim, is_bf16, g_silu_ffn_pipeline);
}

int gollek_metal_geglu_ffn_half(void* C,
                           const void* A,
                           const void* gateW,
                           const void* upW,
                           const void* downW,
                           int M, int input_dim, int intermediate_dim, int output_dim,
                           int is_bf16) {
    return gollek_metal_gated_ffn_half_impl(C, A, gateW, upW, downW,
            M, input_dim, intermediate_dim, output_dim, is_bf16, g_gelu_ffn_pipeline);
}

static int gollek_metal_matvec_tb_half_custom(void* C,
                           const void* A,
                           const void* B,
                           int K, int N) {
    if (!g_initialized) return -1;
    if (g_matvec_half_pipeline == nil) return -3;
    if (K <= 0 || N <= 0) return -2;

    @autoreleasepool {
        id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)K * sizeof(float));
        id<MTLBuffer> bufB = wrap_ptr((void*)B, (size_t)N * K * sizeof(uint16_t));
        if (bufC == nil || bufA == nil || bufB == nil) {
            return -4;
        }

        uint32_t kk = (uint32_t)K;
        uint32_t nn = (uint32_t)N;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_matvec_half_pipeline];
        [enc setBuffer:bufC offset:0 atIndex:0];
        [enc setBuffer:bufA offset:0 atIndex:1];
        [enc setBuffer:bufB offset:0 atIndex:2];
        [enc setBytes:&kk length:sizeof(kk) atIndex:3];
        [enc setBytes:&nn length:sizeof(nn) atIndex:4];

        if (g_matvec_half_pipeline.maxTotalThreadsPerThreadgroup < 256) {
            return -3;
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_matvec_tb_half(void* C,
                           const void* A,
                           const void* B,
                           int K, int N) {
    if (!g_initialized) return -1;
    if (K <= 0 || N <= 0) return -2;

    int max_mps_output = g_mps_matvec_max_output_override >= 0
            ? g_mps_matvec_max_output_override
            : env_int_or_default("GOLLEK_METAL_MPS_MATVEC_MAX_OUTPUT", 640);
    int max_mps_inner = g_mps_matvec_max_inner_override >= 0
            ? g_mps_matvec_max_inner_override
            : env_int_or_default("GOLLEK_METAL_MPS_MATVEC_MAX_INNER", 2048);
    if (!g_disable_mps_matvec_after_validation_failure
            && (g_enable_mps_matvec || env_truthy("GOLLEK_METAL_ENABLE_MPS_MATVEC"))
            && (max_mps_inner <= 0 || K <= max_mps_inner)
            && (max_mps_output <= 0 || N <= max_mps_output)
            && !env_truthy("GOLLEK_METAL_DISABLE_MPS_MATVEC")) {
        @autoreleasepool {
            NSString* shapeKey = mps_matvec_shape_key(K, N);
            BOOL validateEveryCall = env_truthy("GOLLEK_METAL_MPS_MATVEC_VALIDATE_EVERY_CALL");
            int autotuneMaxOutput = g_mps_matvec_autotune_max_output_override >= 0
                    ? g_mps_matvec_autotune_max_output_override
                    : env_int_or_default("GOLLEK_METAL_MPS_MATVEC_AUTOTUNE_MAX_OUTPUT", 1024);
            BOOL autotune = (g_enable_mps_matvec_autotune || env_truthy("GOLLEK_METAL_MPS_MATVEC_AUTOTUNE"))
                    && (autotuneMaxOutput <= 0 || N <= autotuneMaxOutput);
            BOOL shapeFailed = NO;
            BOOL shapeValidated = NO;
            BOOL shapeMpsPreferred = NO;
            BOOL shapeCustomPreferred = NO;
            @synchronized([MPSMatrixVectorMultiplication class]) {
                shapeFailed = g_mps_matvec_failed_shapes != nil
                        && [g_mps_matvec_failed_shapes containsObject:shapeKey];
                shapeValidated = !validateEveryCall
                        && g_mps_matvec_validated_shapes != nil
                        && [g_mps_matvec_validated_shapes containsObject:shapeKey];
                shapeMpsPreferred = autotune
                        && g_mps_matvec_mps_preferred_shapes != nil
                        && [g_mps_matvec_mps_preferred_shapes containsObject:shapeKey];
                shapeCustomPreferred = autotune
                        && g_mps_matvec_custom_preferred_shapes != nil
                        && [g_mps_matvec_custom_preferred_shapes containsObject:shapeKey];
            }
            if (!shapeFailed && !shapeCustomPreferred) {
                uint64_t mpsStart = autotune ? monotonic_nanos() : 0;
                uint16_t* halfA = ensure_half_scratch(&tl_half_input_scratch, &tl_half_input_capacity, (size_t)K);
                uint16_t* halfC = ensure_half_scratch(&tl_half_output_scratch, &tl_half_output_capacity, (size_t)N);
                if (halfA != NULL && halfC != NULL) {
                    const float* input = (const float*)A;
                    for (int i = 0; i < K; i++) {
                        halfA[i] = f32_to_f16_bits(input[i]);
                    }
                }
                id<MTLBuffer> bufC = halfC != NULL ? wrap_ptr(halfC, (size_t)N * sizeof(uint16_t)) : nil;
                id<MTLBuffer> bufA = halfA != NULL ? wrap_ptr(halfA, (size_t)K * sizeof(uint16_t)) : nil;
                id<MTLBuffer> bufB = wrap_ptr((void*)B, (size_t)N * K * sizeof(uint16_t));
                if (bufC != nil && bufA != nil && bufB != nil) {
                    @try {
                        MPSMatrixDescriptor* descB = cached_matrix_descriptor(
                                N, K, K * sizeof(uint16_t), MPSDataTypeFloat16);
                        MPSVectorDescriptor* descA = cached_vector_descriptor(K, MPSDataTypeFloat16);
                        MPSVectorDescriptor* descC = cached_vector_descriptor(N, MPSDataTypeFloat16);
                        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
                        MPSVector* vecA = [[MPSVector alloc] initWithBuffer:bufA descriptor:descA];
                        MPSVector* vecC = [[MPSVector alloc] initWithBuffer:bufC descriptor:descC];
                        MPSMatrixVectorMultiplication* mvec = cached_mvec(NO, N, K, 1.0, 0.0);
                        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
                        [mvec encodeToCommandBuffer:cmd inputMatrix:matB inputVector:vecA resultVector:vecC];
                        [cmd commit];
                        [cmd waitUntilCompleted];
                        if ([cmd status] == MTLCommandBufferStatusCompleted) {
                            float* output = (float*)C;
                            for (int i = 0; i < N; i++) {
                                output[i] = f16_to_f32(halfC[i]);
                            }
                            uint64_t mpsNanos = autotune ? (monotonic_nanos() - mpsStart) : 0;
                            if (shapeValidated
                                    || validate_mps_matvec_half_output((const float*)C, (const float*)A, (const uint16_t*)B, K, N)) {
                                if (!shapeValidated && !validateEveryCall && !env_truthy("GOLLEK_METAL_MPS_MATVEC_SKIP_VALIDATE")) {
                                    @synchronized([MPSMatrixVectorMultiplication class]) {
                                        [g_mps_matvec_validated_shapes addObject:shapeKey];
                                    }
                                    if (env_truthy("GOLLEK_METAL_MPS_MATVEC_DEBUG")) {
                                        fprintf(stderr,
                                                "[gollek-metal] MPS matvec validated shape K=%d N=%d\n",
                                                K, N);
                                    }
                                }
                                if (autotune && !shapeMpsPreferred) {
                                    uint64_t customStart = monotonic_nanos();
                                    int customRc = gollek_metal_matvec_tb_half_custom(C, A, B, K, N);
                                    uint64_t customNanos = monotonic_nanos() - customStart;
                                    if (customRc == 0) {
                                        float margin = env_float_or_default(
                                                "GOLLEK_METAL_MPS_MATVEC_AUTOTUNE_MARGIN", 0.05f);
                                        if (margin < 0.0f) margin = 0.0f;
                                        BOOL preferMps = (double)mpsNanos < ((double)customNanos * (1.0 - (double)margin));
                                        @synchronized([MPSMatrixVectorMultiplication class]) {
                                            if (preferMps) {
                                                [g_mps_matvec_mps_preferred_shapes addObject:shapeKey];
                                            } else {
                                                [g_mps_matvec_custom_preferred_shapes addObject:shapeKey];
                                            }
                                        }
                                        if (env_truthy("GOLLEK_METAL_MPS_MATVEC_DEBUG")) {
                                            fprintf(stderr,
                                                    "[gollek-metal] MPS matvec autotune shape K=%d N=%d mps=%.3fms custom=%.3fms selected=%s\n",
                                                    K, N,
                                                    (double)mpsNanos / 1000000.0,
                                                    (double)customNanos / 1000000.0,
                                                    preferMps ? "mps" : "custom");
                                        }
                                        return 0;
                                    }
                                    @synchronized([MPSMatrixVectorMultiplication class]) {
                                        [g_mps_matvec_mps_preferred_shapes addObject:shapeKey];
                                    }
                                }
                                return 0;
                            }
                            @synchronized([MPSMatrixVectorMultiplication class]) {
                                [g_mps_matvec_failed_shapes addObject:shapeKey];
                            }
                            if (env_truthy("GOLLEK_METAL_MPS_MATVEC_DEBUG")) {
                                fprintf(stderr,
                                        "[gollek-metal] disabling MPS matvec for shape K=%d N=%d; falling back to custom Metal reduction\n",
                                        K, N);
                            }
                        } else {
                            g_disable_mps_matvec_after_validation_failure = YES;
                            if (env_truthy("GOLLEK_METAL_MPS_MATVEC_DEBUG")) {
                                NSString* message = cmd.error.localizedDescription;
                                fprintf(stderr,
                                        "[gollek-metal] MPS matvec command failed (%s); falling back to custom Metal reduction\n",
                                        message != nil ? [message UTF8String] : "unknown error");
                            }
                        }
                    } @catch (NSException* ex) {
                        // Fall back to the custom reduction kernel; correctness is more important than this fast path.
                        g_disable_mps_matvec_after_validation_failure = YES;
                        if (env_truthy("GOLLEK_METAL_MPS_MATVEC_DEBUG")) {
                            NSString* reason = [ex reason];
                            fprintf(stderr,
                                    "[gollek-metal] MPS matvec threw %s; falling back to custom Metal reduction\n",
                                    reason != nil ? [reason UTF8String] : "unknown exception");
                        }
                    }
                }
            }
        }
    }

    if (g_matvec_half_pipeline == nil) return -3;

    @autoreleasepool {
        id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)K * sizeof(float));
        id<MTLBuffer> bufB = wrap_ptr((void*)B, (size_t)N * K * sizeof(uint16_t));
        if (bufC == nil || bufA == nil || bufB == nil) {
            return -4;
        }

        uint32_t kk = (uint32_t)K;
        uint32_t nn = (uint32_t)N;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_matvec_half_pipeline];
        [enc setBuffer:bufC offset:0 atIndex:0];
        [enc setBuffer:bufA offset:0 atIndex:1];
        [enc setBuffer:bufB offset:0 atIndex:2];
        [enc setBytes:&kk length:sizeof(kk) atIndex:3];
        [enc setBytes:&nn length:sizeof(nn) atIndex:4];

        if (g_matvec_half_pipeline.maxTotalThreadsPerThreadgroup < 256) {
            return -3;
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_matvec_t_half(void* C,
                           const void* A,
                           const void* B,
                           int K, int N) {
    if (!g_initialized) return -1;
    if (g_matvec_t_half_pipeline == nil) return -3;
    if (K <= 0 || N <= 0) return -2;

    @autoreleasepool {
        id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)K * sizeof(float));
        id<MTLBuffer> bufB = wrap_ptr((void*)B, (size_t)N * K * sizeof(uint16_t));
        if (bufC == nil || bufA == nil || bufB == nil) {
            return -4;
        }

        uint32_t kk = (uint32_t)K;
        uint32_t nn = (uint32_t)N;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_matvec_t_half_pipeline];
        [enc setBuffer:bufC offset:0 atIndex:0];
        [enc setBuffer:bufA offset:0 atIndex:1];
        [enc setBuffer:bufB offset:0 atIndex:2];
        [enc setBytes:&kk length:sizeof(kk) atIndex:3];
        [enc setBytes:&nn length:sizeof(nn) atIndex:4];

        if (g_matvec_t_half_pipeline.maxTotalThreadsPerThreadgroup < 256) {
            return -3;
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_matvec_tb_half_pair(void* C0, void* C1,
                           const void* A,
                           const void* B0, const void* B1,
                           int K, int N) {
    if (!g_initialized) return -1;
    if (g_matvec_half_pair_pipeline == nil) return -3;
    if (K <= 0 || N <= 0) return -2;

    @autoreleasepool {
        id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)N * sizeof(float));
        id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)K * sizeof(float));
        id<MTLBuffer> bufB0 = wrap_ptr((void*)B0, (size_t)N * K * sizeof(uint16_t));
        id<MTLBuffer> bufB1 = wrap_ptr((void*)B1, (size_t)N * K * sizeof(uint16_t));
        if (bufC0 == nil || bufC1 == nil || bufA == nil || bufB0 == nil || bufB1 == nil) {
            return -4;
        }

        uint32_t kk = (uint32_t)K;
        uint32_t nn = (uint32_t)N;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_matvec_half_pair_pipeline];
        [enc setBuffer:bufC0 offset:0 atIndex:0];
        [enc setBuffer:bufC1 offset:0 atIndex:1];
        [enc setBuffer:bufA offset:0 atIndex:2];
        [enc setBuffer:bufB0 offset:0 atIndex:3];
        [enc setBuffer:bufB1 offset:0 atIndex:4];
        [enc setBytes:&kk length:sizeof(kk) atIndex:5];
        [enc setBytes:&nn length:sizeof(nn) atIndex:6];

        if (g_matvec_half_pair_pipeline.maxTotalThreadsPerThreadgroup < 256) {
            return -3;
        }
        MTLSize grid = MTLSizeMake((NSUInteger)N, 1, 1);
        MTLSize group = MTLSizeMake(256, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int gollek_metal_matvec_tb_half_triple_mixed(void* C0, void* C1, void* C2,
                           const void* A,
                           const void* B0, const void* B1, const void* B2,
                           int K, int N0, int N1, int N2) {
    if (!g_initialized) return -1;
    if (g_matvec_half_triple_mixed_pipeline == nil) return -3;
    if (K <= 0 || N0 <= 0 || N1 <= 0 || N2 <= 0) return -2;

    @autoreleasepool {
        id<MTLBuffer> bufC0 = wrap_ptr(C0, (size_t)N0 * sizeof(float));
        id<MTLBuffer> bufC1 = wrap_ptr(C1, (size_t)N1 * sizeof(float));
        id<MTLBuffer> bufC2 = wrap_ptr(C2, (size_t)N2 * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)K * sizeof(float));
        id<MTLBuffer> bufB0 = wrap_ptr((void*)B0, (size_t)N0 * K * sizeof(uint16_t));
        id<MTLBuffer> bufB1 = wrap_ptr((void*)B1, (size_t)N1 * K * sizeof(uint16_t));
        id<MTLBuffer> bufB2 = wrap_ptr((void*)B2, (size_t)N2 * K * sizeof(uint16_t));
        if (bufC0 == nil || bufC1 == nil || bufC2 == nil || bufA == nil
                || bufB0 == nil || bufB1 == nil || bufB2 == nil) {
            return -4;
        }

        uint32_t kk = (uint32_t)K;
        uint32_t n0 = (uint32_t)N0;
        uint32_t n1 = (uint32_t)N1;
        uint32_t n2 = (uint32_t)N2;
        NSUInteger total = (NSUInteger)N0 + (NSUInteger)N1 + (NSUInteger)N2;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_matvec_half_triple_mixed_pipeline];
        [enc setBuffer:bufC0 offset:0 atIndex:0];
        [enc setBuffer:bufC1 offset:0 atIndex:1];
        [enc setBuffer:bufC2 offset:0 atIndex:2];
        [enc setBuffer:bufA offset:0 atIndex:3];
        [enc setBuffer:bufB0 offset:0 atIndex:4];
        [enc setBuffer:bufB1 offset:0 atIndex:5];
        [enc setBuffer:bufB2 offset:0 atIndex:6];
        [enc setBytes:&kk length:sizeof(kk) atIndex:7];
        [enc setBytes:&n0 length:sizeof(n0) atIndex:8];
        [enc setBytes:&n1 length:sizeof(n1) atIndex:9];
        [enc setBytes:&n2 length:sizeof(n2) atIndex:10];

        if (g_matvec_half_triple_mixed_pipeline.maxTotalThreadsPerThreadgroup < 256) {
            return -3;
        }
        [enc dispatchThreadgroups:MTLSizeMake(total, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

static int gollek_metal_decode_attention_gpu(
        void* out,
        const void* Q, const void* K_cache, const void* V_cache,
        const int* block_table, const int* context_lens,
        int B, int H, int H_kv, int D, int block_size, int max_blocks,
        float scale, int is_causal, int query_start_pos, int sliding_window, float soft_cap) {
    if (!g_initialized || g_decode_attention_pipeline == nil) return -3;
    if (!env_truthy("GOLLEK_METAL_ENABLE_DECODE_ATTENTION_KERNEL")) return -9;
    if (env_truthy("GOLLEK_METAL_DISABLE_DECODE_ATTENTION_KERNEL")) return -9;
    if (B <= 0 || H <= 0 || H_kv <= 0 || D <= 0 || block_size <= 0 || max_blocks <= 0) return -2;
    if (H % H_kv != 0) return -2;
    if (D > 256) return -8;

    int max_context = env_int_or_default("GOLLEK_METAL_DECODE_ATTENTION_MAX_CONTEXT", 4096);
    if (max_context <= 0 || max_context > 4096) {
        max_context = 4096;
    }

    int max_phys = -1;
    for (int b = 0; b < B; b++) {
        int ctx_len = context_lens[b];
        if (ctx_len <= 0) continue;
        int effective_query_start_pos = query_start_pos;
        if (is_causal && sliding_window <= 0 && query_start_pos == 0) {
            effective_query_start_pos = ctx_len - 1;
        }
        int max_pos = is_causal ? effective_query_start_pos : (ctx_len - 1);
        if (max_pos >= ctx_len) max_pos = ctx_len - 1;
        int min_pos = sliding_window > 0 ? (effective_query_start_pos - sliding_window + 1) : 0;
        if (min_pos < 0) min_pos = 0;
        int valid = max_pos >= min_pos ? (max_pos - min_pos + 1) : 0;
        if (valid <= 0) continue;
        if (valid > max_context) return -8;

        int blocks = (ctx_len + block_size - 1) / block_size;
        if (blocks > max_blocks) return -2;
        for (int blk = 0; blk < blocks; blk++) {
            int phys = block_table[b * max_blocks + blk];
            if (phys < 0) return -2;
            if (phys > max_phys) max_phys = phys;
        }
    }
    if (max_phys < 0) {
        memset(out, 0, (size_t)B * H * D * sizeof(float));
        return 0;
    }

    @autoreleasepool {
        size_t q_bytes = (size_t)B * H * D * sizeof(float);
        size_t out_bytes = q_bytes;
        size_t kv_bytes = (size_t)(max_phys + 1) * H_kv * block_size * D * sizeof(float);
        size_t block_table_bytes = (size_t)B * max_blocks * sizeof(int);
        size_t context_lens_bytes = (size_t)B * sizeof(int);

        id<MTLBuffer> bufOut = wrap_ptr(out, out_bytes);
        id<MTLBuffer> bufQ = wrap_ptr((void*)Q, q_bytes);
        id<MTLBuffer> bufK = wrap_ptr((void*)K_cache, kv_bytes);
        id<MTLBuffer> bufV = wrap_ptr((void*)V_cache, kv_bytes);
        id<MTLBuffer> bufBlockTable = wrap_ptr((void*)block_table, block_table_bytes);
        id<MTLBuffer> bufContextLens = wrap_ptr((void*)context_lens, context_lens_bytes);
        if (bufOut == nil || bufQ == nil || bufK == nil || bufV == nil
                || bufBlockTable == nil || bufContextLens == nil) {
            return -4;
        }

        uint32_t h = (uint32_t)H;
        uint32_t hkv = (uint32_t)H_kv;
        uint32_t d = (uint32_t)D;
        uint32_t bs = (uint32_t)block_size;
        uint32_t mb = (uint32_t)max_blocks;
        uint32_t causal = is_causal ? 1u : 0u;
        int32_t qpos = (int32_t)query_start_pos;
        int32_t window = (int32_t)sliding_window;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_decode_attention_pipeline];
        [enc setBuffer:bufOut offset:0 atIndex:0];
        [enc setBuffer:bufQ offset:0 atIndex:1];
        [enc setBuffer:bufK offset:0 atIndex:2];
        [enc setBuffer:bufV offset:0 atIndex:3];
        [enc setBuffer:bufBlockTable offset:0 atIndex:4];
        [enc setBuffer:bufContextLens offset:0 atIndex:5];
        [enc setBytes:&h length:sizeof(h) atIndex:6];
        [enc setBytes:&hkv length:sizeof(hkv) atIndex:7];
        [enc setBytes:&d length:sizeof(d) atIndex:8];
        [enc setBytes:&bs length:sizeof(bs) atIndex:9];
        [enc setBytes:&mb length:sizeof(mb) atIndex:10];
        [enc setBytes:&scale length:sizeof(scale) atIndex:11];
        [enc setBytes:&causal length:sizeof(causal) atIndex:12];
        [enc setBytes:&qpos length:sizeof(qpos) atIndex:13];
        [enc setBytes:&window length:sizeof(window) atIndex:14];
        [enc setBytes:&soft_cap length:sizeof(soft_cap) atIndex:15];

        if (g_decode_attention_pipeline.maxTotalThreadsPerThreadgroup < 256) {
            return -3;
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)B * (NSUInteger)H, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

static inline void resolve_attention_bounds(int ctx_len, int is_causal, int query_start_pos, int query_idx,
        int sliding_window, int* min_pos, int* max_pos) {
    int abs_query_pos = query_start_pos + query_idx;
    int upper = is_causal ? abs_query_pos : (ctx_len - 1);
    if (upper >= ctx_len) upper = ctx_len - 1;
    int lower = sliding_window > 0 ? (abs_query_pos - sliding_window + 1) : 0;
    if (lower < 0) lower = 0;
    *min_pos = lower;
    *max_pos = upper;
}

static int gollek_metal_attention_impl(
        void* out,
        const void* Q, const void* K_cache, const void* V_cache,
        const int* block_table, const int* context_lens,
        int B, int T, int H, int H_kv, int D, int block_size, int max_blocks,
        float scale, int is_causal, int query_start_pos, int sliding_window, float soft_cap) {

    if (!g_initialized) return -1;
    if (H_kv <= 0 || H % H_kv != 0) return -1;

    int gqa_group = H / H_kv;
    if (T == 1) {
        int gpu_rc = gollek_metal_decode_attention_gpu(
                out, Q, K_cache, V_cache, block_table, context_lens,
                B, H, H_kv, D, block_size, max_blocks,
                scale, is_causal, query_start_pos, sliding_window, soft_cap);
        if (gpu_rc == 0) {
            return 0;
        }
        if (gpu_rc != -9 && env_truthy("GOLLEK_METAL_DECODE_ATTENTION_DEBUG")) {
            fprintf(stderr,
                    "[gollek-metal] decode attention GPU kernel unavailable rc=%d; falling back to CPU attention bridge\n",
                    gpu_rc);
        }
    }
    BOOL use_decode_sgemv = (T == 1) && !env_truthy("GOLLEK_METAL_DISABLE_ATTENTION_SGEMV");

    for (int b = 0; b < B; b++) {
        int ctx_len = context_lens[b];
        int num_blocks = (ctx_len + block_size - 1) / block_size;
        size_t kv_elements = (size_t)ctx_len * H_kv * D;
        size_t score_elements = (size_t)H * T * ctx_len;
        ensure_scratch(score_elements > kv_elements ? score_elements : kv_elements);

        float* kPtr = tl_k_scratch;
        float* vPtr = tl_v_scratch;
        float* scoreBuf = tl_score_scratch;
        const float* kcache = (const float*)K_cache;
        const float* vcache = (const float*)V_cache;

        for (int blk = 0; blk < num_blocks; blk++) {
            int phys = block_table[b * max_blocks + blk];
            int tokens_in_blk = (blk == num_blocks - 1) ? (ctx_len - blk * block_size) : block_size;

            for (int h = 0; h < H_kv; h++) {
                const float* src_k = kcache + ((size_t)phys * H_kv + h) * block_size * D;
                const float* src_v = vcache + ((size_t)phys * H_kv + h) * block_size * D;
                float* dst_k = kPtr + ((size_t)h * ctx_len + blk * block_size) * D;
                float* dst_v = vPtr + ((size_t)h * ctx_len + blk * block_size) * D;
                memcpy(dst_k, src_k, (size_t)tokens_in_blk * D * sizeof(float));
                memcpy(dst_v, src_v, (size_t)tokens_in_blk * D * sizeof(float));
            }
        }

        const float* qPtr = (const float*)Q + (size_t)b * T * H * D;
        for (int h = 0; h < H; h++) {
            int kv_h = h / gqa_group;
            for (int t = 0; t < T; t++) {
                const float* qh = qPtr + (t * H + h) * D;
                float* row = scoreBuf + (h * T + t) * ctx_len;
                int min_pos = 0;
                int max_pos = ctx_len - 1;
                int effective_query_start_pos = query_start_pos;
                if (is_causal && sliding_window <= 0 && query_start_pos == 0 && ctx_len >= T) {
                    effective_query_start_pos = ctx_len - T;
                }
                resolve_attention_bounds(ctx_len, is_causal, effective_query_start_pos, t, sliding_window, &min_pos, &max_pos);

                float* oh = (float*)out + (b * T * H + t * H + h) * D;
                memset(oh, 0, (size_t)D * sizeof(float));
                if (max_pos < min_pos) {
                    continue;
                }

                if (use_decode_sgemv) {
                    int valid_len = max_pos - min_pos + 1;
                    const float* kBase = kPtr + ((size_t)kv_h * ctx_len + min_pos) * D;
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                valid_len, D,
                                scale,
                                kBase, D,
                                qh, 1,
                                0.0f,
                                row + min_pos, 1);

                    float mx = -1e30f;
                    for (int s = min_pos; s <= max_pos; s++) {
                        float score = row[s];
                        if (soft_cap > 0.0f) {
                            score = soft_cap * tanhf(score / soft_cap);
                            row[s] = score;
                        }
                        if (score > mx) mx = score;
                    }

                    float sum = 0.0f;
                    for (int s = min_pos; s <= max_pos; s++) {
                        row[s] = expf(row[s] - mx);
                        sum += row[s];
                    }
                    float inv_sum = 1.0f / (sum + 1e-9f);
                    for (int s = min_pos; s <= max_pos; s++) {
                        row[s] *= inv_sum;
                    }

                    const float* vBase = vPtr + ((size_t)kv_h * ctx_len + min_pos) * D;
                    cblas_sgemv(CblasRowMajor, CblasTrans,
                                valid_len, D,
                                1.0f,
                                vBase, D,
                                row + min_pos, 1,
                                0.0f,
                                oh, 1);
                    continue;
                }

                float mx = -1e30f;
                for (int s = min_pos; s <= max_pos; s++) {
                    const float* kh = kPtr + ((size_t)kv_h * ctx_len + s) * D;
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) dot += qh[d] * kh[d];

                    float score = dot * scale;
                    if (soft_cap > 0.0f) {
                        score = soft_cap * tanhf(score / soft_cap);
                    }
                    row[s] = score;
                    if (score > mx) mx = score;
                }

                float sum = 0.0f;
                for (int s = min_pos; s <= max_pos; s++) {
                    row[s] = expf(row[s] - mx);
                    sum += row[s];
                }
                float inv_sum = 1.0f / (sum + 1e-9f);
                for (int s = min_pos; s <= max_pos; s++) {
                    row[s] *= inv_sum;
                }

                for (int s = min_pos; s <= max_pos; s++) {
                    const float* vh = vPtr + ((size_t)kv_h * ctx_len + s) * D;
                    float weight = row[s];
                    for (int d = 0; d < D; d++) oh[d] += weight * vh[d];
                }
            }
        }
    }
    return 0;
}

int gollek_metal_attention(
        void* out,
        const void* Q, const void* K_cache, const void* V_cache,
        const int* block_table, const int* context_lens,
        int B, int T, int H, int D, int block_size, int max_blocks,
        float scale, int is_causal, float soft_cap) {
    return gollek_metal_attention_impl(out, Q, K_cache, V_cache, block_table, context_lens,
            B, T, H, H, D, block_size, max_blocks, scale, is_causal, 0, 0, soft_cap);
}

int gollek_metal_attention_windowed(
        void* out,
        const void* Q, const void* K_cache, const void* V_cache,
        const int* block_table, const int* context_lens,
        int B, int T, int H, int D, int block_size, int max_blocks,
        float scale, int is_causal, int query_start_pos, int sliding_window, float soft_cap) {
    return gollek_metal_attention_impl(out, Q, K_cache, V_cache, block_table, context_lens,
            B, T, H, H, D, block_size, max_blocks, scale, is_causal, query_start_pos, sliding_window, soft_cap);
}

int gollek_metal_attention_gqa(
        void* out,
        const void* Q, const void* K_cache, const void* V_cache,
        const int* block_table, const int* context_lens,
        int B, int T, int H, int H_kv, int D, int block_size, int max_blocks,
        float scale, int is_causal, float soft_cap) {
    return gollek_metal_attention_impl(out, Q, K_cache, V_cache, block_table, context_lens,
            B, T, H, H_kv, D, block_size, max_blocks, scale, is_causal, 0, 0, soft_cap);
}

int gollek_metal_attention_gqa_windowed(
        void* out,
        const void* Q, const void* K_cache, const void* V_cache,
        const int* block_table, const int* context_lens,
        int B, int T, int H, int H_kv, int D, int block_size, int max_blocks,
        float scale, int is_causal, int query_start_pos, int sliding_window, float soft_cap) {
    return gollek_metal_attention_impl(out, Q, K_cache, V_cache, block_table, context_lens,
            B, T, H, H_kv, D, block_size, max_blocks, scale, is_causal, query_start_pos, sliding_window, soft_cap);
}

static int cpu_add(void* C, const void* A, const void* B, int N) {
    const float* a = (const float*)A;
    const float* b = (const float*)B;
    float* c = (float*)C;
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    return 0;
}

static int cpu_rmsnorm(void* out, const void* x, const void* weight, int N, float eps, int addOne) {
    const float* xi = (const float*)x;
    const float* wi = (const float*)weight;
    float* oi = (float*)out;
    float ss = 0;
    for (int i = 0; i < N; i++) ss += xi[i] * xi[i];
    float inv = 1.0f / sqrtf(ss / N + eps);
    for (int i = 0; i < N; i++) {
        float w = wi[i];
        if (addOne) w += 1.0f;
        oi[i] = xi[i] * inv * w;
    }
    return 0;
}

static int cpu_silu_ffn(void* out, const void* gate, const void* up, int N) {
    const float* g = (const float*)gate;
    const float* u = (const float*)up;
    float* o = (float*)out;
    for (int i = 0; i < N; i++) {
        float gi = g[i];
        o[i] = (gi / (1.0f + expf(-gi))) * u[i];
    }
    return 0;
}

static int cpu_gelu_ffn(void* out, const void* gate, const void* up, int N) {
    const float* g = (const float*)gate;
    const float* u = (const float*)up;
    float* o = (float*)out;
    for (int i = 0; i < N; i++) {
        float gi = g[i];
        float inner = 0.79788456f * (gi + 0.044715f * gi * gi * gi);
        float gelu = 0.5f * gi * (1.0f + tanhf(inner));
        o[i] = gelu * u[i];
    }
    return 0;
}

int gollek_metal_add(void* C, const void* A, const void* B, int N) {
    if (!g_initialized || N <= 0) return cpu_add(C, A, B, N);
    if (g_disable_elementwise_kernels || g_add_pipeline == nil) return cpu_add(C, A, B, N);

    @autoreleasepool {
        id<MTLBuffer> bufC = wrap_ptr(C, (size_t)N * sizeof(float));
        id<MTLBuffer> bufA = wrap_ptr((void*)A, (size_t)N * sizeof(float));
        id<MTLBuffer> bufB = wrap_ptr((void*)B, (size_t)N * sizeof(float));
        if (bufC == nil || bufA == nil || bufB == nil) return cpu_add(C, A, B, N);

        unsigned int n = (unsigned int)N;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_add_pipeline];
        [enc setBuffer:bufC offset:0 atIndex:0];
        [enc setBuffer:bufA offset:0 atIndex:1];
        [enc setBuffer:bufB offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        NSUInteger threads = MIN((NSUInteger)256, g_add_pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return cmd.error == nil ? 0 : -5;
    }
}

int gollek_metal_rmsnorm(void* out, const void* x, const void* weight, int N, float eps, int addOne) {
    if (!g_initialized || N <= 0) return cpu_rmsnorm(out, x, weight, N, eps, addOne);
    if (g_disable_elementwise_kernels || g_rmsnorm_pipeline == nil
            || g_rmsnorm_pipeline.maxTotalThreadsPerThreadgroup < 256) {
        return cpu_rmsnorm(out, x, weight, N, eps, addOne);
    }

    @autoreleasepool {
        id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
        id<MTLBuffer> bufX = wrap_ptr((void*)x, (size_t)N * sizeof(float));
        id<MTLBuffer> bufWeight = wrap_ptr((void*)weight, (size_t)N * sizeof(float));
        if (bufOut == nil || bufX == nil || bufWeight == nil) {
            return cpu_rmsnorm(out, x, weight, N, eps, addOne);
        }

        unsigned int n = (unsigned int)N;
        unsigned int add = addOne ? 1u : 0u;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_rmsnorm_pipeline];
        [enc setBuffer:bufOut offset:0 atIndex:0];
        [enc setBuffer:bufX offset:0 atIndex:1];
        [enc setBuffer:bufWeight offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        [enc setBytes:&eps length:sizeof(eps) atIndex:4];
        [enc setBytes:&add length:sizeof(add) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return cmd.error == nil ? 0 : -5;
    }
}

int gollek_metal_silu_ffn(void* out, const void* gate, const void* up, int N) {
    if (!g_initialized || N <= 0) return cpu_silu_ffn(out, gate, up, N);
    if (g_disable_elementwise_kernels || g_silu_ffn_pipeline == nil) return cpu_silu_ffn(out, gate, up, N);

    @autoreleasepool {
        id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
        id<MTLBuffer> bufGate = wrap_ptr((void*)gate, (size_t)N * sizeof(float));
        id<MTLBuffer> bufUp = wrap_ptr((void*)up, (size_t)N * sizeof(float));
        if (bufOut == nil || bufGate == nil || bufUp == nil) return cpu_silu_ffn(out, gate, up, N);

        unsigned int n = (unsigned int)N;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_silu_ffn_pipeline];
        [enc setBuffer:bufOut offset:0 atIndex:0];
        [enc setBuffer:bufGate offset:0 atIndex:1];
        [enc setBuffer:bufUp offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        NSUInteger threads = MIN((NSUInteger)256, g_silu_ffn_pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return cmd.error == nil ? 0 : -5;
    }
}

int gollek_metal_gelu_ffn(void* out, const void* gate, const void* up, int N) {
    if (!g_initialized || N <= 0) return cpu_gelu_ffn(out, gate, up, N);
    if (g_disable_elementwise_kernels || g_gelu_ffn_pipeline == nil) return cpu_gelu_ffn(out, gate, up, N);

    @autoreleasepool {
        id<MTLBuffer> bufOut = wrap_ptr(out, (size_t)N * sizeof(float));
        id<MTLBuffer> bufGate = wrap_ptr((void*)gate, (size_t)N * sizeof(float));
        id<MTLBuffer> bufUp = wrap_ptr((void*)up, (size_t)N * sizeof(float));
        if (bufOut == nil || bufGate == nil || bufUp == nil) return cpu_gelu_ffn(out, gate, up, N);

        unsigned int n = (unsigned int)N;
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_gelu_ffn_pipeline];
        [enc setBuffer:bufOut offset:0 atIndex:0];
        [enc setBuffer:bufGate offset:0 atIndex:1];
        [enc setBuffer:bufUp offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        NSUInteger threads = MIN((NSUInteger)256, g_gelu_ffn_pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return cmd.error == nil ? 0 : -5;
    }
}

int gollek_metal_device_name(char* buf, int bufSz) {
    if (!g_device) return -1;
    snprintf(buf, bufSz, "%s", [[g_device name] UTF8String]);
    return 0;
}

int gollek_metal_is_unified_memory(void) {
    return (g_device && [g_device hasUnifiedMemory]) ? 1 : 0;
}

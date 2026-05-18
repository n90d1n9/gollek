/**
 * gollek_metal_runtime.m — exported runtime/device helpers for the Metal dylib.
 */

#import "gollek_metal_runtime.h"
#import "gollek_metal_elementwise.h"
#import "gollek_metal_mps_cache.h"
#import "gollek_metal_mps_matmul.h"
#import "gollek_metal_mps_matvec_policy.h"
#import "gollek_metal_pipelines.h"
#import "gollek_metal_support.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static NSMutableArray<id<MTLBuffer>>* g_external_allocations = nil;

static NSMutableArray<id<MTLBuffer>>* retained_external_allocations(void) {
    if (g_external_allocations == nil) {
        g_external_allocations = [[NSMutableArray alloc] init];
    }
    return g_external_allocations;
}

int gollek_metal_init(void) {
    if (g_initialized) return 0;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return -1;
    g_queue = [g_device newCommandQueue];
    gollek_metal_mps_matmul_configure_from_env();
    gollek_metal_mps_cache_init(gollek_metal_env_truthy("GOLLEK_METAL_DISABLE_MPS_CACHE"));
    gollek_metal_mps_matvec_policy_init();
    BOOL enable_elementwise_kernels = gollek_metal_env_truthy("GOLLEK_METAL_ENABLE_ELEMENTWISE_KERNELS");
    BOOL explicit_disable_elementwise = gollek_metal_env_truthy("GOLLEK_METAL_DISABLE_ELEMENTWISE_KERNELS");
    gollek_metal_elementwise_set_enabled(enable_elementwise_kernels && !explicit_disable_elementwise);
    gollek_metal_compile_runtime_pipelines(gollek_metal_pipelines());
    g_initialized = YES;
    return 0;
}

long gollek_metal_available_memory(void) {
    if (!g_device) return 0;
    return (long)[g_device recommendedMaxWorkingSetSize];
}

int gollek_metal_set_mps_matvec_enabled(int enabled) {
    return gollek_metal_mps_matvec_set_enabled(enabled);
}

int gollek_metal_set_mps_matvec_autotune_enabled(int enabled) {
    return gollek_metal_mps_matvec_set_autotune_enabled(enabled);
}

int gollek_metal_set_mps_matvec_max_inner(int max_inner) {
    return gollek_metal_mps_matvec_set_max_inner(max_inner);
}

int gollek_metal_set_mps_matvec_max_output(int max_output) {
    return gollek_metal_mps_matvec_set_max_output(max_output);
}

int gollek_metal_set_mps_matvec_autotune_max_output(int max_output) {
    return gollek_metal_mps_matvec_set_autotune_max_output(max_output);
}

void* gollek_metal_alloc(size_t bytes, size_t align) {
    (void)align;
    if (!g_device || bytes == 0) return NULL;
    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buf == nil) return NULL;
    NSMutableArray<id<MTLBuffer>>* allocations = retained_external_allocations();
    @synchronized(allocations) {
        [allocations addObject:buf];
    }
    return [buf contents];
}

static inline BOOL gollek_metal_argmax_rejected(int id,
                                                int reject0,
                                                int reject1,
                                                int reject2,
                                                int reject3,
                                                int reject4,
                                                int reject5,
                                                int reject6,
                                                int reject7) {
    return id == reject0 || id == reject1 || id == reject2 || id == reject3
            || id == reject4 || id == reject5 || id == reject6 || id == reject7;
}

int gollek_metal_argmax_f32(const void* logits,
                            int n,
                            int reject0,
                            int reject1,
                            int reject2,
                            int reject3,
                            int reject4,
                            int reject5,
                            int reject6,
                            int reject7) {
    if (logits == NULL || n <= 0) return -1;
    const float* values = (const float*)logits;
    int best = -1;
    float best_val = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (gollek_metal_argmax_rejected(i, reject0, reject1, reject2, reject3, reject4, reject5, reject6, reject7)) {
            continue;
        }
        float value = values[i];
        if (isnan(value)) {
            continue;
        }
        if (value > best_val) {
            best_val = value;
            best = i;
        }
    }
    return best;
}

int gollek_metal_device_name(char* buf, int bufSz) {
    if (!g_device || buf == NULL || bufSz <= 0) return -1;
    snprintf(buf, (size_t)bufSz, "%s", [[g_device name] UTF8String]);
    return 0;
}

int gollek_metal_is_unified_memory(void) {
    return (g_device && [g_device hasUnifiedMemory]) ? 1 : 0;
}

/**
 * gollek_metal_pipelines.m — runtime Metal pipeline compilation and storage.
 */

#import "gollek_metal_pipelines.h"
#import "gollek_metal_kernel_source.h"
#import "gollek_metal_matvec_tuning.h"
#import "gollek_metal_support.h"

#include <stdlib.h>
#include <string.h>

static GollekMetalPipelines g_pipelines;

GollekMetalPipelines* gollek_metal_pipelines(void) {
    return &g_pipelines;
}

static id<MTLComputePipelineState> compile_pipeline(id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (fn == nil) {
        return nil;
    }
    return [g_device newComputePipelineStateWithFunction:fn error:&error];
}

static BOOL should_compile_128_matvec_pipelines(void) {
    if (gollek_metal_env_truthy("GOLLEK_METAL_DISABLE_MATVEC_128")) {
        return NO;
    }
    if (gollek_metal_env_truthy("GOLLEK_METAL_ENABLE_MATVEC_128_PIPELINES")
            || gollek_metal_env_truthy("GOLLEK_METAL_ENABLE_MATVEC_AUTOTUNE")) {
        return YES;
    }
    const char* forced = getenv("GOLLEK_METAL_MATVEC_THREADS");
    return forced != NULL && strcmp(forced, "128") == 0;
}

static BOOL should_compile_simdgroup_reduction_pipelines(void) {
    return !gollek_metal_env_truthy("GOLLEK_METAL_DISABLE_SIMDGROUP_REDUCTION")
            && gollek_metal_env_truthy("GOLLEK_METAL_ENABLE_SIMDGROUP_REDUCTION");
}

static BOOL should_compile_bf16_fused_gated_pair_pipeline(void) {
    return gollek_metal_env_truthy("GOLLEK_METAL_ENABLE_FUSED_GATED_FFN_MATVEC");
}

static BOOL should_compile_bf16_fused_gated_pair_x4_pipeline(void) {
    return gollek_metal_env_truthy("GOLLEK_METAL_ENABLE_BF16_FUSED_GATED_PAIR_X4");
}

static BOOL should_compile_bf16_x8_pipeline(void) {
    return gollek_metal_env_truthy("GOLLEK_METAL_ENABLE_BF16_MATVEC_X8")
            && !gollek_metal_env_truthy("GOLLEK_METAL_DISABLE_BF16_MATVEC_X8");
}

void gollek_metal_compile_runtime_pipelines(GollekMetalPipelines* pipelines) {
    if (pipelines == NULL || g_device == nil) {
        return;
    }

    NSError* error = nil;
    id<MTLLibrary> library = [g_device newLibraryWithSource:gollek_metal_runtime_kernel_source()
                                                    options:nil
                                                      error:&error];
    if (library == nil) {
        return;
    }

    pipelines->add = compile_pipeline(library, @"gollek_add_kernel");
    pipelines->silu_ffn = compile_pipeline(library, @"gollek_silu_ffn_kernel");
    pipelines->gelu_ffn = compile_pipeline(library, @"gollek_gelu_ffn_kernel");
    pipelines->rmsnorm = compile_pipeline(library, @"gollek_rmsnorm_kernel");
    pipelines->rmsnorm_rows = compile_pipeline(library, @"gollek_rmsnorm_rows_kernel");
    pipelines->matvec_half = compile_pipeline(library, @"gollek_matvec_tb_half_kernel");
    pipelines->matvec_t_half = compile_pipeline(library, @"gollek_matvec_t_half_kernel");
    pipelines->matvec_half_pair = compile_pipeline(library, @"gollek_matvec_tb_half_pair_kernel");
    pipelines->matvec_half_gated_pair = compile_pipeline(library, @"gollek_matvec_tb_half_gated_pair_kernel");
    pipelines->matvec_half_triple_mixed = compile_pipeline(library, @"gollek_matvec_tb_half_triple_mixed_kernel");
    pipelines->decode_attention = compile_pipeline(library, @"gollek_decode_attention_kernel");

    NSError* matvec256Error = nil;
    id<MTLLibrary> matvec256Library = [g_device newLibraryWithSource:gollek_metal_matvec_kernel_source(GOLLEK_MATVEC_THREADS_256)
                                                             options:nil
                                                               error:&matvec256Error];
    if (matvec256Library != nil) {
        pipelines->matvec_bf16 = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_kernel");
        pipelines->matvec_bf16_pair = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_pair_kernel");
        pipelines->matvec_bf16_pair_simd = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_pair_simd_kernel");
        pipelines->matvec_bf16_x4 = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_x4_kernel");
        if (should_compile_bf16_x8_pipeline()) {
            pipelines->matvec_bf16_x8 = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_x8_kernel");
        }
        pipelines->matvec_bf16_pair_x4 = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_pair_x4_kernel");
        if (should_compile_simdgroup_reduction_pipelines()) {
            pipelines->matvec_bf16_x4_simd = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_x4_simd_kernel");
            pipelines->matvec_bf16_pair_x4_simd = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_pair_x4_simd_kernel");
        }
        if (should_compile_bf16_fused_gated_pair_pipeline()) {
            pipelines->matvec_bf16_gated_pair = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_gated_pair_kernel");
            if (!gollek_metal_env_truthy("GOLLEK_METAL_DISABLE_BF16_FUSED_GATED_PAIR_SIMD")) {
                pipelines->matvec_bf16_gated_pair_simd = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_gated_pair_simd_kernel");
            }
        }
        if (should_compile_bf16_fused_gated_pair_x4_pipeline()) {
            pipelines->matvec_bf16_gated_pair_x4 = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_gated_pair_x4_kernel");
        }
        pipelines->matvec_bf16_triple_mixed = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_triple_mixed_kernel");
        pipelines->matvec_bf16_triple_mixed_x4 = compile_pipeline(matvec256Library, @"gollek_matvec_tb_bf16_triple_mixed_x4_kernel");
    }

    if (should_compile_128_matvec_pipelines()) {
        NSError* matvec128Error = nil;
        id<MTLLibrary> matvec128Library = [g_device newLibraryWithSource:gollek_metal_matvec_kernel_source(GOLLEK_MATVEC_THREADS_128)
                                                                 options:nil
                                                                   error:&matvec128Error];
        if (matvec128Library != nil) {
            pipelines->matvec_half_128 = compile_pipeline(matvec128Library, @"gollek_matvec_tb_half_kernel");
            pipelines->matvec_t_half_128 = compile_pipeline(matvec128Library, @"gollek_matvec_t_half_kernel");
            pipelines->matvec_half_pair_128 = compile_pipeline(matvec128Library, @"gollek_matvec_tb_half_pair_kernel");
            pipelines->matvec_half_gated_pair_128 = compile_pipeline(matvec128Library, @"gollek_matvec_tb_half_gated_pair_kernel");
            pipelines->matvec_half_triple_mixed_128 = compile_pipeline(matvec128Library, @"gollek_matvec_tb_half_triple_mixed_kernel");
            pipelines->matvec_bf16_128 = compile_pipeline(matvec128Library, @"gollek_matvec_tb_bf16_kernel");
            pipelines->matvec_bf16_pair_128 = compile_pipeline(matvec128Library, @"gollek_matvec_tb_bf16_pair_kernel");
            pipelines->matvec_bf16_gated_pair_128 = compile_pipeline(matvec128Library, @"gollek_matvec_tb_bf16_gated_pair_kernel");
            pipelines->matvec_bf16_triple_mixed_128 = compile_pipeline(matvec128Library, @"gollek_matvec_tb_bf16_triple_mixed_kernel");
        }
    }
}

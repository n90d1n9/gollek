/**
 * gollek_metal_pipelines.h — runtime Metal pipeline compilation and storage.
 */

#ifndef GOLLEK_METAL_PIPELINES_H
#define GOLLEK_METAL_PIPELINES_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef struct {
    __strong id<MTLComputePipelineState> matvec_half;
    __strong id<MTLComputePipelineState> matvec_t_half;
    __strong id<MTLComputePipelineState> matvec_half_pair;
    __strong id<MTLComputePipelineState> matvec_half_triple_mixed;
    __strong id<MTLComputePipelineState> matvec_bf16;
    __strong id<MTLComputePipelineState> matvec_bf16_pair;
    __strong id<MTLComputePipelineState> matvec_bf16_pair_simd;
    __strong id<MTLComputePipelineState> matvec_bf16_triple_mixed;
    __strong id<MTLComputePipelineState> matvec_bf16_triple_mixed_x4;
    __strong id<MTLComputePipelineState> matvec_bf16_x4;
    __strong id<MTLComputePipelineState> matvec_bf16_x8;
    __strong id<MTLComputePipelineState> matvec_bf16_pair_x4;
    __strong id<MTLComputePipelineState> matvec_bf16_x4_simd;
    __strong id<MTLComputePipelineState> matvec_bf16_pair_x4_simd;
    __strong id<MTLComputePipelineState> matvec_half_gated_pair;
    __strong id<MTLComputePipelineState> matvec_bf16_gated_pair;
    __strong id<MTLComputePipelineState> matvec_bf16_gated_pair_x4;
    __strong id<MTLComputePipelineState> matvec_bf16_gated_pair_simd;
    __strong id<MTLComputePipelineState> matvec_bf16_rows_gated_pair;
    __strong id<MTLComputePipelineState> matvec_bf16_rows;
    __strong id<MTLComputePipelineState> matvec_bf16_rows_gated_pair_x4;
    __strong id<MTLComputePipelineState> matvec_bf16_rows_x4;
    __strong id<MTLComputePipelineState> matvec_half_128;
    __strong id<MTLComputePipelineState> matvec_t_half_128;
    __strong id<MTLComputePipelineState> matvec_half_pair_128;
    __strong id<MTLComputePipelineState> matvec_half_triple_mixed_128;
    __strong id<MTLComputePipelineState> matvec_bf16_128;
    __strong id<MTLComputePipelineState> matvec_bf16_pair_128;
    __strong id<MTLComputePipelineState> matvec_bf16_triple_mixed_128;
    __strong id<MTLComputePipelineState> matvec_half_gated_pair_128;
    __strong id<MTLComputePipelineState> matvec_bf16_gated_pair_128;
    __strong id<MTLComputePipelineState> add;
    __strong id<MTLComputePipelineState> silu_ffn;
    __strong id<MTLComputePipelineState> gelu_ffn;
    __strong id<MTLComputePipelineState> rmsnorm;
    __strong id<MTLComputePipelineState> rmsnorm_rows;
    __strong id<MTLComputePipelineState> decode_attention;
} GollekMetalPipelines;

GollekMetalPipelines* gollek_metal_pipelines(void);
void gollek_metal_compile_runtime_pipelines(GollekMetalPipelines* pipelines);

#endif

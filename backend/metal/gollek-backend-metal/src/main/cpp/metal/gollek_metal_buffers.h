/**
 * gollek_metal_buffers.h — shared Metal buffer helpers.
 */

#ifndef GOLLEK_METAL_BUFFERS_H
#define GOLLEK_METAL_BUFFERS_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stddef.h>

id<MTLBuffer> gollek_metal_wrap_ptr(void* ptr, size_t bytes);
id<MTLBuffer> gollek_metal_wrap_weight_ptr(const void* ptr, size_t bytes);

BOOL gollek_metal_ensure_swiglu_scratch(size_t activation_bytes,
                                        id<MTLBuffer>* gate,
                                        id<MTLBuffer>* up,
                                        id<MTLBuffer>* combined);

#define wrap_ptr gollek_metal_wrap_ptr
#define wrap_weight_ptr gollek_metal_wrap_weight_ptr
#define ensure_swiglu_scratch gollek_metal_ensure_swiglu_scratch

#endif

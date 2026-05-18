/**
 * gollek_metal_buffers.m — shared Metal buffer helpers.
 */

#import "gollek_metal_buffers.h"
#import "gollek_metal_support.h"

static id<MTLBuffer> g_swiglu_gate_scratch = nil;
static id<MTLBuffer> g_swiglu_up_scratch = nil;
static id<MTLBuffer> g_swiglu_combined_scratch = nil;
static size_t g_swiglu_scratch_capacity = 0;
static NSMutableDictionary<NSString*, id<MTLBuffer>>* g_weight_buffer_cache = nil;

id<MTLBuffer> gollek_metal_wrap_ptr(void* ptr, size_t bytes) {
    return [g_device newBufferWithBytesNoCopy:ptr
                                       length:bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

id<MTLBuffer> gollek_metal_wrap_weight_ptr(const void* ptr, size_t bytes) {
    if (ptr == NULL || bytes == 0) return nil;
    if (!gollek_metal_env_truthy("GOLLEK_METAL_ENABLE_WEIGHT_BUFFER_CACHE")
            || gollek_metal_env_truthy("GOLLEK_METAL_DISABLE_WEIGHT_BUFFER_CACHE")) {
        return gollek_metal_wrap_ptr((void*)ptr, bytes);
    }
    NSString* key = [NSString stringWithFormat:@"%p:%zu", ptr, bytes];
    @synchronized([NSMutableDictionary class]) {
        id<MTLBuffer> cached = g_weight_buffer_cache != nil ? [g_weight_buffer_cache objectForKey:key] : nil;
        if (cached != nil) {
            return cached;
        }
    }

    id<MTLBuffer> wrapped = gollek_metal_wrap_ptr((void*)ptr, bytes);
    if (wrapped == nil) {
        return nil;
    }

    @synchronized([NSMutableDictionary class]) {
        if (g_weight_buffer_cache == nil) {
            g_weight_buffer_cache = [[NSMutableDictionary alloc] init];
        }
        int max_entries = gollek_metal_env_int_or_default("GOLLEK_METAL_WEIGHT_BUFFER_CACHE_MAX_ENTRIES", 4096);
        if (max_entries > 0 && [g_weight_buffer_cache count] >= (NSUInteger)max_entries) {
            [g_weight_buffer_cache removeAllObjects];
        }
        [g_weight_buffer_cache setObject:wrapped forKey:key];
    }
    return wrapped;
}

BOOL gollek_metal_ensure_swiglu_scratch(size_t activation_bytes,
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

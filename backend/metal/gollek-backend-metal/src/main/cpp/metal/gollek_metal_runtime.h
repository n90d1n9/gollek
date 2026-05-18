/**
 * gollek_metal_runtime.h — exported runtime/device helpers for the Metal dylib.
 */

#ifndef GOLLEK_METAL_RUNTIME_H
#define GOLLEK_METAL_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int gollek_metal_init(void);
long gollek_metal_available_memory(void);

int gollek_metal_set_mps_matvec_enabled(int enabled);
int gollek_metal_set_mps_matvec_autotune_enabled(int enabled);
int gollek_metal_set_mps_matvec_max_inner(int max_inner);
int gollek_metal_set_mps_matvec_max_output(int max_output);
int gollek_metal_set_mps_matvec_autotune_max_output(int max_output);

void* gollek_metal_alloc(size_t bytes, size_t align);

int gollek_metal_argmax_f32(const void* logits,
                            int n,
                            int reject0,
                            int reject1,
                            int reject2,
                            int reject3,
                            int reject4,
                            int reject5,
                            int reject6,
                            int reject7);

int gollek_metal_device_name(char* buf, int bufSz);
int gollek_metal_is_unified_memory(void);

#ifdef __cplusplus
}
#endif

#endif

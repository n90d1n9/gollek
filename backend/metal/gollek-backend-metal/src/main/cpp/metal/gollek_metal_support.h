/**
 * gollek_metal_support.h — shared state and low-level helpers for the Metal dylib.
 */

#ifndef GOLLEK_METAL_SUPPORT_H
#define GOLLEK_METAL_SUPPORT_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>

extern id<MTLDevice>       g_device;
extern id<MTLCommandQueue> g_queue;
extern BOOL                g_initialized;

BOOL gollek_metal_env_truthy(const char* name);
int gollek_metal_env_int_or_default(const char* name, int default_value);
float gollek_metal_env_float_or_default(const char* name, float default_value);
uint64_t gollek_metal_monotonic_nanos(void);

#endif

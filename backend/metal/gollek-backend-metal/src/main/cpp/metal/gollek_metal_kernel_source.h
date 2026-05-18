/**
 * gollek_metal_kernel_source.h — runtime Metal shader source builders.
 */

#ifndef GOLLEK_METAL_KERNEL_SOURCE_H
#define GOLLEK_METAL_KERNEL_SOURCE_H

#import <Foundation/Foundation.h>

NSString* gollek_metal_runtime_kernel_source(void);
NSString* gollek_metal_matvec_kernel_source(NSUInteger threads);

#endif

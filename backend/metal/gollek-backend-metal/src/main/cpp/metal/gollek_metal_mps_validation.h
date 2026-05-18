/**
 * gollek_metal_mps_validation.h — MPS matvec validation helpers.
 */

#ifndef GOLLEK_METAL_MPS_VALIDATION_H
#define GOLLEK_METAL_MPS_VALIDATION_H

#import <Foundation/Foundation.h>
#include <stdint.h>

NSString* gollek_metal_mps_matvec_shape_key(int K, int N);
float gollek_metal_bf16_to_f32(uint16_t bits);
uint16_t gollek_metal_f32_to_bf16_bits(float value);

BOOL gollek_metal_validate_mps_matvec_half_output(const float* C,
                                                  const float* A,
                                                  const uint16_t* B,
                                                  int K,
                                                  int N);

BOOL gollek_metal_validate_mps_matvec_bf16_output(const float* C,
                                                  const float* A,
                                                  const uint16_t* B,
                                                  int K,
                                                  int N);

#define mps_matvec_shape_key gollek_metal_mps_matvec_shape_key
#define bf16_to_f32_bridge gollek_metal_bf16_to_f32
#define f32_to_bf16_bits_bridge gollek_metal_f32_to_bf16_bits
#define validate_mps_matvec_half_output gollek_metal_validate_mps_matvec_half_output
#define validate_mps_matvec_bf16_output gollek_metal_validate_mps_matvec_bf16_output

#endif

/**
 * gollek_metal_matvec_api.h — exported Metal matvec C API.
 */

#ifndef GOLLEK_METAL_MATVEC_API_H
#define GOLLEK_METAL_MATVEC_API_H

#ifdef __cplusplus
extern "C" {
#endif

int gollek_metal_matvec_tb_half(void* C,
                                const void* A,
                                const void* B,
                                int K,
                                int N);

int gollek_metal_matvec_tb_half_mps(void* C,
                                    const void* A,
                                    const void* B,
                                    int K,
                                    int N);

int gollek_metal_matvec_t_half(void* C,
                               const void* A,
                               const void* B,
                               int K,
                               int N);

int gollek_metal_matvec_tb_bf16(void* C,
                                const void* A,
                                const void* B,
                                int K,
                                int N);

int gollek_metal_matvec_tb_half_pair(void* C0,
                                     void* C1,
                                     const void* A,
                                     const void* B0,
                                     const void* B1,
                                     int K,
                                     int N);

int gollek_metal_matvec_tb_bf16_pair(void* C0,
                                     void* C1,
                                     const void* A,
                                     const void* B0,
                                     const void* B1,
                                     int K,
                                     int N);

int gollek_metal_matvec_tb_half_triple_mixed(void* C0,
                                             void* C1,
                                             void* C2,
                                             const void* A,
                                             const void* B0,
                                             const void* B1,
                                             const void* B2,
                                             int K,
                                             int N0,
                                             int N1,
                                             int N2);

int gollek_metal_matvec_tb_bf16_triple_mixed(void* C0,
                                             void* C1,
                                             void* C2,
                                             const void* A,
                                             const void* B0,
                                             const void* B1,
                                             const void* B2,
                                             int K,
                                             int N0,
                                             int N1,
                                             int N2);

#ifdef __cplusplus
}
#endif

#endif

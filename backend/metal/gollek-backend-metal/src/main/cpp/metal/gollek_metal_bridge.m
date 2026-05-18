/**
 * gollek_metal_bridge.m — compatibility anchor for the Gollek Metal dylib.
 *
 * The exported runtime, buffer, matvec, FFN, attention, and FA4 entry points
 * now live in focused modules next to this file. Keep this translation unit so
 * older build scripts that still mention gollek_metal_bridge.m continue to
 * compile while the bridge stays intentionally small.
 */

#import "gollek_metal_matvec_api.h"

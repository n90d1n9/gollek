package tech.kayys.gollek.gguf.runtime;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;

/**
 * Quant-family operations for prepared GGUF matrices.
 *
 * <p>{@link GgufPreparedCachePolicy} decides whether a tensor belongs to a prepared-cache
 * family; this helper owns the concrete prepare and mat-vec calls for that
 * family so callers do not grow parallel per-quant switches.</p>
 */
final class GgufPrepOps {
    private GgufPrepOps() {
    }

    static boolean rows(
            GGUFModel model,
            GGUFTensorInfo tensor,
            GgufKey key,
            GgufPreparedCachePolicy.Family family,
            long maxBytes,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        var matrix = GgufPreparedMatrixStore.matrixForMatVec(model, tensor, key, family, maxBytes);
        if (matrix == null) {
            return false;
        }
        GgufPrepRows.rowsTrusted(matrix, family, columns, vector, output, rowCount, parallel);
        return true;
    }
}

package tech.kayys.gollek.gguf.runtime;

import tech.kayys.gollek.gguf.core.GgmlType;
import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFParser;
import tech.kayys.gollek.gguf.loader.GGUFReader;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Executes a tiny Java-native tensor primitive against a GGUF model.
 */
public record GgufRuntimeProbe(
        GgufRuntimeProfile profile,
        String tensorName,
        String tensorType,
        long rows,
        long columns,
        int sampledRows,
        long rowDotNanos,
        float rowDotChecksum,
        int matVecRows,
        long matrixCacheNanos,
        long matVecNanos,
        float matVecChecksum,
        long cachedMatVecNanos,
        float cachedMatVecChecksum
) {
    private static final List<String> PREFERRED_PROBE_TENSORS = List.of(
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "token_embd.weight"
    );

    public static GgufRuntimeProbe load(Path modelPath, int requestedRows) throws IOException {
        return load(modelPath, requestedRows, requestedRows);
    }

    public static GgufRuntimeProbe load(Path modelPath, int requestedDotRows, int requestedMatVecRows) throws IOException {
        long startNanos = System.nanoTime();
        try (Arena arena = Arena.ofShared(); GGUFReader reader = new GGUFReader(modelPath, arena)) {
            GGUFModel model = new GGUFParser().parse(reader.segment(), null);
            long loadMillis = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
            return fromModel(model, Files.size(modelPath), loadMillis, requestedDotRows, requestedMatVecRows);
        }
    }

    public static GgufRuntimeProbe fromModel(
            GGUFModel model,
            long modelBytes,
            long loadMillis,
            int requestedRows) {
        return fromModel(model, modelBytes, loadMillis, requestedRows, requestedRows);
    }

    public static GgufRuntimeProbe fromModel(
            GGUFModel model,
            long modelBytes,
            long loadMillis,
            int requestedDotRows,
            int requestedMatVecRows) {
        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model, modelBytes, loadMillis);
        Optional<GGUFTensorInfo> tensor = selectProbeTensor(model);
        if (tensor.isEmpty()) {
            return new GgufRuntimeProbe(profile, "", "", 0, 0, 0, 0, 0.0f, 0, 0, 0, 0.0f, 0, 0.0f);
        }
        return probeTensor(model, tensor.get(), requestedDotRows, requestedMatVecRows, profile);
    }

    public boolean hasTensorProbe() {
        return tensorName != null && !tensorName.isBlank();
    }

    public double rowDotMillis() {
        return rowDotNanos / 1_000_000.0d;
    }

    public double matVecMillis() {
        return matVecNanos / 1_000_000.0d;
    }

    public double matrixCacheMillis() {
        return matrixCacheNanos / 1_000_000.0d;
    }

    public double cachedMatVecMillis() {
        return cachedMatVecNanos / 1_000_000.0d;
    }

    private static Optional<GGUFTensorInfo> selectProbeTensor(GGUFModel model) {
        for (String name : PREFERRED_PROBE_TENSORS) {
            Optional<GGUFTensorInfo> tensor = model.tensors().stream()
                    .filter(candidate -> name.equals(candidate.name()))
                    .filter(GgufRuntimeProbe::canProbe)
                    .findFirst();
            if (tensor.isPresent()) {
                return tensor;
            }
        }
        return model.tensors().stream()
                .filter(GgufRuntimeProbe::canProbe)
                .max(Comparator.comparingLong(GGUFTensorInfo::sizeInBytes));
    }

    private static boolean canProbe(GGUFTensorInfo tensor) {
        return tensor.shape().length >= 2
                && GgufTensorOps.supportsRowDotType(tensor.typeId())
                && GgufTensorOps.matrixColumns(tensor) <= Integer.MAX_VALUE;
    }

    private static GgufRuntimeProbe probeTensor(
            GGUFModel model,
            GGUFTensorInfo tensor,
            int requestedDotRows,
            int requestedMatVecRows,
            GgufRuntimeProfile profile) {
        long rows = GgufTensorOps.matrixRows(tensor);
        long columns = GgufTensorOps.matrixColumns(tensor);
        int sampledRows = (int) Math.max(1, Math.min(rows, Math.max(1, requestedDotRows)));
        int matVecRows = (int) Math.max(1, Math.min(rows, Math.max(1, requestedMatVecRows)));
        float[] vector = deterministicProbeVector((int) columns);

        float rowDotChecksum = 0.0f;
        long startNanos = System.nanoTime();
        for (int row = 0; row < sampledRows; row++) {
            rowDotChecksum += GgufTensorOps.dotRow(model, tensor, row, vector);
        }
        long rowDotNanos = System.nanoTime() - startNanos;

        float[] output = new float[matVecRows];
        long matrixCacheNanos = 0L;
        long matVecNanos;
        long cachedMatVecNanos = 0L;
        float cachedMatVecChecksum = 0.0f;
        if (GgufTensorOps.supportsQ32PreparedType(tensor.typeId())) {
            GgufTensorOps.clearQ32MatrixCache(model);
            long cacheStartNanos = System.nanoTime();
            GgufTensorOps.Q32Matrix matrix = GgufTensorOps.q32MatrixCached(model, tensor);
            matrixCacheNanos = System.nanoTime() - cacheStartNanos;
            startNanos = System.nanoTime();
            GgufTensorOps.matVecRows(matrix, vector, output, matVecRows, true);
            matVecNanos = System.nanoTime() - startNanos;

            float[] cachedOutput = new float[matVecRows];
            long cachedStartNanos = System.nanoTime();
            GgufTensorOps.matVecRows(model, tensor, vector, cachedOutput, matVecRows, true);
            cachedMatVecNanos = System.nanoTime() - cachedStartNanos;
            cachedMatVecChecksum = checksum(cachedOutput);
        } else if (tensor.typeId() == GgmlType.Q4_K.id) {
            GgufTensorOps.clearQ4KMatrixCache(model);
            long cacheStartNanos = System.nanoTime();
            GgufTensorOps.Q4KMatrix matrix = GgufTensorOps.q4KMatrixCached(model, tensor);
            matrixCacheNanos = System.nanoTime() - cacheStartNanos;
            startNanos = System.nanoTime();
            GgufTensorOps.matVecRows(matrix, vector, output, matVecRows, true);
            matVecNanos = System.nanoTime() - startNanos;

            float[] cachedOutput = new float[matVecRows];
            long cachedStartNanos = System.nanoTime();
            GgufTensorOps.matVecRows(model, tensor, vector, cachedOutput, matVecRows, true);
            cachedMatVecNanos = System.nanoTime() - cachedStartNanos;
            cachedMatVecChecksum = checksum(cachedOutput);
        } else if (tensor.typeId() == GgmlType.Q5_K.id) {
            GgufTensorOps.clearQ5KMatrixCache(model);
            long cacheStartNanos = System.nanoTime();
            GgufTensorOps.Q5KMatrix matrix = GgufTensorOps.q5KMatrixCached(model, tensor);
            matrixCacheNanos = System.nanoTime() - cacheStartNanos;
            startNanos = System.nanoTime();
            GgufTensorOps.matVecRows(matrix, vector, output, matVecRows, true);
            matVecNanos = System.nanoTime() - startNanos;

            float[] cachedOutput = new float[matVecRows];
            long cachedStartNanos = System.nanoTime();
            GgufTensorOps.matVecRows(model, tensor, vector, cachedOutput, matVecRows, true);
            cachedMatVecNanos = System.nanoTime() - cachedStartNanos;
            cachedMatVecChecksum = checksum(cachedOutput);
        } else if (tensor.typeId() == GgmlType.Q6_K.id) {
            GgufTensorOps.clearQ6KMatrixCache(model);
            long cacheStartNanos = System.nanoTime();
            GgufTensorOps.Q6KMatrix matrix = GgufTensorOps.q6KMatrixCached(model, tensor);
            matrixCacheNanos = System.nanoTime() - cacheStartNanos;
            startNanos = System.nanoTime();
            GgufTensorOps.matVecRows(matrix, vector, output, matVecRows, true);
            matVecNanos = System.nanoTime() - startNanos;

            float[] cachedOutput = new float[matVecRows];
            long cachedStartNanos = System.nanoTime();
            GgufTensorOps.matVecRows(model, tensor, vector, cachedOutput, matVecRows, true);
            cachedMatVecNanos = System.nanoTime() - cachedStartNanos;
            cachedMatVecChecksum = checksum(cachedOutput);
        } else if (tensor.typeId() == GgmlType.Q8_0.id) {
            GgufTensorOps.clearQ8MatrixCache(model);
            long cacheStartNanos = System.nanoTime();
            GgufTensorOps.Q8Matrix matrix = GgufTensorOps.q8MatrixCached(model, tensor);
            matrixCacheNanos = System.nanoTime() - cacheStartNanos;
            startNanos = System.nanoTime();
            GgufTensorOps.matVecRows(matrix, vector, output, matVecRows, true);
            matVecNanos = System.nanoTime() - startNanos;

            float[] cachedOutput = new float[matVecRows];
            long cachedStartNanos = System.nanoTime();
            GgufTensorOps.matVecRows(model, tensor, vector, cachedOutput, matVecRows, true);
            cachedMatVecNanos = System.nanoTime() - cachedStartNanos;
            cachedMatVecChecksum = checksum(cachedOutput);
        } else {
            startNanos = System.nanoTime();
            GgufTensorOps.matVecRows(model, tensor, vector, output, matVecRows, true);
            matVecNanos = System.nanoTime() - startNanos;
        }
        float matVecChecksum = checksum(output);

        return new GgufRuntimeProbe(
                profile,
                tensor.name(),
                typeLabel(tensor.typeId()),
                rows,
                columns,
                sampledRows,
                rowDotNanos,
                rowDotChecksum,
                matVecRows,
                matrixCacheNanos,
                matVecNanos,
                matVecChecksum,
                cachedMatVecNanos,
                cachedMatVecChecksum);
    }

    private static float[] deterministicProbeVector(int columns) {
        float[] vector = new float[columns];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = ((i % 17) - 8) / 17.0f;
        }
        return vector;
    }

    private static String typeLabel(int typeId) {
        try {
            return GgmlType.fromId(typeId).label;
        } catch (IllegalArgumentException ignored) {
            return "TYPE_" + typeId;
        }
    }

    private static float checksum(float[] values) {
        float total = 0.0f;
        for (float value : values) {
            total += value;
        }
        return total;
    }
}

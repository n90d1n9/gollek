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
import java.util.Locale;
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
        boolean preparedMatVecProbe,
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
            return new GgufRuntimeProbe(profile, "", "", 0, 0, 0, 0, 0.0f, 0, false, 0, 0, 0.0f, 0, 0.0f);
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

    public boolean preparedMatVecReady() {
        return preparedMatVecProbe && matVecChecksumsAgree();
    }

    public boolean matVecChecksumsAgree() {
        float tolerance = Math.max(1.0e-4f, Math.abs(matVecChecksum) * 1.0e-4f);
        return Float.isFinite(matVecChecksum)
                && Float.isFinite(cachedMatVecChecksum)
                && Math.abs(matVecChecksum - cachedMatVecChecksum) <= tolerance;
    }

    public String compactSummary() {
        if (!hasTensorProbe()) {
            return "unavailable, no supported matrix tensor";
        }
        return String.format(
                Locale.ROOT,
                "tensor=%s, type=%s, rows=%d, cols=%d, sampledRows=%d, "
                        + "dot=%.3fms, dotChecksum=%.6g, matVecRows=%d, cache=%.3fms, "
                        + "preparedMatVecReady=%s, parallelMatVec=%.3fms, matVecChecksum=%.6g, "
                        + "cachedGenericMatVec=%.3fms, cachedChecksum=%.6g",
                tensorName,
                tensorType,
                rows,
                columns,
                sampledRows,
                rowDotMillis(),
                rowDotChecksum,
                matVecRows,
                matrixCacheMillis(),
                preparedMatVecReady(),
                matVecMillis(),
                matVecChecksum,
                cachedMatVecMillis(),
                cachedMatVecChecksum);
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
        boolean preparedMatVecProbe = false;
        long cachedMatVecNanos = 0L;
        float cachedMatVecChecksum = 0.0f;
        PreparedMatVecResult preparedMatVec = preparedMatVecProbe(model, tensor, vector, matVecRows);
        if (preparedMatVec != null) {
            preparedMatVecProbe = true;
            matrixCacheNanos = preparedMatVec.matrixCacheNanos();
            matVecNanos = preparedMatVec.matVecNanos();
            output = preparedMatVec.output();
            cachedMatVecNanos = preparedMatVec.cachedMatVecNanos();
            cachedMatVecChecksum = preparedMatVec.cachedMatVecChecksum();
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
                preparedMatVecProbe,
                matrixCacheNanos,
                matVecNanos,
                matVecChecksum,
                cachedMatVecNanos,
                cachedMatVecChecksum);
    }

    private static PreparedMatVecResult preparedMatVecProbe(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            int matVecRows) {
        int typeId = tensor.typeId();
        if (GgufTensorOps.supportsQ32PreparedType(typeId)) {
            return runPreparedMatVecProbe(
                    model,
                    tensor,
                    vector,
                    matVecRows,
                    cacheModel -> GgufTensorOps.clearQ32MatrixCache(cacheModel),
                    GgufTensorOps::q32MatrixCached,
                    (matrix, probeVector, probeOutput, rows, parallel) ->
                            GgufTensorOps.matVecRows(matrix, probeVector, probeOutput, rows, parallel));
        }
        if (typeId == GgmlType.Q2_K.id) {
            return runPreparedMatVecProbe(
                    model,
                    tensor,
                    vector,
                    matVecRows,
                    cacheModel -> GgufTensorOps.clearQ2KMatrixCache(cacheModel),
                    GgufTensorOps::q2KMatrixCached,
                    (matrix, probeVector, probeOutput, rows, parallel) ->
                            GgufTensorOps.matVecRows(matrix, probeVector, probeOutput, rows, parallel));
        }
        if (typeId == GgmlType.Q3_K.id) {
            return runPreparedMatVecProbe(
                    model,
                    tensor,
                    vector,
                    matVecRows,
                    cacheModel -> GgufTensorOps.clearQ3KMatrixCache(cacheModel),
                    GgufTensorOps::q3KMatrixCached,
                    (matrix, probeVector, probeOutput, rows, parallel) ->
                            GgufTensorOps.matVecRows(matrix, probeVector, probeOutput, rows, parallel));
        }
        if (typeId == GgmlType.Q4_K.id) {
            return runPreparedMatVecProbe(
                    model,
                    tensor,
                    vector,
                    matVecRows,
                    cacheModel -> GgufTensorOps.clearQ4KMatrixCache(cacheModel),
                    GgufTensorOps::q4KMatrixCached,
                    (matrix, probeVector, probeOutput, rows, parallel) ->
                            GgufTensorOps.matVecRows(matrix, probeVector, probeOutput, rows, parallel));
        }
        if (typeId == GgmlType.Q5_K.id) {
            return runPreparedMatVecProbe(
                    model,
                    tensor,
                    vector,
                    matVecRows,
                    cacheModel -> GgufTensorOps.clearQ5KMatrixCache(cacheModel),
                    GgufTensorOps::q5KMatrixCached,
                    (matrix, probeVector, probeOutput, rows, parallel) ->
                            GgufTensorOps.matVecRows(matrix, probeVector, probeOutput, rows, parallel));
        }
        if (typeId == GgmlType.Q6_K.id) {
            return runPreparedMatVecProbe(
                    model,
                    tensor,
                    vector,
                    matVecRows,
                    cacheModel -> GgufTensorOps.clearQ6KMatrixCache(cacheModel),
                    GgufTensorOps::q6KMatrixCached,
                    (matrix, probeVector, probeOutput, rows, parallel) ->
                            GgufTensorOps.matVecRows(matrix, probeVector, probeOutput, rows, parallel));
        }
        if (typeId == GgmlType.Q8_0.id || typeId == GgmlType.Q8_1.id || typeId == GgmlType.Q8_K.id
                || typeId == GgmlType.IQ4_NL.id || typeId == GgmlType.IQ4_XS.id) {
            return runPreparedMatVecProbe(
                    model,
                    tensor,
                    vector,
                    matVecRows,
                    cacheModel -> GgufTensorOps.clearQ8MatrixCache(cacheModel),
                    GgufTensorOps::q8MatrixCached,
                    (matrix, probeVector, probeOutput, rows, parallel) ->
                            GgufTensorOps.matVecRows(matrix, probeVector, probeOutput, rows, parallel));
        }
        return null;
    }

    private static <M> PreparedMatVecResult runPreparedMatVecProbe(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            int matVecRows,
            MatrixCacheClearer cacheClearer,
            PreparedMatrixLoader<M> matrixLoader,
            PreparedMatVecRunner<M> matVecRunner) {
        cacheClearer.clear(model);
        long cacheStartNanos = System.nanoTime();
        M matrix = matrixLoader.load(model, tensor);
        long matrixCacheNanos = System.nanoTime() - cacheStartNanos;

        float[] output = new float[matVecRows];
        long startNanos = System.nanoTime();
        matVecRunner.run(matrix, vector, output, matVecRows, true);
        long matVecNanos = System.nanoTime() - startNanos;

        float[] cachedOutput = new float[matVecRows];
        long cachedStartNanos = System.nanoTime();
        GgufTensorOps.matVecRows(model, tensor, vector, cachedOutput, matVecRows, true);
        long cachedMatVecNanos = System.nanoTime() - cachedStartNanos;

        return new PreparedMatVecResult(
                matrixCacheNanos,
                matVecNanos,
                output,
                cachedMatVecNanos,
                checksum(cachedOutput));
    }

    private record PreparedMatVecResult(
            long matrixCacheNanos,
            long matVecNanos,
            float[] output,
            long cachedMatVecNanos,
            float cachedMatVecChecksum) {
    }

    @FunctionalInterface
    private interface MatrixCacheClearer {
        void clear(GGUFModel model);
    }

    @FunctionalInterface
    private interface PreparedMatrixLoader<M> {
        M load(GGUFModel model, GGUFTensorInfo tensor);
    }

    @FunctionalInterface
    private interface PreparedMatVecRunner<M> {
        void run(M matrix, float[] vector, float[] output, int rowCount, boolean parallel);
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

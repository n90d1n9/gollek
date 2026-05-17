package tech.kayys.gollek.gguf.runtime;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.gollek.gguf.core.GgmlType;
import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;
import java.util.stream.IntStream;

/**
 * Small active GGUF tensor primitive surface for Java-native generation work.
 */
public final class GgufTensorOps {
    private static final int Q4_0_BLOCK_SIZE = 32;
    private static final int Q4_0_BLOCK_BYTES = 18;
    private static final int Q4_1_BLOCK_BYTES = 20;
    private static final int Q5_0_BLOCK_BYTES = 22;
    private static final int Q5_1_BLOCK_BYTES = 24;
    private static final int Q8_0_BLOCK_SIZE = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;
    private static final int QK_K = 256;
    private static final int Q4_K_BLOCK_BYTES = 144;
    private static final int Q5_K_BLOCK_BYTES = 176;
    private static final int Q6_K_BLOCK_BYTES = 210;
    private static final VectorSpecies<Byte> Q4_DOT_BYTE_SPECIES = ByteVector.SPECIES_64;
    private static final VectorSpecies<Float> Q4_DOT_FLOAT_SPECIES = FloatVector.SPECIES_256;
    private static final int Q4_DOT_VECTOR_LANES = Q4_DOT_BYTE_SPECIES.length();
    private static final boolean Q4_DOT_VECTOR_ENABLED =
            Boolean.parseBoolean(System.getProperty(
                    "gollek.gguf.quant.vector_dot",
                    System.getProperty("gollek.gguf.q4k.vector_dot", "false")));
    private static final boolean SIGNED_BYTE_DOT_VECTOR_ENABLED =
            Boolean.parseBoolean(System.getProperty("gollek.gguf.signed_byte.vector_dot", "true"));
    private static final long DEFAULT_Q4K_CACHE_MAX_BYTES = 512L * 1024L * 1024L;
    private static final long DEFAULT_PARALLEL_MIN_OPS = 131_072L;
    private static final Map<GGUFModel, Q32ModelCache> Q32_MATRIX_CACHE = new WeakHashMap<>();
    private static final Map<GGUFModel, Q4KModelCache> Q4K_MATRIX_CACHE = new WeakHashMap<>();
    private static final Map<GGUFModel, Q5KModelCache> Q5K_MATRIX_CACHE = new WeakHashMap<>();
    private static final Map<GGUFModel, Q6KModelCache> Q6K_MATRIX_CACHE = new WeakHashMap<>();
    private static final Map<GGUFModel, Q8ModelCache> Q8_MATRIX_CACHE = new WeakHashMap<>();
    private static final ThreadLocal<Q4KWorkBuffer> Q4K_WORK_BUFFER =
            ThreadLocal.withInitial(Q4KWorkBuffer::new);

    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufTensorOps() {
    }

    public record Q4KMatrix(
            int columns,
            int rows,
            int blocksPerRow,
            /**
             * Unpacked Q4 values, one unsigned nibble per byte, grouped in the same 32-value scale groups used by Q4_K.
             */
            byte[] quants,
            float[] groupScales,
            float[] groupMins) {
        public long estimatedBytes() {
            return (long) quants.length
                    + (long) groupScales.length * Float.BYTES
                    + (long) groupMins.length * Float.BYTES;
        }
    }

    public record Q32Matrix(
            int columns,
            int rows,
            int blocksPerRow,
            /**
             * Unpacked 32-value block quants. Symmetric formats store signed values, affine formats store unsigned values.
             */
            byte[] quants,
            float[] blockScales,
            float[] blockBiases) {
        public long estimatedBytes() {
            return (long) quants.length
                    + (long) blockScales.length * Float.BYTES
                    + (long) blockBiases.length * Float.BYTES;
        }
    }

    public record Q5KMatrix(
            int columns,
            int rows,
            int blocksPerRow,
            /**
             * Unpacked Q5 values, one unsigned 5-bit value per byte, grouped in Q5_K's 32-value scale groups.
             */
            byte[] quants,
            float[] groupScales,
            float[] groupMins) {
        public long estimatedBytes() {
            return (long) quants.length
                    + (long) groupScales.length * Float.BYTES
                    + (long) groupMins.length * Float.BYTES;
        }
    }

    public record Q6KMatrix(
            int columns,
            int rows,
            int blocksPerRow,
            /**
             * Unpacked signed Q6 values, one value per byte, grouped in Q6_K's 16-value scale groups.
             */
            byte[] quants,
            float[] groupScales) {
        public long estimatedBytes() {
            return (long) quants.length + (long) groupScales.length * Float.BYTES;
        }
    }

    public record Q8Matrix(
            int columns,
            int rows,
            int blocksPerRow,
            byte[] quants,
            float[] blockScales) {
        public long estimatedBytes() {
            return (long) quants.length + (long) blockScales.length * Float.BYTES;
        }
    }

    public static final class Q4KWorkBuffer {
        private float[] vectorGroupSums = new float[0];

        public int vectorGroupSumCapacity() {
            return vectorGroupSums.length;
        }

        public void clear() {
            vectorGroupSums = new float[0];
        }

        private float[] vectorGroupSums(int requiredLength) {
            if (vectorGroupSums.length < requiredLength) {
                vectorGroupSums = new float[requiredLength];
            }
            return vectorGroupSums;
        }
    }

    public static GGUFTensorInfo findTensor(GGUFModel model, String name) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(name, "name");
        return model.tensors().stream()
                .filter(tensor -> name.equals(tensor.name()))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Tensor not found: " + name));
    }

    public static MemorySegment tensorData(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        long start = model.dataStart() + tensor.offset();
        long end = start + tensor.sizeInBytes();
        if (start < 0 || end < start || end > model.segment().byteSize()) {
            throw new IllegalArgumentException("Tensor data range is outside model segment: " + tensor.name());
        }
        return model.segment().asSlice(start, tensor.sizeInBytes());
    }

    public static long elementCount(GGUFTensorInfo tensor) {
        long elements = 1;
        for (long dimension : tensor.shape()) {
            elements = Math.multiplyExact(elements, dimension);
        }
        return elements;
    }

    /**
     * GGUF stores matrix columns in shape[0]; the remaining dimensions form rows.
     */
    public static long matrixColumns(GGUFTensorInfo tensor) {
        if (tensor.shape().length == 0) {
            throw new IllegalArgumentException("Tensor has no shape: " + tensor.name());
        }
        return tensor.shape()[0];
    }

    public static long matrixRows(GGUFTensorInfo tensor) {
        if (tensor.shape().length <= 1) {
            return 1;
        }
        long rows = 1;
        for (int i = 1; i < tensor.shape().length; i++) {
            rows = Math.multiplyExact(rows, tensor.shape()[i]);
        }
        return rows;
    }

    public static boolean supportsRowDotType(int typeId) {
        return typeId == GgmlType.F32.id
                || typeId == GgmlType.F16.id
                || typeId == GgmlType.BF16.id
                || typeId == GgmlType.Q4_0.id
                || typeId == GgmlType.Q4_1.id
                || typeId == GgmlType.Q5_0.id
                || typeId == GgmlType.Q5_1.id
                || typeId == GgmlType.Q8_0.id
                || typeId == GgmlType.Q4_K.id
                || typeId == GgmlType.Q5_K.id
                || typeId == GgmlType.Q6_K.id;
    }

    public static void dequantizeRow(GGUFModel model, GGUFTensorInfo tensor, long row, float[] dst) {
        int columns = checkedColumns(tensor, dst.length);
        checkRow(tensor, row);
        MemorySegment data = tensorData(model, tensor);
        long rowOffset = row * rowByteSize(tensor, columns);
        dequantizeRow(data, rowOffset, tensor.typeId(), columns, dst, 0);
    }

    public static float dotRow(GGUFModel model, GGUFTensorInfo tensor, long row, float[] vector) {
        int columns = checkedColumns(tensor, vector.length);
        checkRow(tensor, row);
        MemorySegment data = tensorData(model, tensor);
        long rowOffset = row * rowByteSize(tensor, columns);
        return dotRow(data, rowOffset, tensor.typeId(), columns, vector, 0);
    }

    public static void matVec(GGUFModel model, GGUFTensorInfo tensor, float[] vector, float[] output) {
        matVecRows(model, tensor, vector, output, checkedRows(tensor), false);
    }

    public static void matVecParallel(GGUFModel model, GGUFTensorInfo tensor, float[] vector, float[] output) {
        matVecRows(model, tensor, vector, output, checkedRows(tensor), true);
    }

    public static void matVecRows(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(output, "output");
        long rows = matrixRows(tensor);
        if (rowCount < 0 || rowCount > rows) {
            throw new IllegalArgumentException("Requested row count " + rowCount + " is outside tensor rows " + rows);
        }
        if (output.length < rowCount) {
            throw new IllegalArgumentException(
                    "Output length " + output.length + " is smaller than requested rows " + rowCount);
        }
        int columns = checkedColumns(tensor, vector.length);
        int typeId = tensor.typeId();

        if (supportsQ32PreparedType(typeId) && rowCount >= q32CacheMinRows()) {
            matVecRows(q32MatrixCached(model, tensor), vector, output, rowCount, parallel);
            return;
        }
        if (typeId == GgmlType.Q4_K.id) {
            if (rowCount >= q4KCacheMinRows()) {
                matVecRows(q4KMatrixCached(model, tensor), vector, output, rowCount, parallel);
                return;
            }
            MemorySegment data = tensorData(model, tensor);
            long rowBytes = rowByteSize(tensor, columns);
            matVecRowsQ4K(data, columns, rowBytes, vector, output, rowCount, parallel);
            return;
        }
        if (typeId == GgmlType.Q5_K.id && rowCount >= q5KCacheMinRows()) {
            matVecRows(q5KMatrixCached(model, tensor), vector, output, rowCount, parallel);
            return;
        }
        if (typeId == GgmlType.Q6_K.id && rowCount >= q6KCacheMinRows()) {
            matVecRows(q6KMatrixCached(model, tensor), vector, output, rowCount, parallel);
            return;
        }
        if (typeId == GgmlType.Q8_0.id && rowCount >= q8CacheMinRows()) {
            matVecRows(q8MatrixCached(model, tensor), vector, output, rowCount, parallel);
            return;
        }

        MemorySegment data = tensorData(model, tensor);
        long rowBytes = rowByteSize(tensor, columns);
        if (shouldParallelize(parallel, rowCount, columns)) {
            try {
                int chunks = parallelChunkCount(rowCount);
                IntStream.range(0, chunks)
                        .parallel()
                        .forEach(chunk -> {
                            int start = (int) ((long) chunk * rowCount / chunks);
                            int end = (int) ((long) (chunk + 1) * rowCount / chunks);
                            fillMatVecRows(data, typeId, columns, rowBytes, vector, output, start, end);
                        });
                return;
            } catch (WrongThreadException ignored) {
                // Confined FFM segments cannot be read by worker threads; keep correctness and fall back.
            }
        }
        fillMatVecRows(data, typeId, columns, rowBytes, vector, output, 0, rowCount);
    }

    private static int q32CacheMinRows() {
        return Math.max(1, Integer.getInteger("gollek.gguf.q32.cache_min_rows", 32));
    }

    private static int q4KCacheMinRows() {
        return Math.max(1, Integer.getInteger("gollek.gguf.q4k.cache_min_rows", 32));
    }

    private static int q5KCacheMinRows() {
        return Math.max(1, Integer.getInteger("gollek.gguf.q5k.cache_min_rows", 32));
    }

    private static int q6KCacheMinRows() {
        return Math.max(1, Integer.getInteger("gollek.gguf.q6k.cache_min_rows", 32));
    }

    private static int q8CacheMinRows() {
        return Math.max(1, Integer.getInteger("gollek.gguf.q8.cache_min_rows", 32));
    }

    public static Q32Matrix q32MatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        long maxBytes = q32CacheMaxBytes();
        if (maxBytes <= 0) {
            synchronized (Q32_MATRIX_CACHE) {
                Q32_MATRIX_CACHE.remove(model);
            }
            return q32Matrix(model, tensor);
        }
        Q32MatrixKey key = q32MatrixKey(tensor);
        synchronized (Q32_MATRIX_CACHE) {
            Q32ModelCache modelCache = Q32_MATRIX_CACHE.get(model);
            if (modelCache != null) {
                modelCache.evictTo(maxBytes);
                Q32Matrix cached = modelCache.get(key);
                if (cached != null) {
                    return cached;
                }
            }
        }

        Q32Matrix prepared = q32Matrix(model, tensor);
        if (prepared.estimatedBytes() > maxBytes) {
            return prepared;
        }

        synchronized (Q32_MATRIX_CACHE) {
            Q32ModelCache modelCache = Q32_MATRIX_CACHE.computeIfAbsent(model, ignored -> new Q32ModelCache());
            Q32Matrix cached = modelCache.get(key);
            if (cached != null) {
                return cached;
            }
            modelCache.put(key, prepared, maxBytes);
            return prepared;
        }
    }

    public static Q4KMatrix q4KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        long maxBytes = q4KCacheMaxBytes();
        if (maxBytes <= 0) {
            synchronized (Q4K_MATRIX_CACHE) {
                Q4K_MATRIX_CACHE.remove(model);
            }
            return q4KMatrix(model, tensor);
        }
        Q4KMatrixKey key = q4KMatrixKey(tensor);
        synchronized (Q4K_MATRIX_CACHE) {
            Q4KModelCache modelCache = Q4K_MATRIX_CACHE.get(model);
            if (modelCache != null) {
                modelCache.evictTo(maxBytes);
                Q4KMatrix cached = modelCache.get(key);
                if (cached != null) {
                    return cached;
                }
            }
        }

        Q4KMatrix prepared = q4KMatrix(model, tensor);
        if (prepared.estimatedBytes() > maxBytes) {
            return prepared;
        }

        synchronized (Q4K_MATRIX_CACHE) {
            Q4KModelCache modelCache = Q4K_MATRIX_CACHE.computeIfAbsent(model, ignored -> new Q4KModelCache());
            Q4KMatrix cached = modelCache.get(key);
            if (cached != null) {
                return cached;
            }
            modelCache.put(key, prepared, maxBytes);
            return prepared;
        }
    }

    public static Q5KMatrix q5KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        long maxBytes = q5KCacheMaxBytes();
        if (maxBytes <= 0) {
            synchronized (Q5K_MATRIX_CACHE) {
                Q5K_MATRIX_CACHE.remove(model);
            }
            return q5KMatrix(model, tensor);
        }
        Q5KMatrixKey key = q5KMatrixKey(tensor);
        synchronized (Q5K_MATRIX_CACHE) {
            Q5KModelCache modelCache = Q5K_MATRIX_CACHE.get(model);
            if (modelCache != null) {
                modelCache.evictTo(maxBytes);
                Q5KMatrix cached = modelCache.get(key);
                if (cached != null) {
                    return cached;
                }
            }
        }

        Q5KMatrix prepared = q5KMatrix(model, tensor);
        if (prepared.estimatedBytes() > maxBytes) {
            return prepared;
        }

        synchronized (Q5K_MATRIX_CACHE) {
            Q5KModelCache modelCache = Q5K_MATRIX_CACHE.computeIfAbsent(model, ignored -> new Q5KModelCache());
            Q5KMatrix cached = modelCache.get(key);
            if (cached != null) {
                return cached;
            }
            modelCache.put(key, prepared, maxBytes);
            return prepared;
        }
    }

    public static Q6KMatrix q6KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        long maxBytes = q6KCacheMaxBytes();
        if (maxBytes <= 0) {
            synchronized (Q6K_MATRIX_CACHE) {
                Q6K_MATRIX_CACHE.remove(model);
            }
            return q6KMatrix(model, tensor);
        }
        Q6KMatrixKey key = q6KMatrixKey(tensor);
        synchronized (Q6K_MATRIX_CACHE) {
            Q6KModelCache modelCache = Q6K_MATRIX_CACHE.get(model);
            if (modelCache != null) {
                modelCache.evictTo(maxBytes);
                Q6KMatrix cached = modelCache.get(key);
                if (cached != null) {
                    return cached;
                }
            }
        }

        Q6KMatrix prepared = q6KMatrix(model, tensor);
        if (prepared.estimatedBytes() > maxBytes) {
            return prepared;
        }

        synchronized (Q6K_MATRIX_CACHE) {
            Q6KModelCache modelCache = Q6K_MATRIX_CACHE.computeIfAbsent(model, ignored -> new Q6KModelCache());
            Q6KMatrix cached = modelCache.get(key);
            if (cached != null) {
                return cached;
            }
            modelCache.put(key, prepared, maxBytes);
            return prepared;
        }
    }

    public static Q8Matrix q8MatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        long maxBytes = q8CacheMaxBytes();
        if (maxBytes <= 0) {
            synchronized (Q8_MATRIX_CACHE) {
                Q8_MATRIX_CACHE.remove(model);
            }
            return q8Matrix(model, tensor);
        }
        Q8MatrixKey key = q8MatrixKey(tensor);
        synchronized (Q8_MATRIX_CACHE) {
            Q8ModelCache modelCache = Q8_MATRIX_CACHE.get(model);
            if (modelCache != null) {
                modelCache.evictTo(maxBytes);
                Q8Matrix cached = modelCache.get(key);
                if (cached != null) {
                    return cached;
                }
            }
        }

        Q8Matrix prepared = q8Matrix(model, tensor);
        if (prepared.estimatedBytes() > maxBytes) {
            return prepared;
        }

        synchronized (Q8_MATRIX_CACHE) {
            Q8ModelCache modelCache = Q8_MATRIX_CACHE.computeIfAbsent(model, ignored -> new Q8ModelCache());
            Q8Matrix cached = modelCache.get(key);
            if (cached != null) {
                return cached;
            }
            modelCache.put(key, prepared, maxBytes);
            return prepared;
        }
    }

    public static int clearQ4KMatrixCache(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q4K_MATRIX_CACHE) {
            Q4KModelCache removed = Q4K_MATRIX_CACHE.remove(model);
            return removed == null ? 0 : removed.size();
        }
    }

    public static int clearQ32MatrixCache(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q32_MATRIX_CACHE) {
            Q32ModelCache removed = Q32_MATRIX_CACHE.remove(model);
            return removed == null ? 0 : removed.size();
        }
    }

    public static int clearQ5KMatrixCache(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q5K_MATRIX_CACHE) {
            Q5KModelCache removed = Q5K_MATRIX_CACHE.remove(model);
            return removed == null ? 0 : removed.size();
        }
    }

    public static int clearQ6KMatrixCache(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q6K_MATRIX_CACHE) {
            Q6KModelCache removed = Q6K_MATRIX_CACHE.remove(model);
            return removed == null ? 0 : removed.size();
        }
    }

    public static int clearQ8MatrixCache(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q8_MATRIX_CACHE) {
            Q8ModelCache removed = Q8_MATRIX_CACHE.remove(model);
            return removed == null ? 0 : removed.size();
        }
    }

    public static int q4KMatrixCacheSize(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q4K_MATRIX_CACHE) {
            Q4KModelCache modelCache = Q4K_MATRIX_CACHE.get(model);
            return modelCache == null ? 0 : modelCache.size();
        }
    }

    public static int q32MatrixCacheSize(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q32_MATRIX_CACHE) {
            Q32ModelCache modelCache = Q32_MATRIX_CACHE.get(model);
            return modelCache == null ? 0 : modelCache.size();
        }
    }

    public static int q5KMatrixCacheSize(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q5K_MATRIX_CACHE) {
            Q5KModelCache modelCache = Q5K_MATRIX_CACHE.get(model);
            return modelCache == null ? 0 : modelCache.size();
        }
    }

    public static int q6KMatrixCacheSize(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q6K_MATRIX_CACHE) {
            Q6KModelCache modelCache = Q6K_MATRIX_CACHE.get(model);
            return modelCache == null ? 0 : modelCache.size();
        }
    }

    public static int q8MatrixCacheSize(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q8_MATRIX_CACHE) {
            Q8ModelCache modelCache = Q8_MATRIX_CACHE.get(model);
            return modelCache == null ? 0 : modelCache.size();
        }
    }

    public static long q4KMatrixCacheBytes(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q4K_MATRIX_CACHE) {
            Q4KModelCache modelCache = Q4K_MATRIX_CACHE.get(model);
            return modelCache == null ? 0L : modelCache.bytes();
        }
    }

    public static long q32MatrixCacheBytes(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q32_MATRIX_CACHE) {
            Q32ModelCache modelCache = Q32_MATRIX_CACHE.get(model);
            return modelCache == null ? 0L : modelCache.bytes();
        }
    }

    public static long q5KMatrixCacheBytes(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q5K_MATRIX_CACHE) {
            Q5KModelCache modelCache = Q5K_MATRIX_CACHE.get(model);
            return modelCache == null ? 0L : modelCache.bytes();
        }
    }

    public static long q6KMatrixCacheBytes(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q6K_MATRIX_CACHE) {
            Q6KModelCache modelCache = Q6K_MATRIX_CACHE.get(model);
            return modelCache == null ? 0L : modelCache.bytes();
        }
    }

    public static long q8MatrixCacheBytes(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (Q8_MATRIX_CACHE) {
            Q8ModelCache modelCache = Q8_MATRIX_CACHE.get(model);
            return modelCache == null ? 0L : modelCache.bytes();
        }
    }

    private static long q4KCacheMaxBytes() {
        return cacheMaxBytes("gollek.gguf.q4k.cache_max_bytes");
    }

    private static long q32CacheMaxBytes() {
        return cacheMaxBytes("gollek.gguf.q32.cache_max_bytes");
    }

    private static long q5KCacheMaxBytes() {
        return cacheMaxBytes("gollek.gguf.q5k.cache_max_bytes");
    }

    private static long q6KCacheMaxBytes() {
        return cacheMaxBytes("gollek.gguf.q6k.cache_max_bytes");
    }

    private static long q8CacheMaxBytes() {
        return cacheMaxBytes("gollek.gguf.q8.cache_max_bytes");
    }

    private static long cacheMaxBytes(String property) {
        String configured = System.getProperty(property);
        if (configured == null || configured.isBlank()) {
            return DEFAULT_Q4K_CACHE_MAX_BYTES;
        }
        try {
            return Math.max(0L, parseByteSize(configured.trim()));
        } catch (RuntimeException ignored) {
            return DEFAULT_Q4K_CACHE_MAX_BYTES;
        }
    }

    private static long parseByteSize(String value) {
        if (value.isEmpty()) {
            throw new NumberFormatException("empty byte size");
        }
        char suffix = Character.toLowerCase(value.charAt(value.length() - 1));
        long multiplier = switch (suffix) {
            case 'k' -> 1024L;
            case 'm' -> 1024L * 1024L;
            case 'g' -> 1024L * 1024L * 1024L;
            default -> 1L;
        };
        String digits = multiplier == 1L ? value : value.substring(0, value.length() - 1);
        return Math.multiplyExact(Long.parseLong(digits), multiplier);
    }

    static boolean shouldParallelize(boolean requested, int rowCount, int columns) {
        if (!requested || rowCount <= 1) {
            return false;
        }
        long minOps = Math.max(0L, Long.getLong("gollek.gguf.parallel_min_ops", DEFAULT_PARALLEL_MIN_OPS));
        return (long) rowCount * Math.max(0, columns) >= minOps;
    }

    static int parallelChunkCount(int rowCount) {
        if (rowCount <= 1) {
            return 1;
        }
        int threads = Math.max(1,
                Integer.getInteger("gollek.gguf.parallel_threads", Runtime.getRuntime().availableProcessors()));
        int chunksPerThread = Math.max(1, Integer.getInteger("gollek.gguf.parallel_chunks_per_thread", 1));
        long chunks = Math.min((long) rowCount, (long) threads * chunksPerThread);
        return (int) Math.max(1L, chunks);
    }

    public static Q32Matrix q32Matrix(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (!supportsQ32PreparedType(tensor.typeId())) {
            throw new IllegalArgumentException("Tensor is not a supported 32-value quant type: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / Q4_0_BLOCK_SIZE;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        byte[] source = tensorData(model, tensor).toArray(ValueLayout.JAVA_BYTE);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, Q4_0_BLOCK_SIZE)];
        float[] blockScales = new float[totalBlocks];
        float[] blockBiases = new float[totalBlocks];
        int typeId = tensor.typeId();

        for (int block = 0; block < totalBlocks; block++) {
            int sourceOffset = block * q32BlockBytes(typeId);
            int qBase = block * Q4_0_BLOCK_SIZE;
            blockScales[block] = f16ToF32(leShort(source, sourceOffset));

            if (typeId == GgmlType.Q4_0.id) {
                unpackQ4_0Prepared(source, sourceOffset + 2, quants, qBase);
            } else if (typeId == GgmlType.Q4_1.id) {
                blockBiases[block] = f16ToF32(leShort(source, sourceOffset + 2));
                unpackQ4_1Prepared(source, sourceOffset + 4, quants, qBase);
            } else if (typeId == GgmlType.Q5_0.id) {
                int highBits = leInt(source, sourceOffset + 2);
                unpackQ5_0Prepared(source, sourceOffset + 6, highBits, quants, qBase);
            } else if (typeId == GgmlType.Q5_1.id) {
                blockBiases[block] = f16ToF32(leShort(source, sourceOffset + 2));
                int highBits = leInt(source, sourceOffset + 4);
                unpackQ5_1Prepared(source, sourceOffset + 8, highBits, quants, qBase);
            }
        }
        return new Q32Matrix(columns, rows, blocksPerRow, quants, blockScales, blockBiases);
    }

    public static Q4KMatrix q4KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (tensor.typeId() != GgmlType.Q4_K.id) {
            throw new IllegalArgumentException("Tensor is not Q4_K: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / QK_K;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        byte[] source = tensorData(model, tensor).toArray(ValueLayout.JAVA_BYTE);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, QK_K)];
        float[] groupScales = new float[Math.multiplyExact(totalBlocks, 8)];
        float[] groupMins = new float[groupScales.length];

        for (int block = 0; block < totalBlocks; block++) {
            long blockOffset = block * (long) Q4_K_BLOCK_BYTES;
            float d = f16ToF32(leShort(source, blockOffset));
            float dMin = f16ToF32(leShort(source, blockOffset + 2));
            long scalesOffset = blockOffset + 4;
            long quantsOffset = blockOffset + 16;
            int groupBase = block * 8;
            int unpackedBase = block * QK_K;
            int scaleIndex = 0;

            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
                ScaleMin first = scaleMinK4(source, scalesOffset, scaleIndex++);
                ScaleMin second = scaleMinK4(source, scalesOffset, scaleIndex++);
                int groupIndex = groupBase + (superBlockOffset / 64) * 2;
                groupScales[groupIndex] = d * first.scale();
                groupMins[groupIndex] = dMin * first.min();
                groupScales[groupIndex + 1] = d * second.scale();
                groupMins[groupIndex + 1] = dMin * second.min();
                int packedBase = (int) quantsOffset + (superBlockOffset / 64) * 32;
                int outBase = unpackedBase + superBlockOffset;
                for (int i = 0; i < 32; i++) {
                    int quant = source[packedBase + i] & 0xFF;
                    quants[outBase + i] = (byte) (quant & 0x0F);
                    quants[outBase + 32 + i] = (byte) ((quant >>> 4) & 0x0F);
                }
            }
        }
        return new Q4KMatrix(columns, rows, blocksPerRow, quants, groupScales, groupMins);
    }

    public static Q5KMatrix q5KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (tensor.typeId() != GgmlType.Q5_K.id) {
            throw new IllegalArgumentException("Tensor is not Q5_K: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / QK_K;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        byte[] source = tensorData(model, tensor).toArray(ValueLayout.JAVA_BYTE);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, QK_K)];
        float[] groupScales = new float[Math.multiplyExact(totalBlocks, 8)];
        float[] groupMins = new float[groupScales.length];

        for (int block = 0; block < totalBlocks; block++) {
            long blockOffset = block * (long) Q5_K_BLOCK_BYTES;
            float d = f16ToF32(leShort(source, blockOffset));
            float dMin = f16ToF32(leShort(source, blockOffset + 2));
            long scalesOffset = blockOffset + 4;
            long highBitsOffset = blockOffset + 16;
            long quantsOffset = blockOffset + 48;
            int groupBase = block * 8;
            int unpackedBase = block * QK_K;
            int scaleIndex = 0;
            int highMaskLow = 1;
            int highMaskHigh = 2;

            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
                ScaleMin first = scaleMinK4(source, scalesOffset, scaleIndex++);
                ScaleMin second = scaleMinK4(source, scalesOffset, scaleIndex++);
                int groupIndex = groupBase + (superBlockOffset / 64) * 2;
                groupScales[groupIndex] = d * first.scale();
                groupMins[groupIndex] = dMin * first.min();
                groupScales[groupIndex + 1] = d * second.scale();
                groupMins[groupIndex + 1] = dMin * second.min();

                int packedBase = (int) quantsOffset + (superBlockOffset / 64) * 32;
                int highBase = (int) highBitsOffset;
                int outBase = unpackedBase + superBlockOffset;
                for (int i = 0; i < 32; i++) {
                    int quant = source[packedBase + i] & 0xFF;
                    int highBits = source[highBase + i] & 0xFF;
                    int low = (quant & 0x0F) + ((highBits & highMaskLow) != 0 ? 16 : 0);
                    int high = (quant >>> 4) + ((highBits & highMaskHigh) != 0 ? 16 : 0);
                    quants[outBase + i] = (byte) low;
                    quants[outBase + 32 + i] = (byte) high;
                }
                highMaskLow <<= 2;
                highMaskHigh <<= 2;
            }
        }
        return new Q5KMatrix(columns, rows, blocksPerRow, quants, groupScales, groupMins);
    }

    public static Q6KMatrix q6KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (tensor.typeId() != GgmlType.Q6_K.id) {
            throw new IllegalArgumentException("Tensor is not Q6_K: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / QK_K;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        byte[] source = tensorData(model, tensor).toArray(ValueLayout.JAVA_BYTE);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, QK_K)];
        float[] groupScales = new float[Math.multiplyExact(totalBlocks, 16)];

        for (int block = 0; block < totalBlocks; block++) {
            long blockOffset = block * (long) Q6_K_BLOCK_BYTES;
            float d = f16ToF32(leShort(source, blockOffset + 208));
            int groupBase = block * 16;
            int unpackedBase = block * QK_K;
            int scalesOffset = (int) blockOffset + 192;
            for (int group = 0; group < 16; group++) {
                groupScales[groupBase + group] = d * source[scalesOffset + group];
            }

            int lowBitsOffset = (int) blockOffset;
            int highBitsOffset = (int) blockOffset + 128;
            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 128) {
                int lowBitsBase = lowBitsOffset + superBlockOffset / 2;
                int highBitsBase = highBitsOffset + superBlockOffset / 4;
                for (int i = 0; i < 32; i++) {
                    int lowA = source[lowBitsBase + i] & 0xFF;
                    int lowB = source[lowBitsBase + 32 + i] & 0xFF;
                    int highBits = source[highBitsBase + i] & 0xFF;
                    int q1 = ((lowA & 0x0F) | (((highBits >>> 0) & 0x03) << 4)) - 32;
                    int q2 = ((lowB & 0x0F) | (((highBits >>> 2) & 0x03) << 4)) - 32;
                    int q3 = ((lowA >>> 4) | (((highBits >>> 4) & 0x03) << 4)) - 32;
                    int q4 = ((lowB >>> 4) | (((highBits >>> 6) & 0x03) << 4)) - 32;
                    quants[unpackedBase + superBlockOffset + i] = (byte) q1;
                    quants[unpackedBase + superBlockOffset + 32 + i] = (byte) q2;
                    quants[unpackedBase + superBlockOffset + 64 + i] = (byte) q3;
                    quants[unpackedBase + superBlockOffset + 96 + i] = (byte) q4;
                }
            }
        }
        return new Q6KMatrix(columns, rows, blocksPerRow, quants, groupScales);
    }

    public static Q8Matrix q8Matrix(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (tensor.typeId() != GgmlType.Q8_0.id) {
            throw new IllegalArgumentException("Tensor is not Q8_0: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / Q8_0_BLOCK_SIZE;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        byte[] source = tensorData(model, tensor).toArray(ValueLayout.JAVA_BYTE);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, Q8_0_BLOCK_SIZE)];
        float[] blockScales = new float[totalBlocks];

        for (int block = 0; block < totalBlocks; block++) {
            int sourceOffset = block * Q8_0_BLOCK_BYTES;
            blockScales[block] = f16ToF32(leShort(source, sourceOffset));
            System.arraycopy(
                    source,
                    sourceOffset + 2,
                    quants,
                    block * Q8_0_BLOCK_SIZE,
                    Q8_0_BLOCK_SIZE);
        }
        return new Q8Matrix(columns, rows, blocksPerRow, quants, blockScales);
    }

    private static Q4KMatrixKey q4KMatrixKey(GGUFTensorInfo tensor) {
        return new Q4KMatrixKey(
                tensor.name(),
                tensor.typeId(),
                tensor.offset(),
                tensor.sizeInBytes(),
                matrixColumns(tensor),
                matrixRows(tensor));
    }

    private static Q32MatrixKey q32MatrixKey(GGUFTensorInfo tensor) {
        return new Q32MatrixKey(
                tensor.name(),
                tensor.typeId(),
                tensor.offset(),
                tensor.sizeInBytes(),
                matrixColumns(tensor),
                matrixRows(tensor));
    }

    private static Q5KMatrixKey q5KMatrixKey(GGUFTensorInfo tensor) {
        return new Q5KMatrixKey(
                tensor.name(),
                tensor.typeId(),
                tensor.offset(),
                tensor.sizeInBytes(),
                matrixColumns(tensor),
                matrixRows(tensor));
    }

    private static Q6KMatrixKey q6KMatrixKey(GGUFTensorInfo tensor) {
        return new Q6KMatrixKey(
                tensor.name(),
                tensor.typeId(),
                tensor.offset(),
                tensor.sizeInBytes(),
                matrixColumns(tensor),
                matrixRows(tensor));
    }

    private static Q8MatrixKey q8MatrixKey(GGUFTensorInfo tensor) {
        return new Q8MatrixKey(
                tensor.name(),
                tensor.typeId(),
                tensor.offset(),
                tensor.sizeInBytes(),
                matrixColumns(tensor),
                matrixRows(tensor));
    }

    public static void matVecRows(Q4KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        matVecRows(matrix, vector, output, rowCount, parallel, Q4K_WORK_BUFFER.get());
    }

    public static void matVecRows(Q32Matrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(output, "output");
        if (vector.length < matrix.columns()) {
            throw new IllegalArgumentException(
                    "Vector length " + vector.length + " is smaller than columns " + matrix.columns());
        }
        if (rowCount < 0 || rowCount > matrix.rows()) {
            throw new IllegalArgumentException(
                    "Requested row count " + rowCount + " is outside tensor rows " + matrix.rows());
        }
        if (output.length < rowCount) {
            throw new IllegalArgumentException(
                    "Output length " + output.length + " is smaller than requested rows " + rowCount);
        }

        float[] vectorGroupSums = vector32GroupSums(vector, matrix.columns(), Q4K_WORK_BUFFER.get());
        if (shouldParallelize(parallel, rowCount, matrix.columns())) {
            int chunks = parallelChunkCount(rowCount);
            IntStream.range(0, chunks)
                    .parallel()
                    .forEach(chunk -> {
                        int start = (int) ((long) chunk * rowCount / chunks);
                        int end = (int) ((long) (chunk + 1) * rowCount / chunks);
                        fillMatVecRowsQ32(matrix, vector, vectorGroupSums, output, start, end);
                    });
            return;
        }
        fillMatVecRowsQ32(matrix, vector, vectorGroupSums, output, 0, rowCount);
    }

    public static void matVecRows(Q5KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(output, "output");
        if (vector.length < matrix.columns()) {
            throw new IllegalArgumentException(
                    "Vector length " + vector.length + " is smaller than columns " + matrix.columns());
        }
        if (rowCount < 0 || rowCount > matrix.rows()) {
            throw new IllegalArgumentException(
                    "Requested row count " + rowCount + " is outside tensor rows " + matrix.rows());
        }
        if (output.length < rowCount) {
            throw new IllegalArgumentException(
                    "Output length " + output.length + " is smaller than requested rows " + rowCount);
        }

        float[] vectorGroupSums = q4KVectorGroupSums(vector, matrix.columns(), Q4K_WORK_BUFFER.get());
        if (shouldParallelize(parallel, rowCount, matrix.columns())) {
            int chunks = parallelChunkCount(rowCount);
            IntStream.range(0, chunks)
                    .parallel()
                    .forEach(chunk -> {
                        int start = (int) ((long) chunk * rowCount / chunks);
                        int end = (int) ((long) (chunk + 1) * rowCount / chunks);
                        fillMatVecRowsQ5K(matrix, vector, vectorGroupSums, output, start, end);
                    });
            return;
        }
        fillMatVecRowsQ5K(matrix, vector, vectorGroupSums, output, 0, rowCount);
    }

    public static void matVecRows(Q6KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(output, "output");
        if (vector.length < matrix.columns()) {
            throw new IllegalArgumentException(
                    "Vector length " + vector.length + " is smaller than columns " + matrix.columns());
        }
        if (rowCount < 0 || rowCount > matrix.rows()) {
            throw new IllegalArgumentException(
                    "Requested row count " + rowCount + " is outside tensor rows " + matrix.rows());
        }
        if (output.length < rowCount) {
            throw new IllegalArgumentException(
                    "Output length " + output.length + " is smaller than requested rows " + rowCount);
        }

        if (shouldParallelize(parallel, rowCount, matrix.columns())) {
            int chunks = parallelChunkCount(rowCount);
            IntStream.range(0, chunks)
                    .parallel()
                    .forEach(chunk -> {
                        int start = (int) ((long) chunk * rowCount / chunks);
                        int end = (int) ((long) (chunk + 1) * rowCount / chunks);
                        fillMatVecRowsQ6K(matrix, vector, output, start, end);
                    });
            return;
        }
        fillMatVecRowsQ6K(matrix, vector, output, 0, rowCount);
    }

    public static void matVecRows(Q8Matrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(output, "output");
        if (vector.length < matrix.columns()) {
            throw new IllegalArgumentException(
                    "Vector length " + vector.length + " is smaller than columns " + matrix.columns());
        }
        if (rowCount < 0 || rowCount > matrix.rows()) {
            throw new IllegalArgumentException(
                    "Requested row count " + rowCount + " is outside tensor rows " + matrix.rows());
        }
        if (output.length < rowCount) {
            throw new IllegalArgumentException(
                    "Output length " + output.length + " is smaller than requested rows " + rowCount);
        }

        if (shouldParallelize(parallel, rowCount, matrix.columns())) {
            int chunks = parallelChunkCount(rowCount);
            IntStream.range(0, chunks)
                    .parallel()
                    .forEach(chunk -> {
                        int start = (int) ((long) chunk * rowCount / chunks);
                        int end = (int) ((long) (chunk + 1) * rowCount / chunks);
                        fillMatVecRowsQ8(matrix, vector, output, start, end);
                    });
            return;
        }
        fillMatVecRowsQ8(matrix, vector, output, 0, rowCount);
    }

    public static void matVecRows(
            Q4KMatrix matrix,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Q4KWorkBuffer workBuffer) {
        Objects.requireNonNull(matrix, "matrix");
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(workBuffer, "workBuffer");
        if (vector.length < matrix.columns()) {
            throw new IllegalArgumentException(
                    "Vector length " + vector.length + " is smaller than columns " + matrix.columns());
        }
        if (rowCount < 0 || rowCount > matrix.rows()) {
            throw new IllegalArgumentException(
                    "Requested row count " + rowCount + " is outside tensor rows " + matrix.rows());
        }
        if (output.length < rowCount) {
            throw new IllegalArgumentException(
                "Output length " + output.length + " is smaller than requested rows " + rowCount);
        }

        float[] vectorGroupSums = q4KVectorGroupSums(vector, matrix.columns(), workBuffer);
        if (shouldParallelize(parallel, rowCount, matrix.columns())) {
            int chunks = parallelChunkCount(rowCount);
            IntStream.range(0, chunks)
                    .parallel()
                    .forEach(chunk -> {
                        int start = (int) ((long) chunk * rowCount / chunks);
                        int end = (int) ((long) (chunk + 1) * rowCount / chunks);
                        fillMatVecRowsQ4K(matrix, vector, vectorGroupSums, output, start, end);
                    });
            return;
        }
        fillMatVecRowsQ4K(matrix, vector, vectorGroupSums, output, 0, rowCount);
    }

    static void dequantizeQ4KBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        long scalesOffset = blockOffset + 4;
        long quantsOffset = blockOffset + 16;
        int outBase = dstOffset;
        int scaleIndex = 0;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
            ScaleMin first = scaleMinK4(segment, scalesOffset, scaleIndex++);
            ScaleMin second = scaleMinK4(segment, scalesOffset, scaleIndex++);
            float d1 = d * first.scale();
            float m1 = dMin * first.min();
            float d2 = d * second.scale();
            float m2 = dMin * second.min();

            for (int i = 0; i < 32; i++) {
                int quant = u8(segment, quantsOffset + i);
                dst[outBase + superBlockOffset + i] = d1 * (quant & 0x0F) - m1;
            }
            for (int i = 0; i < 32; i++) {
                int quant = u8(segment, quantsOffset + i);
                dst[outBase + superBlockOffset + 32 + i] = d2 * ((quant >>> 4) & 0x0F) - m2;
            }
            quantsOffset += 32;
        }
    }

    static void dequantizeQ4_0Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        long quantsOffset = blockOffset + 2;
        for (int i = 0; i < 16; i++) {
            int quant = u8(segment, quantsOffset + i);
            dst[dstOffset + i] = ((quant & 0x0F) - 8) * d;
            dst[dstOffset + 16 + i] = ((quant >>> 4) - 8) * d;
        }
    }

    static void dequantizeQ4_1Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        long quantsOffset = blockOffset + 4;
        for (int i = 0; i < 16; i++) {
            int quant = u8(segment, quantsOffset + i);
            dst[dstOffset + i] = (quant & 0x0F) * d + m;
            dst[dstOffset + 16 + i] = (quant >>> 4) * d + m;
        }
    }

    static void dequantizeQ5_0Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        int highBits = segment.get(LE_INT, blockOffset + 2);
        long quantsOffset = blockOffset + 6;
        for (int i = 0; i < 16; i++) {
            int quant = u8(segment, quantsOffset + i);
            int low = (quant & 0x0F) | (((highBits >>> i) & 1) << 4);
            int high = ((quant >>> 4) & 0x0F) | (((highBits >>> (i + 16)) & 1) << 4);
            dst[dstOffset + i] = (low - 16) * d;
            dst[dstOffset + 16 + i] = (high - 16) * d;
        }
    }

    static void dequantizeQ5_1Block(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        int highBits = segment.get(LE_INT, blockOffset + 4);
        long quantsOffset = blockOffset + 8;
        for (int i = 0; i < 16; i++) {
            int quant = u8(segment, quantsOffset + i);
            int low = (quant & 0x0F) | (((highBits >>> i) & 1) << 4);
            int high = ((quant >>> 4) & 0x0F) | (((highBits >>> (i + 16)) & 1) << 4);
            dst[dstOffset + i] = low * d + m;
            dst[dstOffset + 16 + i] = high * d + m;
        }
    }

    static void dequantizeQ5KBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        long scalesOffset = blockOffset + 4;
        long highBitsOffset = blockOffset + 16;
        long quantsOffset = blockOffset + 48;
        int scaleIndex = 0;
        int highMaskLow = 1;
        int highMaskHigh = 2;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
            ScaleMin first = scaleMinK4(segment, scalesOffset, scaleIndex++);
            ScaleMin second = scaleMinK4(segment, scalesOffset, scaleIndex++);
            float d1 = d * first.scale();
            float m1 = dMin * first.min();
            float d2 = d * second.scale();
            float m2 = dMin * second.min();

            for (int i = 0; i < 32; i++) {
                int quant = u8(segment, quantsOffset + i);
                int highBits = u8(segment, highBitsOffset + i);
                int low = (quant & 0x0F) + ((highBits & highMaskLow) != 0 ? 16 : 0);
                int high = (quant >>> 4) + ((highBits & highMaskHigh) != 0 ? 16 : 0);
                dst[dstOffset + superBlockOffset + i] = d1 * low - m1;
                dst[dstOffset + superBlockOffset + 32 + i] = d2 * high - m2;
            }
            quantsOffset += 32;
            highMaskLow <<= 2;
            highMaskHigh <<= 2;
        }
    }

    static void dequantizeQ6KBlock(MemorySegment segment, long blockOffset, float[] dst, int dstOffset) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 208));
        long lowBitsOffset = blockOffset;
        long highBitsOffset = blockOffset + 128;
        long scalesOffset = blockOffset + 192;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 128) {
            int scaleBase = superBlockOffset / 16;
            long lowBitsBase = lowBitsOffset + (superBlockOffset / 2L);
            long highBitsBase = highBitsOffset + (superBlockOffset / 4L);
            for (int i = 0; i < 32; i++) {
                int scaleIndex = scaleBase + i / 16;
                int lowA = u8(segment, lowBitsBase + i);
                int lowB = u8(segment, lowBitsBase + 32 + i);
                int highBits = u8(segment, highBitsBase + i);
                int q1 = ((lowA & 0x0F) | (((highBits >>> 0) & 0x03) << 4)) - 32;
                int q2 = ((lowB & 0x0F) | (((highBits >>> 2) & 0x03) << 4)) - 32;
                int q3 = ((lowA >>> 4) | (((highBits >>> 4) & 0x03) << 4)) - 32;
                int q4 = ((lowB >>> 4) | (((highBits >>> 6) & 0x03) << 4)) - 32;
                dst[dstOffset + superBlockOffset + i] =
                        d * segment.get(ValueLayout.JAVA_BYTE, scalesOffset + scaleIndex) * q1;
                dst[dstOffset + superBlockOffset + 32 + i] =
                        d * segment.get(ValueLayout.JAVA_BYTE, scalesOffset + scaleIndex + 2) * q2;
                dst[dstOffset + superBlockOffset + 64 + i] =
                        d * segment.get(ValueLayout.JAVA_BYTE, scalesOffset + scaleIndex + 4) * q3;
                dst[dstOffset + superBlockOffset + 96 + i] =
                        d * segment.get(ValueLayout.JAVA_BYTE, scalesOffset + scaleIndex + 6) * q4;
            }
        }
    }

    private static void dequantizeRow(
            MemorySegment segment,
            long rowOffset,
            int typeId,
            int columns,
            float[] dst,
            int dstOffset) {
        if (typeId == GgmlType.F32.id) {
            for (int i = 0; i < columns; i++) {
                dst[dstOffset + i] = segment.get(LE_FLOAT, rowOffset + i * Float.BYTES);
            }
            return;
        }
        if (typeId == GgmlType.F16.id) {
            for (int i = 0; i < columns; i++) {
                dst[dstOffset + i] = f16ToF32(segment.get(LE_SHORT, rowOffset + i * 2L));
            }
            return;
        }
        if (typeId == GgmlType.BF16.id) {
            for (int i = 0; i < columns; i++) {
                int bits = (segment.get(LE_SHORT, rowOffset + i * 2L) & 0xFFFF) << 16;
                dst[dstOffset + i] = Float.intBitsToFloat(bits);
            }
            return;
        }
        if (typeId == GgmlType.Q4_0.id) {
            for (int block = 0; block < columns / Q4_0_BLOCK_SIZE; block++) {
                dequantizeQ4_0Block(
                        segment,
                        rowOffset + block * (long) Q4_0_BLOCK_BYTES,
                        dst,
                        dstOffset + block * Q4_0_BLOCK_SIZE);
            }
            return;
        }
        if (typeId == GgmlType.Q4_1.id) {
            for (int block = 0; block < columns / Q4_0_BLOCK_SIZE; block++) {
                dequantizeQ4_1Block(
                        segment,
                        rowOffset + block * (long) Q4_1_BLOCK_BYTES,
                        dst,
                        dstOffset + block * Q4_0_BLOCK_SIZE);
            }
            return;
        }
        if (typeId == GgmlType.Q5_0.id) {
            for (int block = 0; block < columns / Q4_0_BLOCK_SIZE; block++) {
                dequantizeQ5_0Block(
                        segment,
                        rowOffset + block * (long) Q5_0_BLOCK_BYTES,
                        dst,
                        dstOffset + block * Q4_0_BLOCK_SIZE);
            }
            return;
        }
        if (typeId == GgmlType.Q5_1.id) {
            for (int block = 0; block < columns / Q4_0_BLOCK_SIZE; block++) {
                dequantizeQ5_1Block(
                        segment,
                        rowOffset + block * (long) Q5_1_BLOCK_BYTES,
                        dst,
                        dstOffset + block * Q4_0_BLOCK_SIZE);
            }
            return;
        }
        if (typeId == GgmlType.Q8_0.id) {
            for (int block = 0; block < columns / 32; block++) {
                long blockOffset = rowOffset + block * 34L;
                float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
                for (int i = 0; i < 32; i++) {
                    dst[dstOffset + block * 32 + i] = segment.get(ValueLayout.JAVA_BYTE, blockOffset + 2 + i) * d;
                }
            }
            return;
        }
        if (typeId == GgmlType.Q4_K.id) {
            for (int block = 0; block < columns / QK_K; block++) {
                dequantizeQ4KBlock(segment, rowOffset + block * Q4_K_BLOCK_BYTES, dst, dstOffset + block * QK_K);
            }
            return;
        }
        if (typeId == GgmlType.Q5_K.id) {
            for (int block = 0; block < columns / QK_K; block++) {
                dequantizeQ5KBlock(segment, rowOffset + block * Q5_K_BLOCK_BYTES, dst, dstOffset + block * QK_K);
            }
            return;
        }
        if (typeId == GgmlType.Q6_K.id) {
            for (int block = 0; block < columns / QK_K; block++) {
                dequantizeQ6KBlock(segment, rowOffset + block * Q6_K_BLOCK_BYTES, dst, dstOffset + block * QK_K);
            }
            return;
        }
        throw new UnsupportedOperationException("Unsupported GGUF row dequant type id: " + typeId);
    }

    private static float dotRow(
            MemorySegment segment,
            long rowOffset,
            int typeId,
            int columns,
            float[] vector,
            int vectorOffset) {
        if (typeId == GgmlType.F32.id) {
            float sum = 0.0f;
            for (int i = 0; i < columns; i++) {
                sum += segment.get(LE_FLOAT, rowOffset + i * Float.BYTES) * vector[vectorOffset + i];
            }
            return sum;
        }
        if (typeId == GgmlType.F16.id) {
            float sum = 0.0f;
            for (int i = 0; i < columns; i++) {
                sum += f16ToF32(segment.get(LE_SHORT, rowOffset + i * 2L)) * vector[vectorOffset + i];
            }
            return sum;
        }
        if (typeId == GgmlType.BF16.id) {
            float sum = 0.0f;
            for (int i = 0; i < columns; i++) {
                int bits = (segment.get(LE_SHORT, rowOffset + i * 2L) & 0xFFFF) << 16;
                sum += Float.intBitsToFloat(bits) * vector[vectorOffset + i];
            }
            return sum;
        }
        if (typeId == GgmlType.Q4_0.id) {
            return dotRowQ4_0(segment, rowOffset, columns, vector, vectorOffset);
        }
        if (typeId == GgmlType.Q4_1.id) {
            return dotRowQ4_1(segment, rowOffset, columns, vector, vectorOffset);
        }
        if (typeId == GgmlType.Q5_0.id) {
            return dotRowQ5_0(segment, rowOffset, columns, vector, vectorOffset);
        }
        if (typeId == GgmlType.Q5_1.id) {
            return dotRowQ5_1(segment, rowOffset, columns, vector, vectorOffset);
        }
        if (typeId == GgmlType.Q8_0.id) {
            float sum = 0.0f;
            for (int block = 0; block < columns / 32; block++) {
                long blockOffset = rowOffset + block * 34L;
                float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
                int vectorBase = vectorOffset + block * 32;
                for (int i = 0; i < 32; i++) {
                    sum += segment.get(ValueLayout.JAVA_BYTE, blockOffset + 2 + i) * d * vector[vectorBase + i];
                }
            }
            return sum;
        }
        if (typeId == GgmlType.Q4_K.id) {
            return dotRowQ4K(segment, rowOffset, columns, vector, vectorOffset);
        }
        if (typeId == GgmlType.Q5_K.id) {
            return dotRowQ5K(segment, rowOffset, columns, vector, vectorOffset);
        }
        if (typeId == GgmlType.Q6_K.id) {
            return dotRowQ6K(segment, rowOffset, columns, vector, vectorOffset);
        }
        throw new UnsupportedOperationException("Unsupported GGUF row dot type id: " + typeId);
    }

    private static float dotRowQ4_0(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum = 0.0f;
        for (int block = 0; block < columns / Q4_0_BLOCK_SIZE; block++) {
            long blockOffset = rowOffset + block * (long) Q4_0_BLOCK_BYTES;
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            int vectorBase = vectorOffset + block * Q4_0_BLOCK_SIZE;
            long quantsOffset = blockOffset + 2;
            float quantDot = 0.0f;
            for (int i = 0; i < 16; i++) {
                int quant = u8(segment, quantsOffset + i);
                quantDot += ((quant & 0x0F) - 8) * vector[vectorBase + i]
                        + ((quant >>> 4) - 8) * vector[vectorBase + 16 + i];
            }
            sum += d * quantDot;
        }
        return sum;
    }

    private static float dotRowQ4_1(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum = 0.0f;
        for (int block = 0; block < columns / Q4_0_BLOCK_SIZE; block++) {
            long blockOffset = rowOffset + block * (long) Q4_1_BLOCK_BYTES;
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
            int vectorBase = vectorOffset + block * Q4_0_BLOCK_SIZE;
            long quantsOffset = blockOffset + 4;
            float quantDot = 0.0f;
            float vectorSum = 0.0f;
            for (int i = 0; i < 16; i++) {
                int quant = u8(segment, quantsOffset + i);
                float first = vector[vectorBase + i];
                float second = vector[vectorBase + 16 + i];
                quantDot += (quant & 0x0F) * first + (quant >>> 4) * second;
                vectorSum += first + second;
            }
            sum += d * quantDot + m * vectorSum;
        }
        return sum;
    }

    private static float dotRowQ5_0(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum = 0.0f;
        for (int block = 0; block < columns / Q4_0_BLOCK_SIZE; block++) {
            long blockOffset = rowOffset + block * (long) Q5_0_BLOCK_BYTES;
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            int highBits = segment.get(LE_INT, blockOffset + 2);
            int vectorBase = vectorOffset + block * Q4_0_BLOCK_SIZE;
            long quantsOffset = blockOffset + 6;
            float quantDot = 0.0f;
            for (int i = 0; i < 16; i++) {
                int quant = u8(segment, quantsOffset + i);
                int low = (quant & 0x0F) | (((highBits >>> i) & 1) << 4);
                int high = ((quant >>> 4) & 0x0F) | (((highBits >>> (i + 16)) & 1) << 4);
                quantDot += (low - 16) * vector[vectorBase + i]
                        + (high - 16) * vector[vectorBase + 16 + i];
            }
            sum += d * quantDot;
        }
        return sum;
    }

    private static float dotRowQ5_1(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum = 0.0f;
        for (int block = 0; block < columns / Q4_0_BLOCK_SIZE; block++) {
            long blockOffset = rowOffset + block * (long) Q5_1_BLOCK_BYTES;
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
            int highBits = segment.get(LE_INT, blockOffset + 4);
            int vectorBase = vectorOffset + block * Q4_0_BLOCK_SIZE;
            long quantsOffset = blockOffset + 8;
            float quantDot = 0.0f;
            float vectorSum = 0.0f;
            for (int i = 0; i < 16; i++) {
                int quant = u8(segment, quantsOffset + i);
                int low = (quant & 0x0F) | (((highBits >>> i) & 1) << 4);
                int high = ((quant >>> 4) & 0x0F) | (((highBits >>> (i + 16)) & 1) << 4);
                float first = vector[vectorBase + i];
                float second = vector[vectorBase + 16 + i];
                quantDot += low * first + high * second;
                vectorSum += first + second;
            }
            sum += d * quantDot + m * vectorSum;
        }
        return sum;
    }

    private static float dotRowQ4K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum = 0.0f;
        for (int block = 0; block < columns / QK_K; block++) {
            long blockOffset = rowOffset + block * Q4_K_BLOCK_BYTES;
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
            long scalesOffset = blockOffset + 4;
            long quantsOffset = blockOffset + 16;
            int vectorBase = vectorOffset + block * QK_K;
            int scaleIndex = 0;

            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
                ScaleMin first = scaleMinK4(segment, scalesOffset, scaleIndex++);
                ScaleMin second = scaleMinK4(segment, scalesOffset, scaleIndex++);
                float d1 = d * first.scale();
                float m1 = dMin * first.min();
                float d2 = d * second.scale();
                float m2 = dMin * second.min();

                float vectorSum1 = 0.0f;
                float vectorSum2 = 0.0f;
                float quantDot1 = 0.0f;
                float quantDot2 = 0.0f;
                for (int i = 0; i < 32; i++) {
                    int quant = u8(segment, quantsOffset + i);
                    float v1 = vector[vectorBase + superBlockOffset + i];
                    float v2 = vector[vectorBase + superBlockOffset + 32 + i];
                    quantDot1 += (quant & 0x0F) * v1;
                    quantDot2 += ((quant >>> 4) & 0x0F) * v2;
                    vectorSum1 += v1;
                    vectorSum2 += v2;
                }
                sum += d1 * quantDot1 - m1 * vectorSum1
                        + d2 * quantDot2 - m2 * vectorSum2;
                quantsOffset += 32;
            }
        }
        return sum;
    }

    private static float dotRowQ5K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum = 0.0f;
        for (int block = 0; block < columns / QK_K; block++) {
            long blockOffset = rowOffset + block * Q5_K_BLOCK_BYTES;
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
            long scalesOffset = blockOffset + 4;
            long highBitsOffset = blockOffset + 16;
            long quantsOffset = blockOffset + 48;
            int vectorBase = vectorOffset + block * QK_K;
            int scaleIndex = 0;
            int highMaskLow = 1;
            int highMaskHigh = 2;

            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
                ScaleMin first = scaleMinK4(segment, scalesOffset, scaleIndex++);
                ScaleMin second = scaleMinK4(segment, scalesOffset, scaleIndex++);
                float d1 = d * first.scale();
                float m1 = dMin * first.min();
                float d2 = d * second.scale();
                float m2 = dMin * second.min();

                float vectorSum1 = 0.0f;
                float vectorSum2 = 0.0f;
                float quantDot1 = 0.0f;
                float quantDot2 = 0.0f;
                for (int i = 0; i < 32; i++) {
                    int quant = u8(segment, quantsOffset + i);
                    int highBits = u8(segment, highBitsOffset + i);
                    int low = (quant & 0x0F) + ((highBits & highMaskLow) != 0 ? 16 : 0);
                    int high = (quant >>> 4) + ((highBits & highMaskHigh) != 0 ? 16 : 0);
                    float v1 = vector[vectorBase + superBlockOffset + i];
                    float v2 = vector[vectorBase + superBlockOffset + 32 + i];
                    quantDot1 += low * v1;
                    quantDot2 += high * v2;
                    vectorSum1 += v1;
                    vectorSum2 += v2;
                }
                sum += d1 * quantDot1 - m1 * vectorSum1
                        + d2 * quantDot2 - m2 * vectorSum2;
                quantsOffset += 32;
                highMaskLow <<= 2;
                highMaskHigh <<= 2;
            }
        }
        return sum;
    }

    private static float dotRowQ6K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum = 0.0f;
        for (int block = 0; block < columns / QK_K; block++) {
            long blockOffset = rowOffset + block * Q6_K_BLOCK_BYTES;
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 208));
            long lowBitsOffset = blockOffset;
            long highBitsOffset = blockOffset + 128;
            long scalesOffset = blockOffset + 192;
            int vectorBase = vectorOffset + block * QK_K;

            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 128) {
                int scaleBase = superBlockOffset / 16;
                long lowBitsBase = lowBitsOffset + (superBlockOffset / 2L);
                long highBitsBase = highBitsOffset + (superBlockOffset / 4L);
                for (int i = 0; i < 32; i++) {
                    int scaleIndex = scaleBase + i / 16;
                    int lowA = u8(segment, lowBitsBase + i);
                    int lowB = u8(segment, lowBitsBase + 32 + i);
                    int highBits = u8(segment, highBitsBase + i);
                    int q1 = ((lowA & 0x0F) | (((highBits >>> 0) & 0x03) << 4)) - 32;
                    int q2 = ((lowB & 0x0F) | (((highBits >>> 2) & 0x03) << 4)) - 32;
                    int q3 = ((lowA >>> 4) | (((highBits >>> 4) & 0x03) << 4)) - 32;
                    int q4 = ((lowB >>> 4) | (((highBits >>> 6) & 0x03) << 4)) - 32;
                    sum += d * segment.get(ValueLayout.JAVA_BYTE, scalesOffset + scaleIndex) * q1
                            * vector[vectorBase + superBlockOffset + i];
                    sum += d * segment.get(ValueLayout.JAVA_BYTE, scalesOffset + scaleIndex + 2) * q2
                            * vector[vectorBase + superBlockOffset + 32 + i];
                    sum += d * segment.get(ValueLayout.JAVA_BYTE, scalesOffset + scaleIndex + 4) * q3
                            * vector[vectorBase + superBlockOffset + 64 + i];
                    sum += d * segment.get(ValueLayout.JAVA_BYTE, scalesOffset + scaleIndex + 6) * q4
                            * vector[vectorBase + superBlockOffset + 96 + i];
                }
            }
        }
        return sum;
    }

    private static float dotRowQ4KWithGroupSums(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            float[] vectorGroupSums) {
        float sum = 0.0f;
        for (int block = 0; block < columns / QK_K; block++) {
            long blockOffset = rowOffset + block * Q4_K_BLOCK_BYTES;
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
            float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
            long scalesOffset = blockOffset + 4;
            long quantsOffset = blockOffset + 16;
            int vectorBase = block * QK_K;
            int scaleIndex = 0;

            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
                ScaleMin first = scaleMinK4(segment, scalesOffset, scaleIndex++);
                ScaleMin second = scaleMinK4(segment, scalesOffset, scaleIndex++);
                float d1 = d * first.scale();
                float m1 = dMin * first.min();
                float d2 = d * second.scale();
                float m2 = dMin * second.min();

                int groupIndex = block * 8 + scaleIndex - 2;
                float quantDot1 = 0.0f;
                float quantDot2 = 0.0f;
                for (int i = 0; i < 32; i++) {
                    int quant = u8(segment, quantsOffset + i);
                    quantDot1 += (quant & 0x0F) * vector[vectorBase + superBlockOffset + i];
                    quantDot2 += ((quant >>> 4) & 0x0F) * vector[vectorBase + superBlockOffset + 32 + i];
                }
                sum += d1 * quantDot1 - m1 * vectorGroupSums[groupIndex]
                        + d2 * quantDot2 - m2 * vectorGroupSums[groupIndex + 1];
                quantsOffset += 32;
            }
        }
        return sum;
    }

    private static void matVecRowsQ4K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        float[] vectorGroupSums = q4KVectorGroupSums(vector, columns, Q4K_WORK_BUFFER.get());
        if (shouldParallelize(parallel, rowCount, columns)) {
            try {
                int chunks = parallelChunkCount(rowCount);
                IntStream.range(0, chunks)
                        .parallel()
                        .forEach(chunk -> {
                            int start = (int) ((long) chunk * rowCount / chunks);
                            int end = (int) ((long) (chunk + 1) * rowCount / chunks);
                            fillMatVecRowsQ4K(data, columns, rowBytes, vector, vectorGroupSums, output, start, end);
                        });
                return;
            } catch (WrongThreadException ignored) {
                // Confined FFM segments cannot be read by worker threads; keep correctness and fall back.
            }
        }
        fillMatVecRowsQ4K(data, columns, rowBytes, vector, vectorGroupSums, output, 0, rowCount);
    }

    private static void fillMatVecRowsQ4K(
            MemorySegment data,
            int columns,
            long rowBytes,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        for (int row = startRow; row < endRow; row++) {
            output[row] = dotRowQ4KWithGroupSums(data, row * rowBytes, columns, vector, vectorGroupSums);
        }
    }

    private static void fillMatVecRowsQ32(
            Q32Matrix matrix,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        float[] blockBiases = matrix.blockBiases();
        for (int row = startRow; row < endRow; row++) {
            output[row] = dotRowQ32Prepared(
                    blocksPerRow,
                    quants,
                    blockScales,
                    blockBiases,
                    row,
                    vector,
                    vectorGroupSums);
        }
    }

    private static void fillMatVecRowsQ4K(
            Q4KMatrix matrix,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] groupScales = matrix.groupScales();
        float[] groupMins = matrix.groupMins();
        for (int row = startRow; row < endRow; row++) {
            output[row] = dotRowQ4KPrepared(
                    blocksPerRow,
                    quants,
                    groupScales,
                    groupMins,
                    row,
                    vector,
                    vectorGroupSums);
        }
    }

    private static void fillMatVecRowsQ5K(
            Q5KMatrix matrix,
            float[] vector,
            float[] vectorGroupSums,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] groupScales = matrix.groupScales();
        float[] groupMins = matrix.groupMins();
        for (int row = startRow; row < endRow; row++) {
            output[row] = dotRowQ5KPrepared(
                    blocksPerRow,
                    quants,
                    groupScales,
                    groupMins,
                    row,
                    vector,
                    vectorGroupSums);
        }
    }

    private static void fillMatVecRowsQ6K(
            Q6KMatrix matrix,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] groupScales = matrix.groupScales();
        for (int row = startRow; row < endRow; row++) {
            output[row] = dotRowQ6KPrepared(blocksPerRow, quants, groupScales, row, vector);
        }
    }

    private static void fillMatVecRowsQ8(
            Q8Matrix matrix,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        int blocksPerRow = matrix.blocksPerRow();
        byte[] quants = matrix.quants();
        float[] blockScales = matrix.blockScales();
        for (int row = startRow; row < endRow; row++) {
            output[row] = dotRowQ8Prepared(blocksPerRow, quants, blockScales, row, vector);
        }
    }

    private static float dotRowQ8Prepared(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            int row,
            float[] vector) {
        float sum = 0.0f;
        for (int block = 0; block < blocksPerRow; block++) {
            int matrixBlock = row * blocksPerRow + block;
            int qBase = matrixBlock * Q8_0_BLOCK_SIZE;
            int vBase = block * Q8_0_BLOCK_SIZE;
            sum += blockScales[matrixBlock] * dotQ8Block(quants, qBase, vector, vBase);
        }
        return sum;
    }

    private static float dotRowQ32Prepared(
            int blocksPerRow,
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
            int row,
            float[] vector,
            float[] vectorGroupSums) {
        float sum = 0.0f;
        for (int block = 0; block < blocksPerRow; block++) {
            int matrixBlock = row * blocksPerRow + block;
            int qBase = matrixBlock * Q4_0_BLOCK_SIZE;
            int vBase = block * Q4_0_BLOCK_SIZE;
            float quantDot = dotQ4Group(quants, qBase, vector, vBase);
            sum += blockScales[matrixBlock] * quantDot
                    + blockBiases[matrixBlock] * vectorGroupSums[block];
        }
        return sum;
    }

    private static float dotRowQ4KPrepared(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int row,
            float[] vector,
            float[] vectorGroupSums) {
        float sum = 0.0f;
        for (int block = 0; block < blocksPerRow; block++) {
            int matrixBlock = row * blocksPerRow + block;
            int quantsOffset = matrixBlock * QK_K;
            int matrixGroupBase = matrixBlock * 8;
            int vectorGroupBase = block * 8;
            int vectorBase = block * QK_K;

            for (int group = 0; group < 8; group++) {
                int matrixGroup = matrixGroupBase + group;
                int vectorGroup = vectorGroupBase + group;
                int qBase = quantsOffset + group * 32;
                int vBase = vectorBase + group * 32;
                float quantDot = dotQ4Group(quants, qBase, vector, vBase);
                sum += groupScales[matrixGroup] * quantDot - groupMins[matrixGroup] * vectorGroupSums[vectorGroup];
            }
        }
        return sum;
    }

    private static float dotRowQ5KPrepared(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            int row,
            float[] vector,
            float[] vectorGroupSums) {
        float sum = 0.0f;
        for (int block = 0; block < blocksPerRow; block++) {
            int matrixBlock = row * blocksPerRow + block;
            int quantsOffset = matrixBlock * QK_K;
            int matrixGroupBase = matrixBlock * 8;
            int vectorGroupBase = block * 8;
            int vectorBase = block * QK_K;

            for (int group = 0; group < 8; group++) {
                int matrixGroup = matrixGroupBase + group;
                int vectorGroup = vectorGroupBase + group;
                int qBase = quantsOffset + group * 32;
                int vBase = vectorBase + group * 32;
                float quantDot = dotQ4Group(quants, qBase, vector, vBase);
                sum += groupScales[matrixGroup] * quantDot - groupMins[matrixGroup] * vectorGroupSums[vectorGroup];
            }
        }
        return sum;
    }

    private static float dotRowQ6KPrepared(
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            int row,
            float[] vector) {
        float sum = 0.0f;
        for (int block = 0; block < blocksPerRow; block++) {
            int matrixBlock = row * blocksPerRow + block;
            int quantsOffset = matrixBlock * QK_K;
            int matrixGroupBase = matrixBlock * 16;
            int vectorBase = block * QK_K;

            for (int group = 0; group < 16; group++) {
                int qBase = quantsOffset + group * 16;
                int vBase = vectorBase + group * 16;
                float quantDot = dotSignedByteBlock(quants, qBase, vector, vBase, 16);
                sum += groupScales[matrixGroupBase + group] * quantDot;
            }
        }
        return sum;
    }

    private static float dotQ4Group(byte[] quants, int qBase, float[] vector, int vBase) {
        if (Q4_DOT_VECTOR_ENABLED) {
            return dotQ4GroupVector(quants, qBase, vector, vBase);
        }
        return dotQ4GroupScalar(quants, qBase, vector, vBase);
    }

    private static float dotQ4GroupVector(byte[] quants, int qBase, float[] vector, int vBase) {
        FloatVector acc = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        for (int i = 0; i < 32; i += Q4_DOT_VECTOR_LANES) {
            ByteVector q = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i);
            FloatVector qf = (FloatVector) q.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector vf = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
            acc = qf.fma(vf, acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static float dotQ4GroupScalar(byte[] quants, int qBase, float[] vector, int vBase) {
        float quantDot = 0.0f;
        for (int i = 0; i < 32; i += 4) {
            quantDot += quants[qBase + i] * vector[vBase + i]
                    + quants[qBase + i + 1] * vector[vBase + i + 1]
                    + quants[qBase + i + 2] * vector[vBase + i + 2]
                    + quants[qBase + i + 3] * vector[vBase + i + 3];
        }
        return quantDot;
    }

    private static float dotQ8Block(byte[] quants, int qBase, float[] vector, int vBase) {
        return dotSignedByteBlock(quants, qBase, vector, vBase, Q8_0_BLOCK_SIZE);
    }

    private static float dotSignedByteBlock(byte[] quants, int qBase, float[] vector, int vBase, int length) {
        if (SIGNED_BYTE_DOT_VECTOR_ENABLED) {
            return dotSignedByteBlockVector(quants, qBase, vector, vBase, length);
        }
        return dotSignedByteBlockScalar(quants, qBase, vector, vBase, length);
    }

    private static float dotSignedByteBlockVector(byte[] quants, int qBase, float[] vector, int vBase, int length) {
        FloatVector acc = FloatVector.zero(Q4_DOT_FLOAT_SPECIES);
        int i = 0;
        int vectorLimit = length - Q4_DOT_VECTOR_LANES;
        for (; i <= vectorLimit; i += Q4_DOT_VECTOR_LANES) {
            ByteVector q = ByteVector.fromArray(Q4_DOT_BYTE_SPECIES, quants, qBase + i);
            FloatVector qf = (FloatVector) q.convertShape(VectorOperators.B2F, Q4_DOT_FLOAT_SPECIES, 0);
            FloatVector vf = FloatVector.fromArray(Q4_DOT_FLOAT_SPECIES, vector, vBase + i);
            acc = qf.fma(vf, acc);
        }
        float quantDot = acc.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            quantDot += quants[qBase + i] * vector[vBase + i];
        }
        return quantDot;
    }

    private static float dotSignedByteBlockScalar(byte[] quants, int qBase, float[] vector, int vBase, int length) {
        float quantDot = 0.0f;
        int i = 0;
        int unrolledLimit = length - 4;
        for (; i <= unrolledLimit; i += 4) {
            quantDot += quants[qBase + i] * vector[vBase + i]
                    + quants[qBase + i + 1] * vector[vBase + i + 1]
                    + quants[qBase + i + 2] * vector[vBase + i + 2]
                    + quants[qBase + i + 3] * vector[vBase + i + 3];
        }
        for (; i < length; i++) {
            quantDot += quants[qBase + i] * vector[vBase + i];
        }
        return quantDot;
    }

    private static float[] q4KVectorGroupSums(float[] vector, int columns) {
        return q4KVectorGroupSums(vector, columns, Q4K_WORK_BUFFER.get());
    }

    private static float[] q4KVectorGroupSums(float[] vector, int columns, Q4KWorkBuffer workBuffer) {
        int blocks = columns / QK_K;
        float[] sums = workBuffer.vectorGroupSums(blocks * 8);
        for (int block = 0; block < blocks; block++) {
            int vectorBase = block * QK_K;
            int groupBase = block * 8;
            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
                int groupIndex = groupBase + (superBlockOffset / 64) * 2;
                float first = 0.0f;
                float second = 0.0f;
                for (int i = 0; i < 32; i++) {
                    first += vector[vectorBase + superBlockOffset + i];
                    second += vector[vectorBase + superBlockOffset + 32 + i];
                }
                sums[groupIndex] = first;
                sums[groupIndex + 1] = second;
            }
        }
        return sums;
    }

    private static float[] vector32GroupSums(float[] vector, int columns, Q4KWorkBuffer workBuffer) {
        int groups = columns / Q4_0_BLOCK_SIZE;
        float[] sums = workBuffer.vectorGroupSums(groups);
        for (int group = 0; group < groups; group++) {
            int vectorBase = group * Q4_0_BLOCK_SIZE;
            float sum = 0.0f;
            for (int i = 0; i < Q4_0_BLOCK_SIZE; i++) {
                sum += vector[vectorBase + i];
            }
            sums[group] = sum;
        }
        return sums;
    }

    private static void fillMatVecRows(
            MemorySegment data,
            int typeId,
            int columns,
            long rowBytes,
            float[] vector,
            float[] output,
            int startRow,
            int endRow) {
        for (int row = startRow; row < endRow; row++) {
            output[row] = dotRow(data, row * rowBytes, typeId, columns, vector, 0);
        }
    }

    private static int checkedColumns(GGUFTensorInfo tensor, int vectorLength) {
        long columns = matrixColumns(tensor);
        if (columns > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor row is too wide for Java arrays: " + columns);
        }
        if (vectorLength < columns) {
            throw new IllegalArgumentException("Vector length " + vectorLength + " is smaller than columns " + columns);
        }
        int cols = (int) columns;
        rowByteSize(tensor, cols);
        return cols;
    }

    private static void checkRow(GGUFTensorInfo tensor, long row) {
        long rows = matrixRows(tensor);
        if (row < 0 || row >= rows) {
            throw new IllegalArgumentException("Row " + row + " is outside tensor row count " + rows);
        }
    }

    private static int checkedRows(GGUFTensorInfo tensor) {
        long rows = matrixRows(tensor);
        if (rows > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor has too many rows for Java arrays: " + rows);
        }
        return (int) rows;
    }

    private static long rowByteSize(GGUFTensorInfo tensor, int columns) {
        return rowByteSize(tensor.typeId(), columns);
    }

    private static long rowByteSize(int typeId, int columns) {
        GgmlType type = GgmlType.fromId(typeId);
        return type.bytesFor(columns);
    }

    static boolean supportsQ32PreparedType(int typeId) {
        return typeId == GgmlType.Q4_0.id
                || typeId == GgmlType.Q4_1.id
                || typeId == GgmlType.Q5_0.id
                || typeId == GgmlType.Q5_1.id;
    }

    private static int q32BlockBytes(int typeId) {
        if (typeId == GgmlType.Q4_0.id) {
            return Q4_0_BLOCK_BYTES;
        }
        if (typeId == GgmlType.Q4_1.id) {
            return Q4_1_BLOCK_BYTES;
        }
        if (typeId == GgmlType.Q5_0.id) {
            return Q5_0_BLOCK_BYTES;
        }
        if (typeId == GgmlType.Q5_1.id) {
            return Q5_1_BLOCK_BYTES;
        }
        throw new IllegalArgumentException("Unsupported Q32 prepared type id: " + typeId);
    }

    private static void unpackQ4_0Prepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        for (int i = 0; i < 16; i++) {
            int quant = source[sourceOffset + i] & 0xFF;
            quants[qBase + i] = (byte) ((quant & 0x0F) - 8);
            quants[qBase + 16 + i] = (byte) ((quant >>> 4) - 8);
        }
    }

    private static void unpackQ4_1Prepared(byte[] source, int sourceOffset, byte[] quants, int qBase) {
        for (int i = 0; i < 16; i++) {
            int quant = source[sourceOffset + i] & 0xFF;
            quants[qBase + i] = (byte) (quant & 0x0F);
            quants[qBase + 16 + i] = (byte) (quant >>> 4);
        }
    }

    private static void unpackQ5_0Prepared(byte[] source, int sourceOffset, int highBits, byte[] quants, int qBase) {
        for (int i = 0; i < 16; i++) {
            int quant = source[sourceOffset + i] & 0xFF;
            int low = (quant & 0x0F) | (((highBits >>> i) & 1) << 4);
            int high = ((quant >>> 4) & 0x0F) | (((highBits >>> (i + 16)) & 1) << 4);
            quants[qBase + i] = (byte) (low - 16);
            quants[qBase + 16 + i] = (byte) (high - 16);
        }
    }

    private static void unpackQ5_1Prepared(byte[] source, int sourceOffset, int highBits, byte[] quants, int qBase) {
        for (int i = 0; i < 16; i++) {
            int quant = source[sourceOffset + i] & 0xFF;
            int low = (quant & 0x0F) | (((highBits >>> i) & 1) << 4);
            int high = ((quant >>> 4) & 0x0F) | (((highBits >>> (i + 16)) & 1) << 4);
            quants[qBase + i] = (byte) low;
            quants[qBase + 16 + i] = (byte) high;
        }
    }

    private static ScaleMin scaleMinK4(MemorySegment segment, long scalesOffset, int index) {
        int scale;
        int min;
        if (index < 4) {
            scale = u8(segment, scalesOffset + index) & 63;
            min = u8(segment, scalesOffset + index + 4) & 63;
        } else {
            scale = (u8(segment, scalesOffset + index + 4) & 0x0F)
                    | ((u8(segment, scalesOffset + index - 4) >>> 6) << 4);
            min = (u8(segment, scalesOffset + index + 4) >>> 4)
                    | ((u8(segment, scalesOffset + index) >>> 6) << 4);
        }
        return new ScaleMin(scale, min);
    }

    private static int u8(MemorySegment segment, long offset) {
        return segment.get(ValueLayout.JAVA_BYTE, offset) & 0xFF;
    }

    private static ScaleMin scaleMinK4(byte[] data, long scalesOffset, int index) {
        int scale;
        int min;
        if (index < 4) {
            scale = u8(data, scalesOffset + index) & 63;
            min = u8(data, scalesOffset + index + 4) & 63;
        } else {
            scale = (u8(data, scalesOffset + index + 4) & 0x0F)
                    | ((u8(data, scalesOffset + index - 4) >>> 6) << 4);
            min = (u8(data, scalesOffset + index + 4) >>> 4)
                    | ((u8(data, scalesOffset + index) >>> 6) << 4);
        }
        return new ScaleMin(scale, min);
    }

    private static int u8(byte[] data, long offset) {
        return data[(int) offset] & 0xFF;
    }

    private static short leShort(byte[] data, long offset) {
        int index = (int) offset;
        return (short) ((data[index] & 0xFF) | ((data[index + 1] & 0xFF) << 8));
    }

    private static int leInt(byte[] data, long offset) {
        int index = (int) offset;
        return (data[index] & 0xFF)
                | ((data[index + 1] & 0xFF) << 8)
                | ((data[index + 2] & 0xFF) << 16)
                | ((data[index + 3] & 0xFF) << 24);
    }

    private static float f16ToF32(short bits) {
        int half = bits & 0xFFFF;
        int sign = (half >>> 15) & 0x1;
        int exponent = (half >>> 10) & 0x1F;
        int mantissa = half & 0x03FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            float value = (float) Math.scalb(mantissa / 1024.0f, -14);
            return sign == 0 ? value : -value;
        }
        if (exponent == 31) {
            if (mantissa == 0) {
                return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            }
            return Float.NaN;
        }

        int floatBits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
        return Float.intBitsToFloat(floatBits);
    }

    private record ScaleMin(int scale, int min) {
    }

    private static final class Q32ModelCache {
        private final LinkedHashMap<Q32MatrixKey, Q32Matrix> matrices = new LinkedHashMap<>(16, 0.75f, true);
        private long bytes;

        Q32Matrix get(Q32MatrixKey key) {
            return matrices.get(key);
        }

        void put(Q32MatrixKey key, Q32Matrix matrix, long maxBytes) {
            Q32Matrix previous = matrices.put(key, matrix);
            if (previous != null) {
                bytes -= previous.estimatedBytes();
            }
            bytes += matrix.estimatedBytes();
            evictTo(maxBytes);
        }

        int size() {
            return matrices.size();
        }

        long bytes() {
            return bytes;
        }

        private void evictTo(long maxBytes) {
            Iterator<Map.Entry<Q32MatrixKey, Q32Matrix>> iterator = matrices.entrySet().iterator();
            while (bytes > maxBytes && iterator.hasNext()) {
                Map.Entry<Q32MatrixKey, Q32Matrix> eldest = iterator.next();
                bytes -= eldest.getValue().estimatedBytes();
                iterator.remove();
            }
        }
    }

    private static final class Q4KModelCache {
        private final LinkedHashMap<Q4KMatrixKey, Q4KMatrix> matrices = new LinkedHashMap<>(16, 0.75f, true);
        private long bytes;

        Q4KMatrix get(Q4KMatrixKey key) {
            return matrices.get(key);
        }

        void put(Q4KMatrixKey key, Q4KMatrix matrix, long maxBytes) {
            Q4KMatrix previous = matrices.put(key, matrix);
            if (previous != null) {
                bytes -= previous.estimatedBytes();
            }
            bytes += matrix.estimatedBytes();
            evictTo(maxBytes);
        }

        int size() {
            return matrices.size();
        }

        long bytes() {
            return bytes;
        }

        private void evictTo(long maxBytes) {
            Iterator<Map.Entry<Q4KMatrixKey, Q4KMatrix>> iterator = matrices.entrySet().iterator();
            while (bytes > maxBytes && iterator.hasNext()) {
                Map.Entry<Q4KMatrixKey, Q4KMatrix> eldest = iterator.next();
                bytes -= eldest.getValue().estimatedBytes();
                iterator.remove();
            }
        }
    }

    private static final class Q5KModelCache {
        private final LinkedHashMap<Q5KMatrixKey, Q5KMatrix> matrices = new LinkedHashMap<>(16, 0.75f, true);
        private long bytes;

        Q5KMatrix get(Q5KMatrixKey key) {
            return matrices.get(key);
        }

        void put(Q5KMatrixKey key, Q5KMatrix matrix, long maxBytes) {
            Q5KMatrix previous = matrices.put(key, matrix);
            if (previous != null) {
                bytes -= previous.estimatedBytes();
            }
            bytes += matrix.estimatedBytes();
            evictTo(maxBytes);
        }

        int size() {
            return matrices.size();
        }

        long bytes() {
            return bytes;
        }

        private void evictTo(long maxBytes) {
            Iterator<Map.Entry<Q5KMatrixKey, Q5KMatrix>> iterator = matrices.entrySet().iterator();
            while (bytes > maxBytes && iterator.hasNext()) {
                Map.Entry<Q5KMatrixKey, Q5KMatrix> eldest = iterator.next();
                bytes -= eldest.getValue().estimatedBytes();
                iterator.remove();
            }
        }
    }

    private static final class Q6KModelCache {
        private final LinkedHashMap<Q6KMatrixKey, Q6KMatrix> matrices = new LinkedHashMap<>(16, 0.75f, true);
        private long bytes;

        Q6KMatrix get(Q6KMatrixKey key) {
            return matrices.get(key);
        }

        void put(Q6KMatrixKey key, Q6KMatrix matrix, long maxBytes) {
            Q6KMatrix previous = matrices.put(key, matrix);
            if (previous != null) {
                bytes -= previous.estimatedBytes();
            }
            bytes += matrix.estimatedBytes();
            evictTo(maxBytes);
        }

        int size() {
            return matrices.size();
        }

        long bytes() {
            return bytes;
        }

        private void evictTo(long maxBytes) {
            Iterator<Map.Entry<Q6KMatrixKey, Q6KMatrix>> iterator = matrices.entrySet().iterator();
            while (bytes > maxBytes && iterator.hasNext()) {
                Map.Entry<Q6KMatrixKey, Q6KMatrix> eldest = iterator.next();
                bytes -= eldest.getValue().estimatedBytes();
                iterator.remove();
            }
        }
    }

    private static final class Q8ModelCache {
        private final LinkedHashMap<Q8MatrixKey, Q8Matrix> matrices = new LinkedHashMap<>(16, 0.75f, true);
        private long bytes;

        Q8Matrix get(Q8MatrixKey key) {
            return matrices.get(key);
        }

        void put(Q8MatrixKey key, Q8Matrix matrix, long maxBytes) {
            Q8Matrix previous = matrices.put(key, matrix);
            if (previous != null) {
                bytes -= previous.estimatedBytes();
            }
            bytes += matrix.estimatedBytes();
            evictTo(maxBytes);
        }

        int size() {
            return matrices.size();
        }

        long bytes() {
            return bytes;
        }

        private void evictTo(long maxBytes) {
            Iterator<Map.Entry<Q8MatrixKey, Q8Matrix>> iterator = matrices.entrySet().iterator();
            while (bytes > maxBytes && iterator.hasNext()) {
                Map.Entry<Q8MatrixKey, Q8Matrix> eldest = iterator.next();
                bytes -= eldest.getValue().estimatedBytes();
                iterator.remove();
            }
        }
    }

    private record Q4KMatrixKey(
            String name,
            int typeId,
            long offset,
            long sizeInBytes,
            long columns,
            long rows) {
    }

    private record Q32MatrixKey(
            String name,
            int typeId,
            long offset,
            long sizeInBytes,
            long columns,
            long rows) {
    }

    private record Q5KMatrixKey(
            String name,
            int typeId,
            long offset,
            long sizeInBytes,
            long columns,
            long rows) {
    }

    private record Q6KMatrixKey(
            String name,
            int typeId,
            long offset,
            long sizeInBytes,
            long columns,
            long rows) {
    }

    private record Q8MatrixKey(
            String name,
            int typeId,
            long offset,
            long sizeInBytes,
            long columns,
            long rows) {
    }
}

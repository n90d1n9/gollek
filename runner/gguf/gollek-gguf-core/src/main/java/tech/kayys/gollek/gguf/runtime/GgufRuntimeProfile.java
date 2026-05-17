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
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Buildable Java-native GGUF readiness profile.
 *
 * <p>This is intentionally stricter than the llama.cpp fallback: Java can parse
 * and inspect many models today, but generation should only be enabled after we
 * can prove the tensor layout and decoder tensor set are complete.</p>
 */
public record GgufRuntimeProfile(
        String architecture,
        int ggufVersion,
        int metadataCount,
        int tensorCount,
        long modelBytes,
        long tensorBytes,
        long knownTypeTensorBytes,
        int requiredDecoderTensorCount,
        int presentDecoderTensorCount,
        List<String> missingDecoderTensorExamples,
        List<TensorTypeSummary> tensorTypes,
        List<String> unknownTensorTypeIds,
        List<String> unsupportedRowDotTypeIds,
        long loadMillis
) {
    private static final Set<Integer> JAVA_KNOWN_LAYOUT_TYPE_IDS = Set.of(
            GgmlType.F32.id,
            GgmlType.F16.id,
            GgmlType.BF16.id,
            GgmlType.Q4_0.id,
            GgmlType.Q4_1.id,
            GgmlType.Q5_0.id,
            GgmlType.Q5_1.id,
            GgmlType.Q8_0.id,
            GgmlType.Q2_K.id,
            GgmlType.Q3_K.id,
            GgmlType.Q4_K.id,
            GgmlType.Q5_K.id,
            GgmlType.Q6_K.id
    );

    public static GgufRuntimeProfile load(Path modelPath) throws IOException {
        long startNanos = System.nanoTime();
        try (Arena arena = Arena.ofConfined(); GGUFReader reader = new GGUFReader(modelPath, arena)) {
            GGUFModel model = new GGUFParser().parse(reader.segment(), null);
            long loadMillis = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
            return fromModel(model, Files.size(modelPath), loadMillis);
        }
    }

    public static GgufRuntimeProfile fromModel(GGUFModel model, long modelBytes, long loadMillis) {
        Map<String, Object> metadata = model.metadata();
        List<GGUFTensorInfo> tensors = model.tensors();
        String architecture = String.valueOf(metadata.getOrDefault("general.architecture", "unknown"))
                .toLowerCase(Locale.ROOT);

        Map<String, TypeAccumulator> byType = new HashMap<>();
        List<String> unknownTypeIds = new ArrayList<>();
        List<String> unsupportedRowDotTypeIds = new ArrayList<>();
        long tensorBytes = 0;
        long knownTypeBytes = 0;

        for (GGUFTensorInfo tensor : tensors) {
            long bytes = Math.max(0, tensor.sizeInBytes());
            tensorBytes += bytes;

            String label = "TYPE_" + tensor.typeId();
            boolean knownForJava = JAVA_KNOWN_LAYOUT_TYPE_IDS.contains(tensor.typeId());
            boolean rowDotSupported = GgufTensorOps.supportsRowDotType(tensor.typeId());
            try {
                label = GgmlType.fromId(tensor.typeId()).label;
            } catch (IllegalArgumentException ignored) {
                unknownTypeIds.add(String.valueOf(tensor.typeId()));
                knownForJava = false;
                rowDotSupported = false;
            }

            byType.computeIfAbsent(label, ignored -> new TypeAccumulator())
                    .add(tensor, bytes, knownForJava);
            if (knownForJava) {
                knownTypeBytes += bytes;
            }
            if (!rowDotSupported) {
                unsupportedRowDotTypeIds.add(String.valueOf(tensor.typeId()));
            }
        }

        List<TensorTypeSummary> summaries = byType.entrySet().stream()
                .map(entry -> entry.getValue().summary(entry.getKey()))
                .sorted(Comparator.comparingLong(TensorTypeSummary::bytes).reversed())
                .toList();

        Set<String> tensorNames = new HashSet<>();
        for (GGUFTensorInfo tensor : tensors) {
            tensorNames.add(tensor.name());
        }

        List<String> required = requiredDecoderTensorNames(metadata, architecture);
        int presentRequired = 0;
        List<String> missingExamples = new ArrayList<>();
        for (String name : required) {
            if (tensorNames.contains(name)) {
                presentRequired++;
            } else if (missingExamples.size() < 8) {
                missingExamples.add(name);
            }
        }

        return new GgufRuntimeProfile(
                architecture,
                model.version(),
                metadata.size(),
                tensors.size(),
                modelBytes,
                tensorBytes,
                knownTypeBytes,
                required.size(),
                presentRequired,
                List.copyOf(missingExamples),
                List.copyOf(summaries),
                unknownTypeIds.stream().distinct().sorted().toList(),
                unsupportedRowDotTypeIds.stream().distinct().sorted().toList(),
                loadMillis);
    }

    public double knownTensorTypeRatio() {
        if (tensorBytes <= 0) {
            return 1.0d;
        }
        return Math.min(1.0d, Math.max(0.0d, knownTypeTensorBytes / (double) tensorBytes));
    }

    public double decoderTensorRatio() {
        if (requiredDecoderTensorCount <= 0) {
            return 0.0d;
        }
        return presentDecoderTensorCount / (double) requiredDecoderTensorCount;
    }

    public boolean decoderTensorSetComplete() {
        return requiredDecoderTensorCount > 0
                && presentDecoderTensorCount == requiredDecoderTensorCount
                && unknownTensorTypeIds.isEmpty();
    }

    public boolean rowDotPrimitivesReady() {
        return decoderTensorSetComplete() && unsupportedRowDotTypeIds.isEmpty();
    }

    public String javaStatus() {
        if (!unknownTensorTypeIds.isEmpty()) {
            return "loader-ready; unknown-tensor-types=" + String.join(",", unknownTensorTypeIds)
                    + "; generation-disabled";
        }
        if (requiredDecoderTensorCount == 0) {
            return "loader-ready; decoder-shape-unknown; generation-disabled";
        }
        if (presentDecoderTensorCount != requiredDecoderTensorCount) {
            return "loader-ready; decoder-tensors-missing="
                    + (requiredDecoderTensorCount - presentDecoderTensorCount)
                    + "; generation-disabled";
        }
        if (!unsupportedRowDotTypeIds.isEmpty()) {
            return "loader-ready; decoder-tensors-ready; row-dot-unsupported-types="
                    + String.join(",", unsupportedRowDotTypeIds)
                    + "; generation-disabled";
        }
        return "loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled";
    }

    public String compactTypeSummary(int limit) {
        if (tensorTypes.isEmpty() || limit <= 0) {
            return "none";
        }
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < Math.min(limit, tensorTypes.size()); i++) {
            TensorTypeSummary summary = tensorTypes.get(i);
            parts.add(summary.label() + ":" + summary.count());
        }
        if (tensorTypes.size() > limit) {
            parts.add("+" + (tensorTypes.size() - limit));
        }
        return String.join(", ", parts);
    }

    private static List<String> requiredDecoderTensorNames(Map<String, Object> metadata, String architecture) {
        int blockCount = metadataInt(metadata,
                architecture + ".block_count",
                "llama.block_count",
                "gemma.block_count",
                "gemma2.block_count",
                "gemma3.block_count",
                "gemma4.block_count",
                "qwen2.block_count",
                "mistral.block_count",
                "phi3.block_count",
                "general.block_count");
        if (blockCount <= 0) {
            return List.of();
        }

        List<String> required = new ArrayList<>(2 + blockCount * 9);
        required.add("token_embd.weight");
        required.add("output_norm.weight");

        for (int layer = 0; layer < blockCount; layer++) {
            String prefix = "blk." + layer + ".";
            required.add(prefix + "attn_norm.weight");
            required.add(prefix + "attn_q.weight");
            required.add(prefix + "attn_k.weight");
            required.add(prefix + "attn_v.weight");
            required.add(prefix + "attn_output.weight");
            required.add(prefix + "ffn_norm.weight");
            required.add(prefix + "ffn_gate.weight");
            required.add(prefix + "ffn_up.weight");
            required.add(prefix + "ffn_down.weight");
        }
        return required;
    }

    private static int metadataInt(Map<String, Object> metadata, String... keys) {
        for (String key : keys) {
            Object value = metadata.get(key);
            if (value instanceof Number number) {
                return number.intValue();
            }
        }
        return 0;
    }

    public record TensorTypeSummary(
            String label,
            int typeId,
            int count,
            long bytes,
            boolean knownForJava
    ) {
    }

    private static final class TypeAccumulator {
        private int typeId = -1;
        private int count;
        private long bytes;
        private boolean knownForJava;

        private void add(GGUFTensorInfo tensor, long tensorBytes, boolean tensorKnownForJava) {
            if (typeId < 0) {
                typeId = tensor.typeId();
            }
            count++;
            bytes += tensorBytes;
            knownForJava |= tensorKnownForJava;
        }

        private TensorTypeSummary summary(String label) {
            return new TensorTypeSummary(label, typeId, count, bytes, knownForJava);
        }
    }
}

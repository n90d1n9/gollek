/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Header-only contract check for Gemma 4 unified decoder and projector tensors.
 */
final class Gemma4UnifiedSafetensorPreflight {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final long MAX_HEADER_BYTES = 100L * 1024L * 1024L;
    private static final int MAX_REPORTED_PROBLEMS = 12;

    private Gemma4UnifiedSafetensorPreflight() {
    }

    record Result(boolean allowed, List<String> messages, ProjectorSummary projectors) {
        static Result pass() {
            return pass(ProjectorSummary.empty());
        }

        static Result pass(ProjectorSummary projectors) {
            return new Result(true, List.of(), projectors);
        }

        static Result invalid(List<String> messages) {
            return invalid(messages, ProjectorSummary.empty());
        }

        static Result invalid(List<String> messages, ProjectorSummary projectors) {
            return new Result(false, List.copyOf(messages), projectors);
        }
    }

    static Result validate(Path modelPath, ModelConfig config, String modelLabel) {
        return inspect(modelPath, config, modelLabel).text();
    }

    static Inspection inspect(Path modelPath, ModelConfig config, String modelLabel) {
        if (modelPath == null || config == null) {
            return Inspection.pass(ProjectorSummary.empty());
        }
        String label = modelLabel == null || modelLabel.isBlank()
                ? modelPath.getFileName().toString()
                : modelLabel;
        try {
            List<Path> files = safetensorFiles(modelPath);
            if (files.isEmpty()) {
                return Inspection.invalid(
                        invalidMessages(label, List.of("no .safetensors file or model.safetensors.index.json was found")),
                        ProjectorSummary.empty());
            }

            Map<String, TensorMeta> tensors = loadTensorHeaders(files);
            ProjectorSummary projectors = ProjectorSummary.from(tensors);
            if (tensors.isEmpty()) {
                return Inspection.invalid(
                        invalidMessages(label, List.of("SafeTensors headers did not contain any tensor entries")),
                        projectors);
            }

            List<String> problems = new ArrayList<>();
            validateTextTowerContract(config, tensors, problems);
            if (!problems.isEmpty()) {
                return Inspection.invalid(invalidMessages(label, problems), projectors);
            }
            return Inspection.pass(projectors);
        } catch (IOException | RuntimeException e) {
            return Inspection.invalid(
                    invalidMessages(label, List.of("could not inspect SafeTensors headers: " + e.getMessage())),
                    ProjectorSummary.empty());
        }
    }

    private static List<String> invalidMessages(String modelLabel, List<String> problems) {
        List<String> messages = new ArrayList<>();
        messages.add("Error: Gemma 4 unified safetensor text preflight failed.");
        messages.add("Checkpoint " + modelLabel
                + " is recognized as gemma4_unified text, but required text decoder SafeTensors headers are missing or incompatible.");
        for (String problem : problems) {
            messages.add("  - " + problem);
        }
        messages.add("This check reads only SafeTensors headers; no 12B weight payload was loaded.");
        return messages;
    }

    private static void validateTextTowerContract(
            ModelConfig config,
            Map<String, TensorMeta> tensors,
            List<String> problems) {
        int hiddenSize = config.hiddenSize();
        int intermediateSize = config.intermediateSize();
        int vocabSize = config.vocabSize();
        int layers = config.numHiddenLayers();
        if (hiddenSize <= 0 || intermediateSize <= 0 || vocabSize <= 0 || layers <= 0) {
            addProblem(problems, "invalid Gemma 4 text config dimensions: hidden=%d, intermediate=%d, vocab=%d, layers=%d"
                    .formatted(hiddenSize, intermediateSize, vocabSize, layers));
            return;
        }

        TensorMeta embed = requireTensor(
                tensors,
                problems,
                textCandidates("embed_tokens.weight"));
        if (embed != null) {
            requireShape(problems, embed, vocabSize, hiddenSize);
        }

        TensorMeta finalNorm = requireTensor(
                tensors,
                problems,
                textCandidates("norm.weight"));
        if (finalNorm != null) {
            requireShape(problems, finalNorm, hiddenSize);
        }

        TensorMeta lmHead = findTensor(
                tensors,
                List.of("lm_head.weight", "model.lm_head.weight", "model.language_model.lm_head.weight"));
        if (lmHead == null && !config.tieWordEmbeddings()) {
            addProblem(problems, "missing tensor: lm_head.weight for untied output embeddings");
        } else if (lmHead != null) {
            requireShape(problems, lmHead, vocabSize, hiddenSize);
        }

        for (int layer = 0; layer < layers; layer++) {
            validateLayer(config, tensors, problems, layer, hiddenSize, intermediateSize);
            if (problems.size() > MAX_REPORTED_PROBLEMS) {
                break;
            }
        }
    }

    private static void validateLayer(
            ModelConfig config,
            Map<String, TensorMeta> tensors,
            List<String> problems,
            int layer,
            int hiddenSize,
            int intermediateSize) {
        String prefix = "layers.%d.".formatted(layer);
        requireVector(tensors, problems, prefix + "layer_scalar", 1);
        requireVector(tensors, problems, prefix + "input_layernorm.weight", hiddenSize);
        requireVector(tensors, problems, prefix + "post_attention_layernorm.weight", hiddenSize);
        requireVector(tensors, problems, prefix + "pre_feedforward_layernorm.weight", hiddenSize);
        requireVector(tensors, problems, prefix + "post_feedforward_layernorm.weight", hiddenSize);

        TensorMeta q = requireMatrixInput(tensors, problems, prefix + "self_attn.q_proj.weight", hiddenSize);
        TensorMeta k = requireMatrixInput(tensors, problems, prefix + "self_attn.k_proj.weight", hiddenSize);
        TensorMeta o = requireMatrixInput(tensors, problems, prefix + "self_attn.o_proj.weight", -1);

        boolean valueProjectionMayBeTiedToKey =
                config.usesAlternativeAttentionForLayer(layer) || config.usesSharedKvCache(layer);
        TensorMeta v = findTensor(tensors, textCandidates(prefix + "self_attn.v_proj.weight"));
        if (v == null && !valueProjectionMayBeTiedToKey) {
            addProblem(problems, "missing tensor: " + preferredName(prefix + "self_attn.v_proj.weight"));
        } else if (v != null) {
            requireRank(problems, v, 2);
            requireDim(problems, v, 1, hiddenSize);
            if (k != null && k.rank() == 2) {
                requireDim(problems, v, 0, k.dim(0));
            }
        }

        if (o != null && q != null && q.rank() == 2) {
            requireShape(problems, o, hiddenSize, q.dim(0));
        }

        TensorMeta qNorm = requireTensor(tensors, problems, textCandidates(prefix + "self_attn.q_norm.weight"));
        TensorMeta kNorm = requireTensor(tensors, problems, textCandidates(prefix + "self_attn.k_norm.weight"));
        validateAttentionNorm(problems, qNorm, q);
        validateAttentionNorm(problems, kNorm, k);

        TensorMeta gate = requireMatrixInput(tensors, problems, prefix + "mlp.gate_proj.weight", hiddenSize);
        TensorMeta up = requireMatrixInput(tensors, problems, prefix + "mlp.up_proj.weight", hiddenSize);
        TensorMeta down = requireMatrixInput(tensors, problems, prefix + "mlp.down_proj.weight", -1);
        if (gate != null) {
            requireDim(problems, gate, 0, intermediateSize);
        }
        if (up != null) {
            requireDim(problems, up, 0, intermediateSize);
        }
        if (down != null) {
            requireShape(problems, down, hiddenSize, intermediateSize);
        }
    }

    private static TensorMeta requireVector(
            Map<String, TensorMeta> tensors,
            List<String> problems,
            String suffix,
            long expectedDim) {
        TensorMeta tensor = requireTensor(tensors, problems, textCandidates(suffix));
        if (tensor != null) {
            requireShape(problems, tensor, expectedDim);
        }
        return tensor;
    }

    private static TensorMeta requireMatrixInput(
            Map<String, TensorMeta> tensors,
            List<String> problems,
            String suffix,
            long expectedInputDim) {
        TensorMeta tensor = requireTensor(tensors, problems, textCandidates(suffix));
        if (tensor != null) {
            requireRank(problems, tensor, 2);
            if (expectedInputDim > 0) {
                requireDim(problems, tensor, 1, expectedInputDim);
            }
        }
        return tensor;
    }

    private static void validateAttentionNorm(List<String> problems, TensorMeta norm, TensorMeta projection) {
        if (norm == null) {
            return;
        }
        requireRank(problems, norm, 1);
        if (projection == null || projection.rank() != 2 || norm.rank() != 1 || norm.dim(0) <= 0) {
            return;
        }
        long projectionRows = projection.dim(0);
        long normDim = norm.dim(0);
        if (projectionRows % normDim != 0) {
            addProblem(problems, "%s shape %s is incompatible with %s rows=%d"
                    .formatted(norm.name(), norm.shapeString(), projection.name(), projectionRows));
        }
    }

    private static TensorMeta requireTensor(
            Map<String, TensorMeta> tensors,
            List<String> problems,
            List<String> candidates) {
        TensorMeta tensor = findTensor(tensors, candidates);
        if (tensor == null) {
            addProblem(problems, "missing tensor: one of " + String.join(", ", candidates));
            return null;
        }
        if (!isSupportedFloatingDtype(tensor.dtype())) {
            addProblem(problems, "unsupported dtype for " + tensor.name() + ": " + tensor.dtype());
        }
        return tensor;
    }

    private static TensorMeta findTensor(Map<String, TensorMeta> tensors, List<String> candidates) {
        for (String candidate : candidates) {
            TensorMeta tensor = tensors.get(candidate);
            if (tensor != null) {
                return tensor;
            }
        }
        return null;
    }

    private static void requireShape(List<String> problems, TensorMeta tensor, long... expected) {
        requireRank(problems, tensor, expected.length);
        if (tensor.rank() != expected.length) {
            return;
        }
        for (int i = 0; i < expected.length; i++) {
            requireDim(problems, tensor, i, expected[i]);
        }
    }

    private static void requireRank(List<String> problems, TensorMeta tensor, int expectedRank) {
        if (tensor.rank() != expectedRank) {
            addProblem(problems, "%s shape %s must have rank %d"
                    .formatted(tensor.name(), tensor.shapeString(), expectedRank));
        }
    }

    private static void requireDim(List<String> problems, TensorMeta tensor, int dim, long expected) {
        if (dim < 0 || dim >= tensor.rank() || tensor.dim(dim) == expected) {
            return;
        }
        addProblem(problems, "%s shape %s has dim[%d]=%d, expected %d"
                .formatted(tensor.name(), tensor.shapeString(), dim, tensor.dim(dim), expected));
    }

    private static void addProblem(List<String> problems, String problem) {
        if (problems.size() < MAX_REPORTED_PROBLEMS) {
            problems.add(problem);
        } else if (problems.size() == MAX_REPORTED_PROBLEMS) {
            problems.add("additional tensor contract problems omitted");
        }
    }

    private static boolean isSupportedFloatingDtype(String dtype) {
        String normalized = dtype == null ? "" : dtype.trim().toUpperCase(Locale.ROOT);
        return normalized.equals("BF16")
                || normalized.equals("F16")
                || normalized.equals("F32")
                || normalized.equals("FLOAT16")
                || normalized.equals("FLOAT32")
                || normalized.equals("BFLOAT16");
    }

    private static List<String> textCandidates(String suffix) {
        return List.of(
                "model.language_model." + suffix,
                "language_model." + suffix,
                "model." + suffix,
                suffix);
    }

    private static String preferredName(String suffix) {
        return "model.language_model." + suffix;
    }

    private static List<Path> safetensorFiles(Path modelPath) throws IOException {
        if (Files.isRegularFile(modelPath) && isSafetensorFile(modelPath)) {
            return List.of(modelPath.toAbsolutePath().normalize());
        }
        Path modelDir = Files.isDirectory(modelPath) ? modelPath : modelPath.getParent();
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            return List.of();
        }

        Optional<List<Path>> indexed = indexedSafetensorFiles(modelDir);
        if (indexed.isPresent()) {
            return indexed.get();
        }

        Path modelSafetensors = modelDir.resolve("model.safetensors");
        if (Files.isRegularFile(modelSafetensors)) {
            return List.of(modelSafetensors.toAbsolutePath().normalize());
        }

        try (var stream = Files.list(modelDir)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(Gemma4UnifiedSafetensorPreflight::isSafetensorFile)
                    .map(path -> path.toAbsolutePath().normalize())
                    .sorted(Comparator.comparing(Path::toString))
                    .toList();
        }
    }

    private static Optional<List<Path>> indexedSafetensorFiles(Path modelDir) throws IOException {
        Path index = modelDir.resolve("model.safetensors.index.json");
        if (!Files.isRegularFile(index)) {
            return Optional.empty();
        }
        JsonNode weightMap = OBJECT_MAPPER.readTree(index.toFile()).path("weight_map");
        if (!weightMap.isObject()) {
            return Optional.of(List.of());
        }
        Set<Path> shards = new LinkedHashSet<>();
        for (JsonNode shardNode : weightMap) {
            String shardName = shardNode.asText("");
            if (!shardName.isBlank()) {
                Path shard = modelDir.resolve(shardName).toAbsolutePath().normalize();
                if (Files.isRegularFile(shard) && isSafetensorFile(shard)) {
                    shards.add(shard);
                }
            }
        }
        return Optional.of(shards.stream()
                .sorted(Comparator.comparing(Path::toString))
                .toList());
    }

    private static boolean isSafetensorFile(Path path) {
        String name = path == null || path.getFileName() == null
                ? ""
                : path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".safetensors") || name.endsWith(".safetensor");
    }

    private static Map<String, TensorMeta> loadTensorHeaders(List<Path> files) throws IOException {
        Map<String, TensorMeta> tensors = new LinkedHashMap<>();
        for (Path file : files) {
            tensors.putAll(loadTensorHeader(file));
        }
        return tensors;
    }

    private static Map<String, TensorMeta> loadTensorHeader(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer headerLengthBuffer = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, headerLengthBuffer, 0);
            headerLengthBuffer.flip();
            long headerLength = headerLengthBuffer.getLong();
            if (headerLength <= 0 || headerLength > MAX_HEADER_BYTES || headerLength > Integer.MAX_VALUE) {
                throw new IOException("invalid SafeTensors header length " + headerLength + " for " + file);
            }

            ByteBuffer headerBuffer = ByteBuffer.allocate((int) headerLength);
            readFully(channel, headerBuffer, Long.BYTES);
            headerBuffer.flip();
            String headerJson = StandardCharsets.UTF_8.decode(headerBuffer).toString();
            JsonNode root = OBJECT_MAPPER.readTree(headerJson);
            Map<String, TensorMeta> tensors = new LinkedHashMap<>();
            for (Map.Entry<String, JsonNode> field : root.properties()) {
                if ("__metadata__".equals(field.getKey())) {
                    continue;
                }
                JsonNode node = field.getValue();
                if (!node.isObject()) {
                    continue;
                }
                tensors.put(field.getKey(), new TensorMeta(
                        field.getKey(),
                        node.path("dtype").asText(""),
                        shape(node.path("shape"))));
            }
            return tensors;
        }
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        long offset = position;
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, offset);
            if (read < 0) {
                throw new IOException("unexpected end of file while reading SafeTensors header");
            }
            offset += read;
        }
    }

    private static long[] shape(JsonNode shapeNode) {
        if (!shapeNode.isArray()) {
            return new long[0];
        }
        long[] shape = new long[shapeNode.size()];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = shapeNode.get(i).asLong();
        }
        return shape;
    }

    private record TensorMeta(String name, String dtype, long[] shape) {
        int rank() {
            return shape.length;
        }

        long dim(int dim) {
            return shape[dim];
        }

        String shapeString() {
            StringBuilder builder = new StringBuilder("[");
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) {
                    builder.append(',');
                }
                builder.append(shape[i]);
            }
            return builder.append(']').toString();
        }
    }

    record Inspection(Result text, ProjectorSummary projectors) {
        static Inspection pass(ProjectorSummary projectors) {
            return new Inspection(Result.pass(projectors), projectors);
        }

        static Inspection invalid(List<String> messages, ProjectorSummary projectors) {
            return new Inspection(Result.invalid(messages, projectors), projectors);
        }
    }

    record ProjectorSummary(boolean visionProjection, boolean audioProjection) {
        static ProjectorSummary empty() {
            return new ProjectorSummary(false, false);
        }

        static ProjectorSummary from(Map<String, TensorMeta> tensors) {
            if (tensors == null || tensors.isEmpty()) {
                return empty();
            }
            return new ProjectorSummary(
                    hasAny(tensors, "model.embed_vision.embedding_projection.weight",
                            "embed_vision.embedding_projection.weight"),
                    hasAny(tensors, "model.embed_audio.embedding_projection.weight",
                            "embed_audio.embedding_projection.weight"));
        }

        String display() {
            if (visionProjection && audioProjection) {
                return "vision/audio projector tensors detected";
            }
            if (visionProjection) {
                return "vision projector tensor detected";
            }
            if (audioProjection) {
                return "audio projector tensor detected";
            }
            return "vision/audio projector tensors were not detected";
        }

        private static boolean hasAny(Map<String, TensorMeta> tensors, String... names) {
            for (String name : names) {
                if (tensors.containsKey(name)) {
                    return true;
                }
            }
            return false;
        }
    }
}

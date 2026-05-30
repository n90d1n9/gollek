package tech.kayys.gollek.train.data.multimodal;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Persistable membership manifest for reproducible multimodal train/validation/test experiments.
 */
public record MultimodalSplitManifest(
        int sampleCount,
        List<Integer> trainIndices,
        List<Integer> validationIndices,
        List<Integer> testIndices,
        List<SampleFingerprint> samples) {
    static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    public MultimodalSplitManifest {
        if (sampleCount < 3) {
            throw new IllegalArgumentException("sampleCount must be at least 3");
        }
        trainIndices = immutableIndices(trainIndices, "trainIndices");
        validationIndices = immutableIndices(validationIndices, "validationIndices");
        testIndices = immutableIndices(testIndices, "testIndices");
        samples = List.copyOf(Objects.requireNonNull(samples, "samples must not be null"));
        requireFullCoverage(sampleCount, trainIndices, validationIndices, testIndices);
        if (samples.size() != sampleCount) {
            throw new IllegalArgumentException(
                    "samples size must match sampleCount, got " + samples.size() + " and " + sampleCount);
        }
        for (int index = 0; index < samples.size(); index++) {
            if (samples.get(index).index() != index) {
                throw new IllegalArgumentException("samples must be sorted by contiguous source index");
            }
        }
    }

    public static MultimodalSplitManifest capture(Dataset.ThreeWaySplit<List<MultimodalContent>> split) {
        Objects.requireNonNull(split, "split must not be null");
        List<Integer> train = MultimodalDatasetSplits.sourceIndices(split.train());
        List<Integer> validation = MultimodalDatasetSplits.sourceIndices(split.validation());
        List<Integer> test = MultimodalDatasetSplits.sourceIndices(split.test());
        int sampleCount = train.size() + validation.size() + test.size();

        Map<Integer, SampleFingerprint> fingerprints = new LinkedHashMap<>();
        capturePartition(split.train(), train, fingerprints);
        capturePartition(split.validation(), validation, fingerprints);
        capturePartition(split.test(), test, fingerprints);

        List<SampleFingerprint> samples = new ArrayList<>(sampleCount);
        for (int index = 0; index < sampleCount; index++) {
            SampleFingerprint fingerprint = fingerprints.get(index);
            if (fingerprint == null) {
                throw new IllegalArgumentException("split does not cover source index " + index);
            }
            samples.add(fingerprint);
        }
        return new MultimodalSplitManifest(sampleCount, train, validation, test, samples);
    }

    public static MultimodalSplitManifest read(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        return MAPPER.readValue(path.toFile(), MultimodalSplitManifest.class);
    }

    public void writeTo(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        Path parent = path.toAbsolutePath().normalize().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        MAPPER.writeValue(path.toFile(), this);
    }

    public Dataset.ThreeWaySplit<List<MultimodalContent>> applyTo(
            Dataset<? extends List<MultimodalContent>> dataset) {
        validateAgainst(dataset);
        return MultimodalDatasetSplits.threeWayByIndices(dataset, trainIndices, validationIndices, testIndices);
    }

    public MultimodalManifestAuditReport audit(Dataset<? extends List<MultimodalContent>> dataset) {
        return MultimodalManifestAuditReport.inspect(sampleCount, samples, dataset);
    }

    public boolean matches(Dataset<? extends List<MultimodalContent>> dataset) {
        return audit(dataset).matches();
    }

    public void validateAgainst(Dataset<? extends List<MultimodalContent>> dataset) {
        audit(dataset).throwIfInvalid();
    }

    public String summary() {
        return "Multimodal split manifest: samples=" + sampleCount
                + ", train=" + trainIndices.size()
                + ", validation=" + validationIndices.size()
                + ", test=" + testIndices.size();
    }

    private static void capturePartition(
            Dataset<List<MultimodalContent>> partition,
            List<Integer> sourceIndices,
            Map<Integer, SampleFingerprint> fingerprints) {
        if (partition.size() != sourceIndices.size()) {
            throw new IllegalArgumentException("partition size does not match source index count");
        }
        for (int row = 0; row < partition.size(); row++) {
            int sourceIndex = sourceIndices.get(row);
            SampleFingerprint previous = fingerprints.put(
                    sourceIndex,
                    fingerprint(sourceIndex, partition.get(row)));
            if (previous != null) {
                throw new IllegalArgumentException("source index appears in more than one partition: " + sourceIndex);
            }
        }
    }

    static SampleFingerprint fingerprint(int index, List<MultimodalContent> sample) {
        List<MultimodalContent> parts = MultimodalDatasetSupport.immutableSample(sample, "sample");
        List<String> sourcePaths = MultimodalDatasetSplits.sourcePaths(parts).stream().sorted().toList();
        List<String> mimeTypes = parts.stream()
                .map(MultimodalContent::getMimeType)
                .filter(value -> value != null && !value.isBlank())
                .map(String::trim)
                .sorted()
                .toList();
        return new SampleFingerprint(
                index,
                MultimodalDatasetSplits.signature(parts),
                digest(parts),
                sourcePaths,
                mimeTypes);
    }

    private static String digest(List<MultimodalContent> sample) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (MultimodalContent content : sample) {
                update(digest, "modality", content.getModality().name());
                update(digest, "text", content.getText());
                update(digest, "mimeType", content.getMimeType());
                update(digest, "base64Data", content.getBase64Data());
                update(digest, "uri", content.getUri());
                update(digest, "documentFormat", content.getDocumentFormat());
                update(digest, "rawBytes", content.getRawBytes());
                update(digest, "embedding", content.getEmbedding());
                update(digest, "timeSeries", content.getTimeSeries());
                update(digest, "samplingRateHz", Long.toString(content.getSamplingRateHz()));
                update(digest, "metadata", metadataDigest(content.getMetadata()));
            }
            return hex(digest.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private static String metadataDigest(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return null;
        }
        List<String> entries = metadata.entrySet().stream()
                .map(entry -> entry.getKey() + "=" + Objects.toString(entry.getValue(), "null"))
                .sorted()
                .toList();
        return String.join("\n", entries);
    }

    private static void update(MessageDigest digest, String key, String value) {
        digest.update(key.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        if (value != null) {
            digest.update(value.getBytes(StandardCharsets.UTF_8));
        }
        digest.update((byte) 0x1F);
    }

    private static void update(MessageDigest digest, String key, byte[] value) {
        digest.update(key.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        if (value != null) {
            digest.update(value);
        }
        digest.update((byte) 0x1F);
    }

    private static void update(MessageDigest digest, String key, float[] value) {
        digest.update(key.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        if (value != null) {
            for (float item : value) {
                digest.update(Float.toString(item).getBytes(StandardCharsets.UTF_8));
                digest.update((byte) 0);
            }
        }
        digest.update((byte) 0x1F);
    }

    private static void update(MessageDigest digest, String key, double[] value) {
        digest.update(key.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        if (value != null) {
            for (double item : value) {
                digest.update(Double.toString(item).getBytes(StandardCharsets.UTF_8));
                digest.update((byte) 0);
            }
        }
        digest.update((byte) 0x1F);
    }

    private static String hex(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            builder.append(Character.forDigit((value >>> 4) & 0xF, 16));
            builder.append(Character.forDigit(value & 0xF, 16));
        }
        return builder.toString();
    }

    static List<Integer> immutableIndices(List<Integer> indices, String owner) {
        Objects.requireNonNull(indices, owner + " must not be null");
        List<Integer> copy = new ArrayList<>(indices.size());
        for (Integer index : indices) {
            int value = Objects.requireNonNull(index, owner + " must not contain null indices");
            if (value < 0) {
                throw new IllegalArgumentException(owner + " must not contain negative indices");
            }
            copy.add(value);
        }
        return Collections.unmodifiableList(copy);
    }

    @SafeVarargs
    static void requireFullCoverage(int sampleCount, List<Integer>... partitions) {
        Set<Integer> seen = new HashSet<>();
        for (List<Integer> partition : partitions) {
            if (partition.isEmpty()) {
                throw new IllegalArgumentException("split manifest partitions must not be empty");
            }
            for (int index : partition) {
                if (index >= sampleCount) {
                    throw new IllegalArgumentException(
                            "split index " + index + " is outside sampleCount " + sampleCount);
                }
                if (!seen.add(index)) {
                    throw new IllegalArgumentException("split index appears in multiple partitions: " + index);
                }
            }
        }
        if (seen.size() != sampleCount) {
            throw new IllegalArgumentException("split manifest must cover every source sample exactly once");
        }
    }

    public record SampleFingerprint(
            int index,
            String signature,
            String digest,
            List<String> sourcePaths,
            List<String> mimeTypes) {
        public SampleFingerprint {
            if (index < 0) {
                throw new IllegalArgumentException("index must be non-negative");
            }
            signature = Objects.requireNonNull(signature, "signature must not be null");
            digest = Objects.requireNonNull(digest, "digest must not be null");
            sourcePaths = List.copyOf(Objects.requireNonNull(sourcePaths, "sourcePaths must not be null"));
            mimeTypes = List.copyOf(Objects.requireNonNull(mimeTypes, "mimeTypes must not be null"));
        }
    }
}

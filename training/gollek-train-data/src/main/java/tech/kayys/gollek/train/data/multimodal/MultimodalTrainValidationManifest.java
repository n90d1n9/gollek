package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Persistable membership manifest for reproducible multimodal train/validation experiments.
 */
public record MultimodalTrainValidationManifest(
        int sampleCount,
        List<Integer> trainIndices,
        List<Integer> validationIndices,
        List<MultimodalSplitManifest.SampleFingerprint> samples) {
    public MultimodalTrainValidationManifest {
        if (sampleCount < 2) {
            throw new IllegalArgumentException("sampleCount must be at least 2");
        }
        trainIndices = MultimodalSplitManifest.immutableIndices(trainIndices, "trainIndices");
        validationIndices = MultimodalSplitManifest.immutableIndices(validationIndices, "validationIndices");
        samples = List.copyOf(Objects.requireNonNull(samples, "samples must not be null"));
        MultimodalSplitManifest.requireFullCoverage(sampleCount, trainIndices, validationIndices);
        if (samples.size() != sampleCount) {
            throw new IllegalArgumentException("samples size must match sampleCount");
        }
        for (int index = 0; index < samples.size(); index++) {
            if (samples.get(index).index() != index) {
                throw new IllegalArgumentException("samples must be sorted by contiguous source index");
            }
        }
    }

    public static MultimodalTrainValidationManifest capture(Dataset.Split<List<MultimodalContent>> split) {
        Objects.requireNonNull(split, "split must not be null");
        List<Integer> train = MultimodalDatasetSplits.sourceIndices(split.train());
        List<Integer> validation = MultimodalDatasetSplits.sourceIndices(split.validation());
        int sampleCount = train.size() + validation.size();

        Map<Integer, MultimodalSplitManifest.SampleFingerprint> fingerprints = new LinkedHashMap<>();
        capturePartition(split.train(), train, fingerprints);
        capturePartition(split.validation(), validation, fingerprints);

        List<MultimodalSplitManifest.SampleFingerprint> samples = new ArrayList<>(sampleCount);
        for (int index = 0; index < sampleCount; index++) {
            MultimodalSplitManifest.SampleFingerprint fingerprint = fingerprints.get(index);
            if (fingerprint == null) {
                throw new IllegalArgumentException("split does not cover source index " + index);
            }
            samples.add(fingerprint);
        }
        return new MultimodalTrainValidationManifest(sampleCount, train, validation, samples);
    }

    public static MultimodalTrainValidationManifest read(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        return MultimodalSplitManifest.MAPPER.readValue(path.toFile(), MultimodalTrainValidationManifest.class);
    }

    public void writeTo(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        Path parent = path.toAbsolutePath().normalize().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        MultimodalSplitManifest.MAPPER.writeValue(path.toFile(), this);
    }

    public Dataset.Split<List<MultimodalContent>> applyTo(Dataset<? extends List<MultimodalContent>> dataset) {
        validateAgainst(dataset);
        return MultimodalDatasetSplits.splitByIndices(dataset, trainIndices, validationIndices);
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
        return "Multimodal train/validation manifest: samples=" + sampleCount
                + ", train=" + trainIndices.size()
                + ", validation=" + validationIndices.size();
    }

    private static void capturePartition(
            Dataset<List<MultimodalContent>> partition,
            List<Integer> sourceIndices,
            Map<Integer, MultimodalSplitManifest.SampleFingerprint> fingerprints) {
        if (partition.size() != sourceIndices.size()) {
            throw new IllegalArgumentException("partition size does not match source index count");
        }
        for (int row = 0; row < partition.size(); row++) {
            int sourceIndex = sourceIndices.get(row);
            MultimodalSplitManifest.SampleFingerprint previous = fingerprints.put(
                    sourceIndex,
                    MultimodalSplitManifest.fingerprint(sourceIndex, partition.get(row)));
            if (previous != null) {
                throw new IllegalArgumentException("source index appears in more than one partition: " + sourceIndex);
            }
        }
    }
}

package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Persistable fold membership manifest for reproducible multimodal cross-validation experiments.
 */
public record MultimodalCrossValidationManifest(
        int sampleCount,
        int foldCount,
        List<FoldMembership> folds,
        List<MultimodalSplitManifest.SampleFingerprint> samples) {
    public MultimodalCrossValidationManifest {
        if (sampleCount < 2) {
            throw new IllegalArgumentException("sampleCount must be at least 2");
        }
        if (foldCount < 2) {
            throw new IllegalArgumentException("foldCount must be at least 2");
        }
        folds = List.copyOf(Objects.requireNonNull(folds, "folds must not be null"));
        samples = List.copyOf(Objects.requireNonNull(samples, "samples must not be null"));
        if (folds.size() != foldCount) {
            throw new IllegalArgumentException("folds size must match foldCount");
        }
        if (samples.size() != sampleCount) {
            throw new IllegalArgumentException("samples size must match sampleCount");
        }
        Set<Integer> validationCoverage = new HashSet<>();
        for (int index = 0; index < folds.size(); index++) {
            FoldMembership fold = folds.get(index);
            if (fold.foldIndex() != index) {
                throw new IllegalArgumentException("fold indices must be contiguous and sorted");
            }
            MultimodalSplitManifest.requireFullCoverage(sampleCount, fold.trainIndices(), fold.validationIndices());
            validationCoverage.addAll(fold.validationIndices());
        }
        if (validationCoverage.size() != sampleCount) {
            throw new IllegalArgumentException("each source sample must appear in validation exactly once");
        }
        for (int index = 0; index < samples.size(); index++) {
            if (samples.get(index).index() != index) {
                throw new IllegalArgumentException("samples must be sorted by contiguous source index");
            }
        }
    }

    public static MultimodalCrossValidationManifest capture(
            List<Dataset.Fold<List<MultimodalContent>>> folds) {
        Objects.requireNonNull(folds, "folds must not be null");
        if (folds.size() < 2) {
            throw new IllegalArgumentException("folds must contain at least two folds");
        }
        List<FoldMembership> memberships = new ArrayList<>(folds.size());
        int sampleCount = -1;
        List<MultimodalSplitManifest.SampleFingerprint> samples = null;
        for (int index = 0; index < folds.size(); index++) {
            Dataset.Fold<List<MultimodalContent>> fold =
                    Objects.requireNonNull(folds.get(index), "folds must not contain null");
            if (fold.foldIndex() != index || fold.foldCount() != folds.size()) {
                throw new IllegalArgumentException("fold metadata must be contiguous and match fold list size");
            }
            List<Integer> train = MultimodalDatasetSplits.sourceIndices(fold.train());
            List<Integer> validation = MultimodalDatasetSplits.sourceIndices(fold.validation());
            int foldSampleCount = train.size() + validation.size();
            if (sampleCount < 0) {
                sampleCount = foldSampleCount;
                samples = captureSamples(fold.train(), train, fold.validation(), validation, sampleCount);
            } else if (sampleCount != foldSampleCount) {
                throw new IllegalArgumentException("all folds must cover the same source dataset size");
            }
            memberships.add(new FoldMembership(index, train, validation));
        }
        return new MultimodalCrossValidationManifest(sampleCount, folds.size(), memberships, samples);
    }

    public static MultimodalCrossValidationManifest read(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        return MultimodalSplitManifest.MAPPER.readValue(path.toFile(), MultimodalCrossValidationManifest.class);
    }

    public void writeTo(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        Path parent = path.toAbsolutePath().normalize().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        MultimodalSplitManifest.MAPPER.writeValue(path.toFile(), this);
    }

    public List<Dataset.Fold<List<MultimodalContent>>> applyTo(
            Dataset<? extends List<MultimodalContent>> dataset) {
        validateAgainst(dataset);
        List<Dataset.Fold<List<MultimodalContent>>> replayed = new ArrayList<>(foldCount);
        for (FoldMembership fold : folds) {
            replayed.add(MultimodalDatasetSplits.foldByIndices(
                    dataset,
                    fold.foldIndex(),
                    foldCount,
                    fold.trainIndices(),
                    fold.validationIndices()));
        }
        return List.copyOf(replayed);
    }

    public boolean matches(Dataset<? extends List<MultimodalContent>> dataset) {
        return audit(dataset).matches();
    }

    public void validateAgainst(Dataset<? extends List<MultimodalContent>> dataset) {
        audit(dataset).throwIfInvalid();
    }

    public MultimodalManifestAuditReport audit(Dataset<? extends List<MultimodalContent>> dataset) {
        return MultimodalManifestAuditReport.inspect(sampleCount, samples, dataset);
    }

    public String summary() {
        return "Multimodal cross-validation manifest: samples=" + sampleCount
                + ", folds=" + foldCount
                + ", validationSamples=" + totalValidationSamples();
    }

    public int totalValidationSamples() {
        int count = 0;
        for (FoldMembership fold : folds) {
            count += fold.validationIndices().size();
        }
        return count;
    }

    private static List<MultimodalSplitManifest.SampleFingerprint> captureSamples(
            Dataset<List<MultimodalContent>> train,
            List<Integer> trainIndices,
            Dataset<List<MultimodalContent>> validation,
            List<Integer> validationIndices,
            int sampleCount) {
        List<MultimodalSplitManifest.SampleFingerprint> samples = new ArrayList<>(sampleCount);
        for (int index = 0; index < sampleCount; index++) {
            samples.add(null);
        }
        capturePartition(train, trainIndices, samples);
        capturePartition(validation, validationIndices, samples);
        for (int index = 0; index < samples.size(); index++) {
            if (samples.get(index) == null) {
                throw new IllegalArgumentException("fold does not cover source index " + index);
            }
        }
        return List.copyOf(samples);
    }

    private static void capturePartition(
            Dataset<List<MultimodalContent>> partition,
            List<Integer> sourceIndices,
            List<MultimodalSplitManifest.SampleFingerprint> samples) {
        if (partition.size() != sourceIndices.size()) {
            throw new IllegalArgumentException("partition size does not match source index count");
        }
        for (int row = 0; row < partition.size(); row++) {
            int sourceIndex = sourceIndices.get(row);
            if (samples.get(sourceIndex) != null) {
                throw new IllegalArgumentException("source index appears in both train and validation: " + sourceIndex);
            }
            samples.set(sourceIndex, MultimodalSplitManifest.fingerprint(sourceIndex, partition.get(row)));
        }
    }

    public record FoldMembership(
            int foldIndex,
            List<Integer> trainIndices,
            List<Integer> validationIndices) {
        public FoldMembership {
            if (foldIndex < 0) {
                throw new IllegalArgumentException("foldIndex must be non-negative");
            }
            trainIndices = MultimodalSplitManifest.immutableIndices(trainIndices, "trainIndices");
            validationIndices = MultimodalSplitManifest.immutableIndices(validationIndices, "validationIndices");
        }
    }
}

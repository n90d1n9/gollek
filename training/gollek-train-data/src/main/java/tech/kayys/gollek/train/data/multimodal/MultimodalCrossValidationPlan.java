package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.Dataset;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Immutable trainer-ready result for multimodal cross-validation setup.
 */
public record MultimodalCrossValidationPlan(
        List<Dataset.Fold<List<MultimodalContent>>> folds,
        MultimodalValidationResult validationResult,
        MultimodalCrossValidationReport crossValidationReport,
        MultimodalCrossValidationPlanner.SplitStrategy strategy,
        double signatureTolerance,
        double mimeTypeTolerance) {
    public MultimodalCrossValidationPlan {
        Objects.requireNonNull(folds, "folds must not be null");
        List<Dataset.Fold<List<MultimodalContent>>> copy = new ArrayList<>(folds.size());
        for (Dataset.Fold<List<MultimodalContent>> fold : folds) {
            copy.add(Objects.requireNonNull(fold, "folds must not contain null"));
        }
        if (copy.isEmpty()) {
            throw new IllegalArgumentException("folds must not be empty");
        }
        folds = Collections.unmodifiableList(copy);
        validationResult = Objects.requireNonNull(validationResult, "validationResult must not be null");
        crossValidationReport = Objects.requireNonNull(crossValidationReport, "crossValidationReport must not be null");
        strategy = Objects.requireNonNull(strategy, "strategy must not be null");
        MultimodalSplitReport.requireTolerance(signatureTolerance);
        MultimodalSplitReport.requireTolerance(mimeTypeTolerance);
    }

    public int foldCount() {
        return folds.size();
    }

    public Dataset.Fold<List<MultimodalContent>> fold(int foldIndex) {
        if (foldIndex < 0 || foldIndex >= folds.size()) {
            throw new IndexOutOfBoundsException("foldIndex must be between 0 and " + (folds.size() - 1));
        }
        return folds.get(foldIndex);
    }

    public Dataset<List<MultimodalContent>> train(int foldIndex) {
        return fold(foldIndex).train();
    }

    public Dataset<List<MultimodalContent>> validation(int foldIndex) {
        return fold(foldIndex).validation();
    }

    public DataLoader.Builder<List<MultimodalContent>> trainLoaderBuilder(int foldIndex) {
        return DataLoader.builder(train(foldIndex));
    }

    public DataLoader.Builder<List<MultimodalContent>> validationLoaderBuilder(int foldIndex) {
        return DataLoader.builder(validation(foldIndex));
    }

    public DataLoader<List<MultimodalContent>> trainLoader(int foldIndex, int batchSize) {
        return trainLoaderBuilder(foldIndex)
                .batchSize(batchSize)
                .build();
    }

    public DataLoader<List<MultimodalContent>> validationLoader(int foldIndex, int batchSize) {
        return validationLoaderBuilder(foldIndex)
                .batchSize(batchSize)
                .build();
    }

    public <B> DataLoader.CollatingDataLoader<List<MultimodalContent>, B> trainLoader(
            int foldIndex,
            int batchSize,
            Function<? super List<List<MultimodalContent>>, ? extends B> collateFn) {
        return trainLoaderBuilder(foldIndex)
                .batchSize(batchSize)
                .collate(collateFn);
    }

    public <B> DataLoader.CollatingDataLoader<List<MultimodalContent>, B> validationLoader(
            int foldIndex,
            int batchSize,
            Function<? super List<List<MultimodalContent>>, ? extends B> collateFn) {
        return validationLoaderBuilder(foldIndex)
                .batchSize(batchSize)
                .collate(collateFn);
    }

    public DataLoader.CollatingDataLoader<List<MultimodalContent>, TextAssetBatch> trainTextAssetLoader(
            int foldIndex,
            int batchSize,
            ModalityType assetModality) {
        return trainLoader(foldIndex, batchSize, MultimodalCollators.textAssetBatch(assetModality));
    }

    public DataLoader.CollatingDataLoader<List<MultimodalContent>, TextAssetBatch> validationTextAssetLoader(
            int foldIndex,
            int batchSize,
            ModalityType assetModality) {
        return validationLoader(foldIndex, batchSize, MultimodalCollators.textAssetBatch(assetModality));
    }

    public MultimodalCrossValidationManifest foldManifest() {
        return MultimodalCrossValidationManifest.capture(folds);
    }

    public void writeFoldManifest(Path path) throws IOException {
        foldManifest().writeTo(path);
    }

    public boolean isReady() {
        return validationResult.isValid()
                && crossValidationReport.isLeakageFree()
                && crossValidationReport.isSignatureBalanced(signatureTolerance)
                && crossValidationReport.isMimeTypeBalanced(mimeTypeTolerance);
    }

    public void throwIfInvalid() {
        validationResult.throwIfInvalid();
        crossValidationReport.throwIfInvalid(signatureTolerance, mimeTypeTolerance);
    }

    public String summary() {
        return "Multimodal cross-validation plan: strategy=" + strategy
                + ", folds=" + foldCount()
                + ", validationSamples=" + crossValidationReport.totalValidationSamples()
                + ", validationErrors=" + validationResult.errors().size()
                + ", validationWarnings=" + validationResult.warnings().size()
                + ", sourceLeakage=" + !crossValidationReport.isLeakageFree()
                + ", maxSignatureDrift=" + crossValidationReport.maxSampleSignatureShareDelta()
                + ", maxMimeTypeDrift=" + crossValidationReport.maxMimeTypeShareDelta();
    }
}

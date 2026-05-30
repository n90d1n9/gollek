package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.Dataset;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Immutable trainer-ready result for a multimodal train/validation split.
 */
public record MultimodalTrainValidationPlan(
        Dataset.Split<List<MultimodalContent>> split,
        MultimodalValidationResult validationResult,
        MultimodalSplitReport splitReport,
        MultimodalTrainValidationPlanner.SplitStrategy strategy,
        double trainFraction,
        double signatureTolerance,
        double mimeTypeTolerance) {
    public MultimodalTrainValidationPlan {
        split = Objects.requireNonNull(split, "split must not be null");
        validationResult = Objects.requireNonNull(validationResult, "validationResult must not be null");
        splitReport = Objects.requireNonNull(splitReport, "splitReport must not be null");
        strategy = Objects.requireNonNull(strategy, "strategy must not be null");
        if (!Double.isFinite(trainFraction) || trainFraction <= 0.0 || trainFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction must be finite and between 0 and 1");
        }
        MultimodalSplitReport.requireTolerance(signatureTolerance);
        MultimodalSplitReport.requireTolerance(mimeTypeTolerance);
    }

    public Dataset<List<MultimodalContent>> train() {
        return split.train();
    }

    public Dataset<List<MultimodalContent>> validation() {
        return split.validation();
    }

    public DataLoader.Builder<List<MultimodalContent>> trainLoaderBuilder() {
        return DataLoader.builder(train());
    }

    public DataLoader.Builder<List<MultimodalContent>> validationLoaderBuilder() {
        return DataLoader.builder(validation());
    }

    public DataLoader<List<MultimodalContent>> trainLoader(int batchSize) {
        return trainLoaderBuilder()
                .batchSize(batchSize)
                .build();
    }

    public DataLoader<List<MultimodalContent>> validationLoader(int batchSize) {
        return validationLoaderBuilder()
                .batchSize(batchSize)
                .build();
    }

    public <B> DataLoader.CollatingDataLoader<List<MultimodalContent>, B> trainLoader(
            int batchSize,
            Function<? super List<List<MultimodalContent>>, ? extends B> collateFn) {
        return trainLoaderBuilder()
                .batchSize(batchSize)
                .collate(collateFn);
    }

    public <B> DataLoader.CollatingDataLoader<List<MultimodalContent>, B> validationLoader(
            int batchSize,
            Function<? super List<List<MultimodalContent>>, ? extends B> collateFn) {
        return validationLoaderBuilder()
                .batchSize(batchSize)
                .collate(collateFn);
    }

    public DataLoader.CollatingDataLoader<List<MultimodalContent>, TextAssetBatch> trainTextAssetLoader(
            int batchSize,
            ModalityType assetModality) {
        return trainLoader(batchSize, MultimodalCollators.textAssetBatch(assetModality));
    }

    public DataLoader.CollatingDataLoader<List<MultimodalContent>, TextAssetBatch> validationTextAssetLoader(
            int batchSize,
            ModalityType assetModality) {
        return validationLoader(batchSize, MultimodalCollators.textAssetBatch(assetModality));
    }

    public MultimodalTrainValidationManifest splitManifest() {
        return MultimodalTrainValidationManifest.capture(split);
    }

    public void writeSplitManifest(Path path) throws IOException {
        splitManifest().writeTo(path);
    }

    public boolean isReady() {
        return validationResult.isValid()
                && !splitReport.hasSourceLeakage()
                && splitReport.isSignatureBalanced(signatureTolerance)
                && splitReport.isMimeTypeBalanced(mimeTypeTolerance);
    }

    public void throwIfInvalid() {
        validationResult.throwIfInvalid();
        splitReport.throwIfInvalid(signatureTolerance, mimeTypeTolerance);
    }

    public String summary() {
        return "Multimodal train/validation plan: strategy=" + strategy
                + ", trainSamples=" + train().size()
                + ", validationSamples=" + validation().size()
                + ", trainFraction=" + trainFraction
                + ", validationErrors=" + validationResult.errors().size()
                + ", validationWarnings=" + validationResult.warnings().size()
                + ", sourceLeakage=" + splitReport.hasSourceLeakage()
                + ", maxSignatureDrift=" + splitReport.maxSampleSignatureShareDelta()
                + ", maxMimeTypeDrift=" + splitReport.maxMimeTypeShareDelta();
    }
}

package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Builds validated, leakage-aware multimodal train/validation plans for trainers.
 */
public final class MultimodalTrainValidationPlanner {
    private static final long DEFAULT_SEED = 0x51A7E5EEDL;
    private static final double DEFAULT_TRAIN_FRACTION = 0.8;
    private static final double DEFAULT_BALANCE_TOLERANCE = 0.25;

    private MultimodalTrainValidationPlanner() {
    }

    public static Builder builder(Dataset<? extends List<MultimodalContent>> dataset) {
        return new Builder(dataset);
    }

    public static Builder textAssetTraining(
            Dataset<? extends List<MultimodalContent>> dataset,
            ModalityType assetModality) {
        return builder(dataset).validator(MultimodalDatasetValidator.textAssetTraining(assetModality));
    }

    public static ReplayBuilder replay(
            Dataset<? extends List<MultimodalContent>> dataset,
            MultimodalTrainValidationManifest manifest) {
        return new ReplayBuilder(dataset, manifest);
    }

    public static ReplayBuilder replay(
            Dataset<? extends List<MultimodalContent>> dataset,
            Path manifestPath) throws IOException {
        return replay(dataset, MultimodalTrainValidationManifest.read(manifestPath));
    }

    public static ReplayBuilder replayTextAssetTraining(
            Dataset<? extends List<MultimodalContent>> dataset,
            MultimodalTrainValidationManifest manifest,
            ModalityType assetModality) {
        return replay(dataset, manifest).validator(MultimodalDatasetValidator.textAssetTraining(assetModality));
    }

    public static ReplayBuilder replayTextAssetTraining(
            Dataset<? extends List<MultimodalContent>> dataset,
            Path manifestPath,
            ModalityType assetModality) throws IOException {
        return replayTextAssetTraining(dataset, MultimodalTrainValidationManifest.read(manifestPath), assetModality);
    }

    public enum SplitStrategy {
        STRATIFIED_BY_SIGNATURE,
        GROUPED_BY_SOURCE_PATH,
        STRATIFIED_GROUPED_BY_SOURCE_PATH,
        REPLAYED_MANIFEST
    }

    public static final class Builder {
        private final Dataset<? extends List<MultimodalContent>> dataset;
        private double trainFraction = DEFAULT_TRAIN_FRACTION;
        private long seed = DEFAULT_SEED;
        private SplitStrategy strategy = SplitStrategy.STRATIFIED_GROUPED_BY_SOURCE_PATH;
        private MultimodalDatasetValidator validator;
        private double signatureTolerance = DEFAULT_BALANCE_TOLERANCE;
        private double mimeTypeTolerance = DEFAULT_BALANCE_TOLERANCE;
        private boolean failOnValidationErrors = true;
        private boolean failOnAuditFailure = true;

        private Builder(Dataset<? extends List<MultimodalContent>> dataset) {
            this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        }

        public Builder trainFraction(double trainFraction) {
            if (!Double.isFinite(trainFraction) || trainFraction <= 0.0 || trainFraction >= 1.0) {
                throw new IllegalArgumentException("trainFraction must be finite and between 0 and 1");
            }
            this.trainFraction = trainFraction;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder strategy(SplitStrategy strategy) {
            this.strategy = Objects.requireNonNull(strategy, "strategy must not be null");
            return this;
        }

        public Builder validator(MultimodalDatasetValidator validator) {
            this.validator = Objects.requireNonNull(validator, "validator must not be null");
            return this;
        }

        public Builder balanceTolerance(double tolerance) {
            return balanceTolerances(tolerance, tolerance);
        }

        public Builder balanceTolerances(double signatureTolerance, double mimeTypeTolerance) {
            MultimodalSplitReport.requireTolerance(signatureTolerance);
            MultimodalSplitReport.requireTolerance(mimeTypeTolerance);
            this.signatureTolerance = signatureTolerance;
            this.mimeTypeTolerance = mimeTypeTolerance;
            return this;
        }

        public Builder failOnValidationErrors(boolean failOnValidationErrors) {
            this.failOnValidationErrors = failOnValidationErrors;
            return this;
        }

        public Builder failOnAuditFailure(boolean failOnAuditFailure) {
            this.failOnAuditFailure = failOnAuditFailure;
            return this;
        }

        public MultimodalTrainValidationPlan build() {
            MultimodalValidationResult validation = resolvedValidator().validate(dataset);
            if (failOnValidationErrors) {
                validation.throwIfInvalid();
            }

            Dataset.Split<List<MultimodalContent>> split = switch (strategy) {
                case STRATIFIED_BY_SIGNATURE -> MultimodalDatasetSplits.stratifiedBySignature(
                        dataset,
                        trainFraction,
                        seed);
                case GROUPED_BY_SOURCE_PATH -> MultimodalDatasetSplits.groupedBySourcePath(
                        dataset,
                        trainFraction,
                        seed);
                case STRATIFIED_GROUPED_BY_SOURCE_PATH -> MultimodalDatasetSplits.stratifiedGroupedBySourcePath(
                        dataset,
                        trainFraction,
                        seed);
                case REPLAYED_MANIFEST -> throw new IllegalStateException(
                        "Use MultimodalTrainValidationPlanner.replay(dataset, manifest) to replay saved splits");
            };
            MultimodalSplitReport report = MultimodalSplitDiagnostics.inspect(split);
            MultimodalTrainValidationPlan plan = new MultimodalTrainValidationPlan(
                    split,
                    validation,
                    report,
                    strategy,
                    trainFraction,
                    signatureTolerance,
                    mimeTypeTolerance);
            if (failOnAuditFailure) {
                plan.throwIfInvalid();
            }
            return plan;
        }

        private MultimodalDatasetValidator resolvedValidator() {
            if (validator != null) {
                return validator;
            }
            return defaultValidator(2);
        }
    }

    public static final class ReplayBuilder {
        private final Dataset<? extends List<MultimodalContent>> dataset;
        private final MultimodalTrainValidationManifest manifest;
        private MultimodalDatasetValidator validator;
        private double signatureTolerance = DEFAULT_BALANCE_TOLERANCE;
        private double mimeTypeTolerance = DEFAULT_BALANCE_TOLERANCE;
        private boolean failOnValidationErrors = true;
        private boolean failOnAuditFailure = true;

        private ReplayBuilder(
                Dataset<? extends List<MultimodalContent>> dataset,
                MultimodalTrainValidationManifest manifest) {
            this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
            this.manifest = Objects.requireNonNull(manifest, "manifest must not be null");
        }

        public ReplayBuilder validator(MultimodalDatasetValidator validator) {
            this.validator = Objects.requireNonNull(validator, "validator must not be null");
            return this;
        }

        public ReplayBuilder balanceTolerance(double tolerance) {
            return balanceTolerances(tolerance, tolerance);
        }

        public ReplayBuilder balanceTolerances(double signatureTolerance, double mimeTypeTolerance) {
            MultimodalSplitReport.requireTolerance(signatureTolerance);
            MultimodalSplitReport.requireTolerance(mimeTypeTolerance);
            this.signatureTolerance = signatureTolerance;
            this.mimeTypeTolerance = mimeTypeTolerance;
            return this;
        }

        public ReplayBuilder failOnValidationErrors(boolean failOnValidationErrors) {
            this.failOnValidationErrors = failOnValidationErrors;
            return this;
        }

        public ReplayBuilder failOnAuditFailure(boolean failOnAuditFailure) {
            this.failOnAuditFailure = failOnAuditFailure;
            return this;
        }

        public MultimodalTrainValidationPlan build() {
            MultimodalValidationResult validation = resolvedValidator().validate(dataset);
            if (failOnValidationErrors) {
                validation.throwIfInvalid();
            }

            Dataset.Split<List<MultimodalContent>> split = manifest.applyTo(dataset);
            MultimodalSplitReport report = MultimodalSplitDiagnostics.inspect(split);
            MultimodalTrainValidationPlan plan = new MultimodalTrainValidationPlan(
                    split,
                    validation,
                    report,
                    SplitStrategy.REPLAYED_MANIFEST,
                    (double) manifest.trainIndices().size() / manifest.sampleCount(),
                    signatureTolerance,
                    mimeTypeTolerance);
            if (failOnAuditFailure) {
                plan.throwIfInvalid();
            }
            return plan;
        }

        private MultimodalDatasetValidator resolvedValidator() {
            if (validator != null) {
                return validator;
            }
            return defaultValidator(2);
        }
    }

    private static MultimodalDatasetValidator defaultValidator(int minSamples) {
        return MultimodalDatasetValidator.builder()
                .minSamples(minSamples)
                .duplicateSourcePaths(MultimodalDatasetValidator.IssuePolicy.WARNING)
                .unresolvedAssets(MultimodalDatasetValidator.IssuePolicy.WARNING)
                .build();
    }
}

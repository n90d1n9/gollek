package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Builds validated, leakage-aware multimodal train/validation/test plans.
 */
public final class MultimodalTrainValidationTestPlanner {
    private static final long DEFAULT_SEED = 0x7E57E5EEDL;
    private static final double DEFAULT_TRAIN_FRACTION = 0.7;
    private static final double DEFAULT_VALIDATION_FRACTION = 0.15;
    private static final double DEFAULT_BALANCE_TOLERANCE = 0.25;

    private MultimodalTrainValidationTestPlanner() {
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
            MultimodalSplitManifest manifest) {
        return new ReplayBuilder(dataset, manifest);
    }

    public static ReplayBuilder replay(
            Dataset<? extends List<MultimodalContent>> dataset,
            Path manifestPath) throws IOException {
        return replay(dataset, MultimodalSplitManifest.read(manifestPath));
    }

    public static ReplayBuilder replayTextAssetTraining(
            Dataset<? extends List<MultimodalContent>> dataset,
            MultimodalSplitManifest manifest,
            ModalityType assetModality) {
        return replay(dataset, manifest).validator(MultimodalDatasetValidator.textAssetTraining(assetModality));
    }

    public static ReplayBuilder replayTextAssetTraining(
            Dataset<? extends List<MultimodalContent>> dataset,
            Path manifestPath,
            ModalityType assetModality) throws IOException {
        return replayTextAssetTraining(dataset, MultimodalSplitManifest.read(manifestPath), assetModality);
    }

    public static final class Builder {
        private final Dataset<? extends List<MultimodalContent>> dataset;
        private double trainFraction = DEFAULT_TRAIN_FRACTION;
        private double validationFraction = DEFAULT_VALIDATION_FRACTION;
        private long seed = DEFAULT_SEED;
        private MultimodalDatasetValidator validator;
        private double signatureTolerance = DEFAULT_BALANCE_TOLERANCE;
        private double mimeTypeTolerance = DEFAULT_BALANCE_TOLERANCE;
        private boolean failOnValidationErrors = true;
        private boolean failOnAuditFailure = true;

        private Builder(Dataset<? extends List<MultimodalContent>> dataset) {
            this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        }

        public Builder fractions(double trainFraction, double validationFraction) {
            requireFractions(trainFraction, validationFraction);
            this.trainFraction = trainFraction;
            this.validationFraction = validationFraction;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
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

        public MultimodalTrainValidationTestPlan build() {
            MultimodalValidationResult validation = resolvedValidator().validate(dataset);
            if (failOnValidationErrors) {
                validation.throwIfInvalid();
            }

            Dataset.ThreeWaySplit<List<MultimodalContent>> split =
                    MultimodalDatasetSplits.stratifiedGroupedThreeWayBySourcePath(
                            dataset,
                            trainFraction,
                            validationFraction,
                            seed);
            MultimodalThreeWaySplitReport report = MultimodalThreeWaySplitDiagnostics.inspect(split);
            MultimodalTrainValidationTestPlan plan = new MultimodalTrainValidationTestPlan(
                    split,
                    validation,
                    report,
                    trainFraction,
                    validationFraction,
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
            return defaultValidator(3);
        }
    }

    public static final class ReplayBuilder {
        private final Dataset<? extends List<MultimodalContent>> dataset;
        private final MultimodalSplitManifest manifest;
        private MultimodalDatasetValidator validator;
        private double signatureTolerance = DEFAULT_BALANCE_TOLERANCE;
        private double mimeTypeTolerance = DEFAULT_BALANCE_TOLERANCE;
        private boolean failOnValidationErrors = true;
        private boolean failOnAuditFailure = true;

        private ReplayBuilder(
                Dataset<? extends List<MultimodalContent>> dataset,
                MultimodalSplitManifest manifest) {
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

        public MultimodalTrainValidationTestPlan build() {
            MultimodalValidationResult validation = resolvedValidator().validate(dataset);
            if (failOnValidationErrors) {
                validation.throwIfInvalid();
            }

            Dataset.ThreeWaySplit<List<MultimodalContent>> split = manifest.applyTo(dataset);
            MultimodalThreeWaySplitReport report = MultimodalThreeWaySplitDiagnostics.inspect(split);
            MultimodalTrainValidationTestPlan plan = new MultimodalTrainValidationTestPlan(
                    split,
                    validation,
                    report,
                    (double) manifest.trainIndices().size() / manifest.sampleCount(),
                    (double) manifest.validationIndices().size() / manifest.sampleCount(),
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
            return defaultValidator(3);
        }
    }

    static void requireFractions(double trainFraction, double validationFraction) {
        if (!Double.isFinite(trainFraction) || trainFraction <= 0.0 || trainFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction must be finite and between 0 and 1");
        }
        if (!Double.isFinite(validationFraction) || validationFraction <= 0.0 || validationFraction >= 1.0) {
            throw new IllegalArgumentException("validationFraction must be finite and between 0 and 1");
        }
        if (trainFraction + validationFraction >= 1.0) {
            throw new IllegalArgumentException("trainFraction + validationFraction must be less than 1");
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

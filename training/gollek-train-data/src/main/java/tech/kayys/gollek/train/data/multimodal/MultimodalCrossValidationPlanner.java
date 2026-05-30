package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Builds leakage-aware multimodal cross-validation plans for trainers.
 */
public final class MultimodalCrossValidationPlanner {
    private static final long DEFAULT_SEED = 0x5EED5EEDL;
    private static final double DEFAULT_BALANCE_TOLERANCE = 0.25;

    private MultimodalCrossValidationPlanner() {
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
            MultimodalCrossValidationManifest manifest) {
        return new ReplayBuilder(dataset, manifest);
    }

    public static ReplayBuilder replay(
            Dataset<? extends List<MultimodalContent>> dataset,
            Path manifestPath) throws IOException {
        return replay(dataset, MultimodalCrossValidationManifest.read(manifestPath));
    }

    public static ReplayBuilder replayTextAssetTraining(
            Dataset<? extends List<MultimodalContent>> dataset,
            MultimodalCrossValidationManifest manifest,
            ModalityType assetModality) {
        return replay(dataset, manifest).validator(MultimodalDatasetValidator.textAssetTraining(assetModality));
    }

    public static ReplayBuilder replayTextAssetTraining(
            Dataset<? extends List<MultimodalContent>> dataset,
            Path manifestPath,
            ModalityType assetModality) throws IOException {
        return replayTextAssetTraining(dataset, MultimodalCrossValidationManifest.read(manifestPath), assetModality);
    }

    public enum SplitStrategy {
        STRATIFIED_BY_SIGNATURE,
        GROUPED_BY_SOURCE_PATH,
        STRATIFIED_GROUPED_BY_SOURCE_PATH,
        REPLAYED_MANIFEST
    }

    public static final class Builder {
        private final Dataset<? extends List<MultimodalContent>> dataset;
        private int folds = 5;
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

        public Builder folds(int folds) {
            if (folds < 2) {
                throw new IllegalArgumentException("folds must be at least 2");
            }
            this.folds = folds;
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

        public MultimodalCrossValidationPlan build() {
            MultimodalValidationResult validation = resolvedValidator().validate(dataset);
            if (failOnValidationErrors) {
                validation.throwIfInvalid();
            }

            List<Dataset.Fold<List<MultimodalContent>>> generatedFolds = switch (strategy) {
                case STRATIFIED_BY_SIGNATURE -> MultimodalDatasetSplits.stratifiedKFoldBySignature(
                        dataset,
                        folds,
                        seed);
                case GROUPED_BY_SOURCE_PATH -> MultimodalDatasetSplits.groupedKFoldBySourcePath(
                        dataset,
                        folds,
                        seed);
                case STRATIFIED_GROUPED_BY_SOURCE_PATH -> MultimodalDatasetSplits.stratifiedGroupedKFoldBySourcePath(
                        dataset,
                        folds,
                        seed);
                case REPLAYED_MANIFEST -> throw new IllegalStateException(
                        "Use MultimodalCrossValidationPlanner.replay(dataset, manifest) to replay saved folds");
            };
            MultimodalCrossValidationReport report = MultimodalCrossValidationDiagnostics.inspect(generatedFolds);
            MultimodalCrossValidationPlan plan = new MultimodalCrossValidationPlan(
                    generatedFolds,
                    validation,
                    report,
                    strategy,
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
            return defaultValidator(folds);
        }
    }

    public static final class ReplayBuilder {
        private final Dataset<? extends List<MultimodalContent>> dataset;
        private final MultimodalCrossValidationManifest manifest;
        private MultimodalDatasetValidator validator;
        private double signatureTolerance = DEFAULT_BALANCE_TOLERANCE;
        private double mimeTypeTolerance = DEFAULT_BALANCE_TOLERANCE;
        private boolean failOnValidationErrors = true;
        private boolean failOnAuditFailure = true;

        private ReplayBuilder(
                Dataset<? extends List<MultimodalContent>> dataset,
                MultimodalCrossValidationManifest manifest) {
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

        public MultimodalCrossValidationPlan build() {
            MultimodalValidationResult validation = resolvedValidator().validate(dataset);
            if (failOnValidationErrors) {
                validation.throwIfInvalid();
            }

            List<Dataset.Fold<List<MultimodalContent>>> replayedFolds = manifest.applyTo(dataset);
            MultimodalCrossValidationReport report = MultimodalCrossValidationDiagnostics.inspect(replayedFolds);
            MultimodalCrossValidationPlan plan = new MultimodalCrossValidationPlan(
                    replayedFolds,
                    validation,
                    report,
                    SplitStrategy.REPLAYED_MANIFEST,
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
            return defaultValidator(manifest.foldCount());
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

package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Materialized profile, split, and mini-batch epochs for one dataset plan.
 */
public record DiscreteTokenDatasetPlan(
        DiscreteTokenDatasetPlanConfig config,
        DiscreteTokenDatasetProfile profile,
        DiscreteTokenDatasetSplit split,
        DiscreteTokenDatasetEpoch trainEpoch,
        DiscreteTokenDatasetEpoch validationEpoch,
        DiscreteTokenDatasetEpoch testEpoch) {

    public DiscreteTokenDatasetPlan {
        config = Objects.requireNonNull(config, "config must not be null");
        profile = Objects.requireNonNull(profile, "profile must not be null");
        split = Objects.requireNonNull(split, "split must not be null");
        trainEpoch = Objects.requireNonNull(trainEpoch, "trainEpoch must not be null");
        validationEpoch = Objects.requireNonNull(validationEpoch, "validationEpoch must not be null");
        testEpoch = Objects.requireNonNull(testEpoch, "testEpoch must not be null");
        if (profile.exampleCount() != split.exampleCount()) {
            throw new IllegalArgumentException("profile example count must equal split example count");
        }
        if (trainEpoch.exampleCount() != split.trainCount()) {
            throw new IllegalArgumentException("train epoch example count must equal split train count");
        }
        if (validationEpoch.exampleCount() != split.validationCount()) {
            throw new IllegalArgumentException("validation epoch example count must equal split validation count");
        }
        if (testEpoch.exampleCount() != split.testCount()) {
            throw new IllegalArgumentException("test epoch example count must equal split test count");
        }
        requirePadTokens(trainEpoch, config, "trainEpoch");
        requirePadTokens(validationEpoch, config, "validationEpoch");
        requirePadTokens(testEpoch, config, "testEpoch");
    }

    public boolean hasValidationEpoch() {
        return validationEpoch.hasBatches();
    }

    public boolean hasTestEpoch() {
        return testEpoch.hasBatches();
    }

    public double trainPaddingRate() {
        return trainEpoch.paddingRate();
    }

    public long emittedTrainingExamples() {
        return trainEpoch.emittedExampleCount();
    }

    public DiscreteTokenDatasetPlanDiagnostics diagnostics() {
        return DiscreteTokenDatasetPlanDiagnostics.from(this);
    }

    public DiscreteTokenDatasetPlanDiagnostics diagnostics(DiscreteTokenDatasetPlanDiagnosticsPolicy policy) {
        return DiscreteTokenDatasetPlanDiagnostics.from(this, policy);
    }

    public DiscreteTokenDatasetPlanReadinessReport readiness(DiscreteTokenDatasetPlanReadinessGate gate) {
        return Objects.requireNonNull(gate, "gate must not be null").evaluate(this);
    }

    public DiscreteTokenDatasetFingerprint fingerprint() {
        return DiscreteTokenDatasetFingerprint.fromPlan(this);
    }

    public DiscreteTokenDatasetFingerprintMatch verifyFingerprint(
            DiscreteTokenDatasetFingerprint expectedFingerprint) {
        return DiscreteTokenDatasetFingerprintMatch.verify(expectedFingerprint, this);
    }

    public DiscreteTokenDatasetFingerprintMatch verifyFingerprintMetadata(Map<?, ?> expectedFingerprintMetadata) {
        return DiscreteTokenDatasetFingerprintMatch.verifyMetadata(expectedFingerprintMetadata, this);
    }

    public DiscreteTokenDatasetFingerprintMatch verifyFingerprintMetadataSection(Map<?, ?> expectedMetadata) {
        return DiscreteTokenDatasetFingerprintMatch.verifyMetadataSection(expectedMetadata, this);
    }

    public DiscreteTokenDatasetFingerprintMatch verifyFingerprintMetadataSection(
            Map<?, ?> expectedMetadata,
            String key) {
        return DiscreteTokenDatasetFingerprintMatch.verifyMetadataSection(expectedMetadata, key, this);
    }

    public DiscreteTokenDatasetPlanReport report() {
        return DiscreteTokenDatasetPlanReport.from(this);
    }

    public DiscreteTokenDatasetPlanReport report(DiscreteTokenDatasetPlanReadinessGate gate) {
        return DiscreteTokenDatasetPlanReport.from(this, gate);
    }

    private static void requirePadTokens(
            DiscreteTokenDatasetEpoch epoch,
            DiscreteTokenDatasetPlanConfig config,
            String name) {
        if (epoch.inputPadToken() != config.inputPadToken()) {
            throw new IllegalArgumentException(name + " input pad token must match config");
        }
        if (epoch.targetPadToken() != config.targetPadToken()) {
            throw new IllegalArgumentException(name + " target pad token must match config");
        }
    }
}

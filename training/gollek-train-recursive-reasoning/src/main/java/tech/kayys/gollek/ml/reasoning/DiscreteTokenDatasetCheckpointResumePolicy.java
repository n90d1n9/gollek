package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Trainer-facing policy for checkpoint resume preflight decisions.
 */
public record DiscreteTokenDatasetCheckpointResumePolicy(
        DiscreteTokenDatasetPlanReadinessGate currentPlanGate,
        DiscreteTokenDatasetCheckpointResumeExpectation expectation,
        DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode) {

    public DiscreteTokenDatasetCheckpointResumePolicy {
        currentPlanGate = Objects.requireNonNull(currentPlanGate, "currentPlanGate must not be null");
        expectation = Objects.requireNonNull(expectation, "expectation must not be null");
        compatibilityMode = Objects.requireNonNull(compatibilityMode, "compatibilityMode must not be null");
    }

    public DiscreteTokenDatasetCheckpointResumePolicy(
            DiscreteTokenDatasetPlanReadinessGate currentPlanGate,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        this(
                currentPlanGate,
                expectation,
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT);
    }

    public static DiscreteTokenDatasetCheckpointResumePolicy training() {
        return new DiscreteTokenDatasetCheckpointResumePolicy(
                DiscreteTokenDatasetPlanReadinessGate.training(),
                DiscreteTokenDatasetCheckpointResumeExpectation.none(),
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT);
    }

    public static DiscreteTokenDatasetCheckpointResumePolicy strict() {
        return new DiscreteTokenDatasetCheckpointResumePolicy(
                DiscreteTokenDatasetPlanReadinessGate.strict(),
                DiscreteTokenDatasetCheckpointResumeExpectation.none(),
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT);
    }

    public static DiscreteTokenDatasetCheckpointResumePolicy compatible() {
        return new DiscreteTokenDatasetCheckpointResumePolicy(
                DiscreteTokenDatasetPlanReadinessGate.lenient(),
                DiscreteTokenDatasetCheckpointResumeExpectation.none(),
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.COMPATIBLE);
    }

    public static DiscreteTokenDatasetCheckpointResumePolicy lenient() {
        return new DiscreteTokenDatasetCheckpointResumePolicy(
                DiscreteTokenDatasetPlanReadinessGate.lenient(),
                DiscreteTokenDatasetCheckpointResumeExpectation.none(),
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.COMPATIBLE);
    }

    public static DiscreteTokenDatasetCheckpointResumePolicy force() {
        return new DiscreteTokenDatasetCheckpointResumePolicy(
                DiscreteTokenDatasetPlanReadinessGate.lenient(),
                DiscreteTokenDatasetCheckpointResumeExpectation.none(),
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.FORCE);
    }

    public static DiscreteTokenDatasetCheckpointResumePolicy fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetCheckpointResumePolicy(
                DiscreteTokenDatasetPlanReadinessGate.fromMetadata(requiredMap(metadata, "currentPlanGate")),
                expectationFromMetadata(metadata),
                compatibilityModeFromMetadata(metadata));
    }

    public DiscreteTokenDatasetCheckpointResumePolicy withCurrentPlanGate(
            DiscreteTokenDatasetPlanReadinessGate currentPlanGate) {
        return new DiscreteTokenDatasetCheckpointResumePolicy(currentPlanGate, expectation, compatibilityMode);
    }

    public DiscreteTokenDatasetCheckpointResumePolicy withExpectation(
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return new DiscreteTokenDatasetCheckpointResumePolicy(currentPlanGate, expectation, compatibilityMode);
    }

    public DiscreteTokenDatasetCheckpointResumePolicy withCompatibilityMode(
            DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode) {
        return new DiscreteTokenDatasetCheckpointResumePolicy(currentPlanGate, expectation, compatibilityMode);
    }

    public DiscreteTokenDatasetPlanReport currentReport(DiscreteTokenDatasetPlan plan) {
        Objects.requireNonNull(plan, "plan must not be null");
        return plan.report(currentPlanGate);
    }

    public DiscreteTokenDatasetCheckpointResumeReport evaluate(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlan plan) {
        return evaluate(DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(checkpointMetadata), plan);
    }

    public DiscreteTokenDatasetCheckpointResumeReport evaluate(
            DiscreteTokenDatasetCheckpointManifest manifest,
            DiscreteTokenDatasetPlan plan) {
        return evaluate(DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest), plan);
    }

    public DiscreteTokenDatasetCheckpointResumeReport evaluate(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlan plan) {
        Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        return DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(
                checkpoint,
                currentReport(plan),
                expectation,
                compatibilityMode)
                .withPolicyMetadata(toMetadata());
    }

    public DiscreteTokenDatasetCheckpointResumeReport evaluate(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlanReport report) {
        Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        return DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(
                checkpoint,
                report,
                expectation,
                compatibilityMode)
                .withPolicyMetadata(toMetadata());
    }

    public DiscreteTokenDatasetCheckpointResumeReport requireReady(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlan plan) {
        DiscreteTokenDatasetCheckpointResumeReport report = evaluate(checkpointMetadata, plan);
        report.requireReady();
        return report;
    }

    public DiscreteTokenDatasetCheckpointResumeReport requireReady(
            DiscreteTokenDatasetCheckpointManifest manifest,
            DiscreteTokenDatasetPlan plan) {
        DiscreteTokenDatasetCheckpointResumeReport report = evaluate(manifest, plan);
        report.requireReady();
        return report;
    }

    public DiscreteTokenDatasetCheckpointResumeReport requireReady(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlan plan) {
        DiscreteTokenDatasetCheckpointResumeReport report = evaluate(checkpoint, plan);
        report.requireReady();
        return report;
    }

    public DiscreteTokenDatasetCheckpointResumeReport requireReady(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlanReport currentPlanReport) {
        DiscreteTokenDatasetCheckpointResumeReport report = evaluate(checkpoint, currentPlanReport);
        report.requireReady();
        return report;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("currentPlanGate", currentPlanGate.toMetadata());
        metadata.put("expectation", expectation.toMetadata());
        metadata.put("compatibilityMode", compatibilityMode.id());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityModeFromMetadata(
            Map<?, ?> metadata) {
        Object value = metadata.containsKey("compatibilityMode")
                ? metadata.get("compatibilityMode")
                : metadata.get("mode");
        return DiscreteTokenDatasetCheckpointResumeCompatibilityMode.fromMetadataValue(value);
    }

    private static DiscreteTokenDatasetCheckpointResumeExpectation expectationFromMetadata(Map<?, ?> metadata) {
        if (!metadata.containsKey("expectation") || metadata.get("expectation") == null) {
            return DiscreteTokenDatasetCheckpointResumeExpectation.none();
        }
        return DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(requiredMap(metadata, "expectation"));
    }

    private static Map<?, ?> requiredMap(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Map<?, ?> map) {
            return map;
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    private static Object required(Map<?, ?> metadata, String key) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            throw new IllegalArgumentException("metadata field '" + key + "' is required");
        }
        return metadata.get(key);
    }
}

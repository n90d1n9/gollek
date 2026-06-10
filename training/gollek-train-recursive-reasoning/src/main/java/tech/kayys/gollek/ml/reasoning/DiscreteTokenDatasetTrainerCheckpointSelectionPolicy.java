package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Read-side policy for choosing a recursive-reasoning trainer checkpoint from an inventory.
 */
public record DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(
        boolean requireReady,
        boolean requireResumeReport,
        boolean failOnInventoryFailures,
        boolean failOnLineageIssues,
        DiscreteTokenDatasetCheckpointResumeExpectation expectation) {

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(
            boolean requireReady,
            boolean requireResumeReport,
            boolean failOnInventoryFailures,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        this(requireReady, requireResumeReport, failOnInventoryFailures, false, expectation);
    }

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy {
        expectation = Objects.requireNonNull(expectation, "expectation must not be null");
    }

    public static Builder builder() {
        return new Builder();
    }

    public static DiscreteTokenDatasetTrainerCheckpointSelectionPolicy latest() {
        return builder()
                .requireReady(false)
                .requireResumeReport(false)
                .build();
    }

    public static DiscreteTokenDatasetTrainerCheckpointSelectionPolicy latestReady() {
        return builder()
                .requireReady(true)
                .requireResumeReport(false)
                .build();
    }

    public static DiscreteTokenDatasetTrainerCheckpointSelectionPolicy latestResumeReady() {
        return builder()
                .requireReady(true)
                .requireResumeReport(true)
                .build();
    }

    public static DiscreteTokenDatasetTrainerCheckpointSelectionPolicy strictLatestReady() {
        return latestReady().withFailOnInventoryFailures(true);
    }

    public static DiscreteTokenDatasetTrainerCheckpointSelectionPolicy strictLatestResumeReady() {
        return latestResumeReady().withFailOnInventoryFailures(true);
    }

    public static DiscreteTokenDatasetTrainerCheckpointSelectionPolicy fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "requireReady"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "requireResumeReport"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "failOnInventoryFailures"),
                DiscreteTokenDatasetMetadataSupport.optionalBoolean(metadata, "failOnLineageIssues", false),
                expectationFromMetadata(metadata));
    }

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy withRequireReady(boolean requireReady) {
        return new DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(
                requireReady,
                requireResumeReport,
                failOnInventoryFailures,
                failOnLineageIssues,
                expectation);
    }

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy withRequireResumeReport(
            boolean requireResumeReport) {
        return new DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(
                requireReady,
                requireResumeReport,
                failOnInventoryFailures,
                failOnLineageIssues,
                expectation);
    }

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy withFailOnInventoryFailures(
            boolean failOnInventoryFailures) {
        return new DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(
                requireReady,
                requireResumeReport,
                failOnInventoryFailures,
                failOnLineageIssues,
                expectation);
    }

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy withFailOnLineageIssues(
            boolean failOnLineageIssues) {
        return new DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(
                requireReady,
                requireResumeReport,
                failOnInventoryFailures,
                failOnLineageIssues,
                expectation);
    }

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy withExpectation(
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return new DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(
                requireReady,
                requireResumeReport,
                failOnInventoryFailures,
                failOnLineageIssues,
                expectation);
    }

    public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy withoutExpectation() {
        return withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.none());
    }

    public Optional<DiscreteTokenDatasetTrainerCheckpointSnapshot> select(
            DiscreteTokenDatasetTrainerCheckpointInventory inventory) {
        Objects.requireNonNull(inventory, "inventory must not be null");
        if (failOnInventoryFailures) {
            inventory.requireNoFailures();
        }
        if (failOnLineageIssues) {
            DiscreteTokenDatasetCheckpointLineageGraph.fromInventory(inventory).requireHealthy();
        }
        return candidates(inventory).stream().max(checkpointComparator());
    }

    public DiscreteTokenDatasetTrainerCheckpointSnapshot require(
            DiscreteTokenDatasetTrainerCheckpointInventory inventory) {
        Objects.requireNonNull(inventory, "inventory must not be null");
        return select(inventory).orElseThrow(() -> new IllegalStateException(
                "no recursive-reasoning checkpoint matched selection policy under "
                        + inventory.rootDir()
                        + ": "
                        + summary()));
    }

    public List<DiscreteTokenDatasetTrainerCheckpointSnapshot> candidates(
            DiscreteTokenDatasetTrainerCheckpointInventory inventory) {
        Objects.requireNonNull(inventory, "inventory must not be null");
        if (failOnInventoryFailures) {
            inventory.requireNoFailures();
        }
        if (failOnLineageIssues) {
            DiscreteTokenDatasetCheckpointLineageGraph.fromInventory(inventory).requireHealthy();
        }
        return inventory.checkpoints().stream()
                .filter(this::accepts)
                .sorted(checkpointComparator())
                .toList();
    }

    public boolean accepts(DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint) {
        return rejectionReasons(checkpoint).isEmpty();
    }

    public List<String> rejectionReasons(DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint) {
        Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        List<String> reasons = new ArrayList<>();
        if (requireReady && !checkpoint.ready()) {
            reasons.add("checkpoint is not ready: " + checkpoint.status());
        }
        if (requireResumeReport && !checkpoint.resumeReportPresent()) {
            reasons.add("checkpoint resume report is missing");
        }
        if (requireResumeReport && checkpoint.resumeReportPresent() && !checkpoint.resumeReady()) {
            reasons.add("checkpoint resume report is not ready: " + checkpoint.resumeStatus());
        }
        reasons.addAll(expectation.rejectionReasons(checkpoint.manifest()));
        return List.copyOf(reasons);
    }

    public String summary() {
        List<String> parts = new ArrayList<>();
        parts.add(requireReady ? "ready checkpoint" : "any checkpoint");
        if (requireResumeReport) {
            parts.add("resume report required");
        }
        if (failOnInventoryFailures) {
            parts.add("inventory failures rejected");
        }
        if (failOnLineageIssues) {
            parts.add("lineage issues rejected");
        }
        if (expectation.active()) {
            parts.add("identity expectation active");
        }
        return String.join(", ", parts);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("requireReady", requireReady);
        metadata.put("requireResumeReport", requireResumeReport);
        metadata.put("failOnInventoryFailures", failOnInventoryFailures);
        metadata.put("failOnLineageIssues", failOnLineageIssues);
        metadata.put("expectation", expectation.toMetadata());
        metadata.put("summary", summary());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static Comparator<DiscreteTokenDatasetTrainerCheckpointSnapshot> checkpointComparator() {
        return Comparator
                .comparingLong((DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint) ->
                        checkpoint.manifest().createdAtEpochMillis())
                .thenComparingLong(checkpoint -> checkpoint.manifest().checkpointStep())
                .thenComparing(checkpoint -> checkpoint.manifest().runId())
                .thenComparing(checkpoint -> checkpoint.checkpointDir().toString());
    }

    private static DiscreteTokenDatasetCheckpointResumeExpectation expectationFromMetadata(Map<?, ?> metadata) {
        if (!metadata.containsKey("expectation") || metadata.get("expectation") == null) {
            return DiscreteTokenDatasetCheckpointResumeExpectation.none();
        }
        Object value = metadata.get("expectation");
        if (value instanceof Map<?, ?> map) {
            return DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(map);
        }
        throw new IllegalArgumentException("metadata field 'expectation' must be a map");
    }

    public static final class Builder {
        private boolean requireReady = true;
        private boolean requireResumeReport;
        private boolean failOnInventoryFailures;
        private boolean failOnLineageIssues;
        private DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.none();

        private Builder() {}

        public Builder requireReady(boolean requireReady) {
            this.requireReady = requireReady;
            return this;
        }

        public Builder requireResumeReport(boolean requireResumeReport) {
            this.requireResumeReport = requireResumeReport;
            return this;
        }

        public Builder failOnInventoryFailures(boolean failOnInventoryFailures) {
            this.failOnInventoryFailures = failOnInventoryFailures;
            return this;
        }

        public Builder failOnLineageIssues(boolean failOnLineageIssues) {
            this.failOnLineageIssues = failOnLineageIssues;
            return this;
        }

        public Builder expectation(DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
            this.expectation = Objects.requireNonNull(expectation, "expectation must not be null");
            return this;
        }

        public Builder noExpectation() {
            this.expectation = DiscreteTokenDatasetCheckpointResumeExpectation.none();
            return this;
        }

        public DiscreteTokenDatasetTrainerCheckpointSelectionPolicy build() {
            return new DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(
                    requireReady,
                    requireResumeReport,
                    failOnInventoryFailures,
                    failOnLineageIssues,
                    expectation);
        }
    }
}

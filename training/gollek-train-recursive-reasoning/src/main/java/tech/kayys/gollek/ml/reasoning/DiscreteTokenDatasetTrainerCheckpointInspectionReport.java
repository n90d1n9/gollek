package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Read-side checkpoint selection audit for CLIs, JBang inspectors, dashboards, and CI gates.
 */
public record DiscreteTokenDatasetTrainerCheckpointInspectionReport(
        Path rootDir,
        DiscreteTokenDatasetTrainerCheckpointInventory inventory,
        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy,
        Optional<DiscreteTokenDatasetTrainerCheckpointSnapshot> selectedCheckpoint,
        Optional<String> selectionFailure,
        List<CheckpointDecision> checkpointDecisions) {

    public DiscreteTokenDatasetTrainerCheckpointInspectionReport {
        rootDir = Objects.requireNonNull(rootDir, "rootDir must not be null");
        inventory = Objects.requireNonNull(inventory, "inventory must not be null");
        policy = Objects.requireNonNull(policy, "policy must not be null");
        selectedCheckpoint = Objects.requireNonNullElse(selectedCheckpoint, Optional.empty());
        selectionFailure = selectionFailure == null
                ? Optional.empty()
                : selectionFailure.map(DiscreteTokenDatasetTrainerCheckpointInspectionReport::requireText);
        checkpointDecisions = checkpointDecisions == null || checkpointDecisions.isEmpty()
                ? List.of()
                : checkpointDecisions.stream()
                        .map(decision -> Objects.requireNonNull(
                                decision,
                                "checkpoint decision must not be null"))
                        .toList();
        if (!rootDir.equals(inventory.rootDir())) {
            throw new IllegalArgumentException("report rootDir must match inventory rootDir");
        }
    }

    public static DiscreteTokenDatasetTrainerCheckpointInspectionReport inspect(
            Path rootDir,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) throws IOException {
        Objects.requireNonNull(rootDir, "rootDir must not be null");
        return inspect(DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir), policy);
    }

    public static DiscreteTokenDatasetTrainerCheckpointInspectionReport inspect(
            DiscreteTokenDatasetTrainerCheckpointInventory inventory,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) {
        Objects.requireNonNull(inventory, "inventory must not be null");
        Objects.requireNonNull(policy, "policy must not be null");

        List<CheckpointDecision> decisions = inventory.checkpoints().stream()
                .map(checkpoint -> CheckpointDecision.from(policy, checkpoint))
                .toList();

        Optional<String> selectionFailure = selectionFailure(inventory, policy);
        Optional<DiscreteTokenDatasetTrainerCheckpointSnapshot> selectedCheckpoint = selectionFailure.isPresent()
                ? Optional.empty()
                : policy.select(inventory);

        return new DiscreteTokenDatasetTrainerCheckpointInspectionReport(
                inventory.rootDir(),
                inventory,
                policy,
                selectedCheckpoint,
                selectionFailure,
                decisions);
    }

    public boolean selected() {
        return selectedCheckpoint.isPresent();
    }

    public boolean selectionSatisfied() {
        return selectedCheckpoint.isPresent() && selectionFailure.isEmpty();
    }

    public int acceptedCount() {
        return (int) checkpointDecisions.stream()
                .filter(CheckpointDecision::accepted)
                .count();
    }

    public int rejectedCount() {
        return checkpointDecisions.size() - acceptedCount();
    }

    public DiscreteTokenDatasetTrainerCheckpointSnapshot requireSelectedCheckpoint() {
        if (selectionFailure.isPresent()) {
            throw new IllegalStateException(selectionFailure.orElseThrow());
        }
        return selectedCheckpoint.orElseThrow(() -> new IllegalStateException(
                "no recursive-reasoning checkpoint matched selection policy under "
                        + rootDir
                        + ": "
                        + policy.summary()));
    }

    public Optional<DiscreteTokenDatasetTrainerCheckpointRestorePlan> selectedRestorePlan() {
        return selectedCheckpoint.map(checkpoint ->
                DiscreteTokenDatasetTrainerCheckpointRestorePlan.fromCheckpoint(rootDir, checkpoint, policy));
    }

    public DiscreteTokenDatasetCheckpointLineageGraph lineageGraph() {
        return DiscreteTokenDatasetCheckpointLineageGraph.fromInventory(inventory);
    }

    public Optional<DiscreteTokenDatasetCheckpointLineageGraph.LineageChain> selectedLineageChain() {
        DiscreteTokenDatasetCheckpointLineageGraph graph = lineageGraph();
        return selectedCheckpoint.map(graph::chainFor);
    }

    public DiscreteTokenDatasetCheckpointLineageGraph.LineageChain requireSelectedLineageChain() {
        return selectedLineageChain().orElseThrow(() -> new IllegalStateException(
                "no selected checkpoint lineage chain is available under "
                        + rootDir
                        + ": "
                        + policy.summary()));
    }

    public DiscreteTokenDatasetTrainerCheckpointRestorePlan requireRestorePlan() {
        return DiscreteTokenDatasetTrainerCheckpointRestorePlan.fromInspectionReport(this);
    }

    public List<CheckpointDecision> acceptedDecisions() {
        return checkpointDecisions.stream()
                .filter(CheckpointDecision::accepted)
                .toList();
    }

    public List<CheckpointDecision> rejectedDecisions() {
        return checkpointDecisions.stream()
                .filter(decision -> !decision.accepted())
                .toList();
    }

    public String summary() {
        String selectedRun = selectedCheckpoint
                .map(checkpoint -> checkpoint.manifest().runId())
                .orElse("none");
        String suffix = selectionFailure
                .map(failure -> "; selection blocked: " + failure)
                .orElse("");
        return "checkpoint inspection "
                + inventory.checkpointCount()
                + " checkpoint(s), "
                + acceptedCount()
                + " accepted, "
                + rejectedCount()
                + " rejected, selected="
                + selectedRun
                + suffix;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("rootDir", rootDir.toString());
        metadata.put("selectionSatisfied", selectionSatisfied());
        metadata.put("selectedCheckpointPresent", selectedCheckpoint.isPresent());
        metadata.put("selectionFailurePresent", selectionFailure.isPresent());
        metadata.put("acceptedCount", acceptedCount());
        metadata.put("rejectedCount", rejectedCount());
        metadata.put("summary", summary());
        metadata.put("policy", policy.toMetadata());
        metadata.put("inventory", inventory.toMetadata());
        metadata.put("lineageGraph", lineageGraph().toMetadata());
        selectedCheckpoint.ifPresent(checkpoint -> metadata.put("selectedCheckpoint", checkpoint.toMetadata()));
        selectedRestorePlan().ifPresent(plan -> metadata.put("restorePlan", plan.toMetadata()));
        selectedLineageChain().ifPresent(chain -> metadata.put("selectedLineageChain", chain.toMetadata()));
        selectionFailure.ifPresent(failure -> metadata.put("selectionFailure", failure));
        metadata.put("checkpointDecisions", checkpointDecisions.stream()
                .map(CheckpointDecision::toMetadata)
                .toList());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static Optional<String> selectionFailure(
            DiscreteTokenDatasetTrainerCheckpointInventory inventory,
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) {
        if (!policy.failOnInventoryFailures() || inventory.failures().isEmpty()) {
            if (!policy.failOnLineageIssues()) {
                return Optional.empty();
            }
            DiscreteTokenDatasetCheckpointLineageGraph graph =
                    DiscreteTokenDatasetCheckpointLineageGraph.fromInventory(inventory);
            return graph.healthy()
                    ? Optional.empty()
                    : Optional.of(graph.healthSummary());
        }
        return Optional.of("checkpoint inventory has "
                + inventory.failureCount()
                + " read failure(s): "
                + inventory.failures().get(0).summary());
    }

    private static String requireText(String value) {
        value = Objects.requireNonNull(value, "selectionFailure must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException("selectionFailure must not be blank");
        }
        return value;
    }

    public record CheckpointDecision(
            DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint,
            boolean accepted,
            List<String> rejectionReasons) {

        public CheckpointDecision {
            checkpoint = Objects.requireNonNull(checkpoint, "checkpoint must not be null");
            rejectionReasons = rejectionReasons == null || rejectionReasons.isEmpty()
                    ? List.of()
                    : rejectionReasons.stream()
                            .map(CheckpointDecision::requireReason)
                            .toList();
            if (accepted && !rejectionReasons.isEmpty()) {
                throw new IllegalArgumentException("accepted checkpoint decision must not have rejection reasons");
            }
            if (!accepted && rejectionReasons.isEmpty()) {
                throw new IllegalArgumentException("rejected checkpoint decision must include rejection reasons");
            }
        }

        public static CheckpointDecision from(
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy,
                DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint) {
            Objects.requireNonNull(policy, "policy must not be null");
            List<String> rejectionReasons = policy.rejectionReasons(checkpoint);
            return new CheckpointDecision(checkpoint, rejectionReasons.isEmpty(), rejectionReasons);
        }

        public String summary() {
            String prefix = accepted ? "accepted" : "rejected";
            String suffix = accepted ? "" : ": " + String.join("; ", rejectionReasons);
            return prefix
                    + " "
                    + checkpoint.manifest().runId()
                    + " "
                    + checkpoint.status()
                    + suffix;
        }

        public Map<String, Object> toMetadata() {
            DiscreteTokenDatasetCheckpointManifestSnapshot manifest = checkpoint.manifest();
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("checkpointDir", checkpoint.checkpointDir().toString());
            metadata.put("runId", manifest.runId());
            metadata.put("experimentName", manifest.experimentName());
            metadata.put("modelFamily", manifest.modelFamily());
            metadata.put("seed", manifest.seed());
            metadata.put("checkpointStep", manifest.checkpointStep());
            metadata.put("createdAtEpochMillis", manifest.createdAtEpochMillis());
            metadata.put("lineage", manifest.lineage().toMetadata());
            metadata.put("status", checkpoint.status());
            metadata.put("resumeStatus", checkpoint.resumeStatus());
            metadata.put("ready", checkpoint.ready());
            metadata.put("resumeReportPresent", checkpoint.resumeReportPresent());
            metadata.put("accepted", accepted);
            metadata.put("rejectionReasons", rejectionReasons);
            metadata.put("summary", summary());
            metadata.put("checkpoint", checkpoint.toMetadata());
            return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        }

        private static String requireReason(String reason) {
            reason = Objects.requireNonNull(reason, "rejection reason must not be null");
            if (reason.isBlank()) {
                throw new IllegalArgumentException("rejection reason must not be blank");
            }
            return reason;
        }
    }
}

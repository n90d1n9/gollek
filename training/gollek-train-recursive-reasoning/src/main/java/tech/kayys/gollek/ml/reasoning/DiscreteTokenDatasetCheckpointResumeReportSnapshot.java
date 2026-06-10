package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Read-side view of a persisted checkpoint resume report sidecar.
 */
public record DiscreteTokenDatasetCheckpointResumeReportSnapshot(
        String status,
        boolean ready,
        boolean schemaAccepted,
        boolean datasetAccepted,
        boolean fingerprintMatched,
        boolean expectationAccepted,
        boolean currentPlanChecked,
        boolean currentPlanAccepted,
        String currentPlanGateStatus,
        DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode,
        boolean forceAccepted,
        List<String> compatibilityWarnings,
        List<DiscreteTokenDatasetCheckpointResumeGate> gates,
        boolean policyTracked,
        String runId,
        long checkpointStep,
        String schemaVersion,
        String datasetGateStatus,
        List<String> rejectionReasons,
        DiscreteTokenDatasetCheckpointResumeExpectation expectation,
        DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
        Map<String, Object> policyMetadata,
        Map<String, Object> currentPlanReportMetadata,
        DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint) {

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot {
        status = DiscreteTokenDatasetMetadataSupport.requireText(status, "status");
        currentPlanGateStatus =
                DiscreteTokenDatasetMetadataSupport.requireText(currentPlanGateStatus, "currentPlanGateStatus");
        runId = DiscreteTokenDatasetMetadataSupport.requireText(runId, "runId");
        if (checkpointStep < 0L) {
            throw new IllegalArgumentException("checkpointStep must be >= 0 but was " + checkpointStep);
        }
        schemaVersion = DiscreteTokenDatasetMetadataSupport.requireText(schemaVersion, "schemaVersion");
        datasetGateStatus = DiscreteTokenDatasetMetadataSupport.requireText(datasetGateStatus, "datasetGateStatus");
        rejectionReasons = DiscreteTokenDatasetMetadataSupport.optionalTextList(
                rejectionReasons,
                "rejectionReasons");
        compatibilityMode = Objects.requireNonNull(compatibilityMode, "compatibilityMode must not be null");
        compatibilityWarnings = DiscreteTokenDatasetMetadataSupport.optionalTextList(
                compatibilityWarnings,
                "compatibilityWarnings");
        gates = immutableGates(gates);
        expectation = Objects.requireNonNull(expectation, "expectation must not be null");
        fingerprintMatch = Objects.requireNonNull(fingerprintMatch, "fingerprintMatch must not be null");
        policyMetadata = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(policyMetadata, "policyMetadata");
        currentPlanReportMetadata = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(
                currentPlanReportMetadata,
                "currentPlanReportMetadata");
        checkpoint = Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        verifyConsistency(
                status,
                ready,
                schemaAccepted,
                datasetAccepted,
                fingerprintMatched,
                expectationAccepted,
                currentPlanChecked,
                currentPlanAccepted,
                currentPlanGateStatus,
                compatibilityMode,
                forceAccepted,
                compatibilityWarnings,
                gates,
                policyTracked,
                runId,
                checkpointStep,
                schemaVersion,
                datasetGateStatus,
                rejectionReasons,
                expectation,
                fingerprintMatch,
                policyMetadata,
                currentPlanReportMetadata,
                checkpoint);
    }

    public static DiscreteTokenDatasetCheckpointResumeReportSnapshot fromReport(
            DiscreteTokenDatasetCheckpointResumeReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return fromMetadata(report.toMetadata());
    }

    public static DiscreteTokenDatasetCheckpointResumeReportSnapshot fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        List<DiscreteTokenDatasetCheckpointResumeGate> gates = gatesFromMetadata(metadata);
        verifyOptionalGateSummary(metadata, gates);
        verifyOptionalActionArtifacts(metadata, gates);
        return new DiscreteTokenDatasetCheckpointResumeReportSnapshot(
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "status"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "ready"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "schemaAccepted"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "datasetAccepted"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "fingerprintMatched"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "expectationAccepted"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "currentPlanChecked"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "currentPlanAccepted"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "currentPlanGateStatus"),
                compatibilityModeFromMetadata(metadata),
                DiscreteTokenDatasetMetadataSupport.optionalBoolean(metadata, "forceAccepted", false),
                DiscreteTokenDatasetMetadataSupport.optionalStringList(metadata, "compatibilityWarnings"),
                gates,
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "policyTracked"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "runId"),
                DiscreteTokenDatasetMetadataSupport.requiredLong(metadata, "checkpointStep"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "schemaVersion"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "datasetGateStatus"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "rejectionReasons"),
                DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, "expectation")),
                DiscreteTokenDatasetFingerprintMatch.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, "fingerprintMatch")),
                DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "policy"),
                DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "currentPlanReport"),
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, "checkpoint")));
    }

    public String summary() {
        String prefix = "checkpoint resume "
                + status
                + ": "
                + runId
                + " step "
                + checkpointStep
                + " dataset "
                + checkpoint.fingerprint().shortValue();
        if (ready) {
            return prefix;
        }
        return prefix + " (" + String.join("; ", rejectionReasons) + ")";
    }

    public void requireReady() {
        if (!ready) {
            throw new IllegalStateException(summary());
        }
    }

    public DiscreteTokenDatasetCheckpointResumeGateSummary gateSummary() {
        return DiscreteTokenDatasetCheckpointResumeGateSummary.fromGates(gates);
    }

    public DiscreteTokenDatasetCheckpointResumeActionPlan actionPlan() {
        return gateSummary().actionPlan();
    }

    public DiscreteTokenDatasetCheckpointResumeReadinessBadge readinessBadge() {
        return gateSummary().readinessBadge();
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> nextActions() {
        return gateSummary().nextActions();
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeAction> primaryAction() {
        return gateSummary().primaryAction();
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> requiredActions() {
        return gateSummary().requiredActions();
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> warningActions() {
        return gateSummary().warningActions();
    }

    public List<String> actionCodes() {
        return gateSummary().actionCodes();
    }

    public List<String> requiredActionCodes() {
        return gateSummary().requiredActionCodes();
    }

    public List<String> warningActionCodes() {
        return gateSummary().warningActionCodes();
    }

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot requireAllGatesAccepted() {
        gateSummary().requireAllAccepted();
        return this;
    }

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot requireNoRequiredActions() {
        actionPlan().requireNoRequiredActions();
        return this;
    }

    public List<DiscreteTokenDatasetCheckpointResumeGate> blockedGates() {
        return gates.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGate::blocked)
                .toList();
    }

    public List<DiscreteTokenDatasetCheckpointResumeGate> warningGates() {
        return gates.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGate::warning)
                .toList();
    }

    public Map<String, DiscreteTokenDatasetCheckpointResumeGate> gatesById() {
        Map<String, DiscreteTokenDatasetCheckpointResumeGate> index = new LinkedHashMap<>();
        for (DiscreteTokenDatasetCheckpointResumeGate gate : gates) {
            if (index.put(gate.id(), gate) != null) {
                throw new IllegalStateException("duplicate resume gate id: " + gate.id());
            }
        }
        return Collections.unmodifiableMap(index);
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeGate> gate(String id) {
        return Optional.ofNullable(gatesById().get(DiscreteTokenDatasetMetadataSupport.requireText(id, "id")));
    }

    public DiscreteTokenDatasetCheckpointResumeGate requireGate(String id) {
        return gate(id)
                .orElseThrow(() -> new IllegalArgumentException("unknown checkpoint resume gate: " + id));
    }

    public boolean gateAccepted(String id) {
        return requireGate(id).accepted();
    }

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot requireGateAccepted(String id) {
        requireGate(id).requireAccepted();
        return this;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status);
        metadata.put("ready", ready);
        metadata.put("schemaAccepted", schemaAccepted);
        metadata.put("datasetAccepted", datasetAccepted);
        metadata.put("fingerprintMatched", fingerprintMatched);
        metadata.put("expectationAccepted", expectationAccepted);
        metadata.put("currentPlanChecked", currentPlanChecked);
        metadata.put("currentPlanAccepted", currentPlanAccepted);
        metadata.put("currentPlanGateStatus", currentPlanGateStatus);
        metadata.put("compatibilityMode", compatibilityMode.id());
        metadata.put("forceAccepted", forceAccepted);
        metadata.put("compatibilityWarnings", compatibilityWarnings);
        metadata.put("gateSummary", gateSummary().toMetadata());
        metadata.put("actionPlan", actionPlan().toMetadata());
        metadata.put("readinessBadge", readinessBadge().toMetadata());
        metadata.put("gates", gates.stream()
                .map(DiscreteTokenDatasetCheckpointResumeGate::toMetadata)
                .toList());
        metadata.put("policyTracked", policyTracked);
        metadata.put("runId", runId);
        metadata.put("checkpointStep", checkpointStep);
        metadata.put("schemaVersion", schemaVersion);
        metadata.put("datasetGateStatus", datasetGateStatus);
        metadata.put("rejectionReasons", rejectionReasons);
        metadata.put("expectation", expectation.toMetadata());
        metadata.put("fingerprintMatch", fingerprintMatch.toMetadata());
        if (policyTracked) {
            metadata.put("policy", policyMetadata);
        }
        if (currentPlanChecked) {
            metadata.put("currentPlanReport", currentPlanReportMetadata);
        }
        metadata.put("checkpoint", checkpoint.toMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void verifyConsistency(
            String status,
            boolean ready,
            boolean schemaAccepted,
            boolean datasetAccepted,
            boolean fingerprintMatched,
            boolean expectationAccepted,
            boolean currentPlanChecked,
            boolean currentPlanAccepted,
            String currentPlanGateStatus,
            DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode,
            boolean forceAccepted,
            List<String> compatibilityWarnings,
            List<DiscreteTokenDatasetCheckpointResumeGate> gates,
            boolean policyTracked,
            String runId,
            long checkpointStep,
            String schemaVersion,
            String datasetGateStatus,
            List<String> rejectionReasons,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            Map<String, Object> policyMetadata,
            Map<String, Object> currentPlanReportMetadata,
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint) {
        String expectedStatus = ready ? "ready" : "blocked";
        if (!expectedStatus.equals(status)) {
            throw new IllegalArgumentException("status must be " + expectedStatus + " when ready=" + ready);
        }
        if (ready && !rejectionReasons.isEmpty()) {
            throw new IllegalArgumentException("ready resume reports must not carry rejection reasons");
        }
        if (forceAccepted != (compatibilityMode.force() && ready && !compatibilityWarnings.isEmpty())) {
            throw new IllegalArgumentException("forceAccepted must match compatibility mode and warnings");
        }
        if (!runId.equals(checkpoint.runId())) {
            throw new IllegalArgumentException("runId must match checkpoint runId");
        }
        if (checkpointStep != checkpoint.checkpointStep()) {
            throw new IllegalArgumentException("checkpointStep must match checkpoint checkpointStep");
        }
        if (!schemaVersion.equals(checkpoint.schemaVersion())) {
            throw new IllegalArgumentException("schemaVersion must match checkpoint schemaVersion");
        }
        if (!datasetGateStatus.equals(checkpoint.datasetGateStatus())) {
            throw new IllegalArgumentException("datasetGateStatus must match checkpoint datasetGateStatus");
        }
        if (schemaAccepted != checkpoint.isCurrentSchema()) {
            throw new IllegalArgumentException("schemaAccepted must match checkpoint schema status");
        }
        if (datasetAccepted != checkpoint.datasetAccepted()) {
            throw new IllegalArgumentException("datasetAccepted must match checkpoint dataset status");
        }
        if (fingerprintMatched != fingerprintMatch.matches()) {
            throw new IllegalArgumentException("fingerprintMatched must match fingerprint comparison");
        }
        if (expectationAccepted != expectation.accepts(checkpoint)) {
            throw new IllegalArgumentException("expectationAccepted must match expectation result");
        }
        if (policyTracked != !policyMetadata.isEmpty()) {
            throw new IllegalArgumentException("policyTracked must match policy metadata presence");
        }
        if (currentPlanChecked != !currentPlanReportMetadata.isEmpty()) {
            throw new IllegalArgumentException("currentPlanChecked must match currentPlanReport presence");
        }
        verifyCurrentPlanReportConsistency(
                currentPlanChecked,
                currentPlanAccepted,
                currentPlanGateStatus,
                currentPlanReportMetadata);
        boolean computedReady = schemaAccepted
                && (datasetAccepted || compatibilityMode.allowRejectedCheckpointDataset())
                && (fingerprintMatched || compatibilityMode.allowFingerprintMismatch())
                && expectationAccepted
                && (currentPlanAccepted || compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus));
        if (ready != computedReady) {
            throw new IllegalArgumentException("ready must match resume report acceptance fields");
        }
        verifyGateConsistency(
                schemaAccepted,
                datasetAccepted,
                fingerprintMatched,
                expectationAccepted,
                currentPlanAccepted,
                currentPlanGateStatus,
                compatibilityMode,
                ready,
                gates);
    }

    private static void verifyGateConsistency(
            boolean schemaAccepted,
            boolean datasetAccepted,
            boolean fingerprintMatched,
            boolean expectationAccepted,
            boolean currentPlanAccepted,
            String currentPlanGateStatus,
            DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode,
            boolean ready,
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        Map<String, Boolean> expected = new LinkedHashMap<>();
        expected.put(DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_SCHEMA, schemaAccepted);
        expected.put(
                DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_DATASET,
                datasetAccepted || compatibilityMode.allowRejectedCheckpointDataset());
        expected.put(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                fingerprintMatched || compatibilityMode.allowFingerprintMismatch());
        expected.put(DiscreteTokenDatasetCheckpointResumeGate.RESUME_EXPECTATION, expectationAccepted);
        expected.put(
                DiscreteTokenDatasetCheckpointResumeGate.CURRENT_PLAN,
                currentPlanAccepted || compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus));

        Map<String, DiscreteTokenDatasetCheckpointResumeGate> actual = new LinkedHashMap<>();
        for (DiscreteTokenDatasetCheckpointResumeGate gate : gates) {
            if (actual.put(gate.id(), gate) != null) {
                throw new IllegalArgumentException("duplicate resume gate id: " + gate.id());
            }
        }
        for (Map.Entry<String, Boolean> entry : expected.entrySet()) {
            DiscreteTokenDatasetCheckpointResumeGate gate = actual.get(entry.getKey());
            if (gate == null) {
                throw new IllegalArgumentException("resume gate is missing: " + entry.getKey());
            }
            if (gate.accepted() != entry.getValue()) {
                throw new IllegalArgumentException("resume gate acceptance does not match field: " + entry.getKey());
            }
        }
        if (ready != gates.stream().allMatch(DiscreteTokenDatasetCheckpointResumeGate::accepted)) {
            throw new IllegalArgumentException("ready must match resume gate acceptance");
        }
    }

    private static void verifyCurrentPlanReportConsistency(
            boolean currentPlanChecked,
            boolean currentPlanAccepted,
            String currentPlanGateStatus,
            Map<String, Object> currentPlanReportMetadata) {
        if (!currentPlanChecked) {
            if (!"not-checked".equals(currentPlanGateStatus)) {
                throw new IllegalArgumentException(
                        "currentPlanGateStatus must be not-checked when currentPlanChecked is false");
            }
            if (!currentPlanAccepted) {
                throw new IllegalArgumentException(
                        "currentPlanAccepted must be true when currentPlanChecked is false");
            }
            return;
        }

        boolean reportAccepted = DiscreteTokenDatasetMetadataSupport.requiredBoolean(
                currentPlanReportMetadata,
                "accepted");
        if (currentPlanAccepted != reportAccepted) {
            throw new IllegalArgumentException("currentPlanAccepted must match currentPlanReport accepted");
        }
        String reportGateStatus = DiscreteTokenDatasetMetadataSupport.requiredString(
                currentPlanReportMetadata,
                "gateStatus");
        if (!currentPlanGateStatus.equals(reportGateStatus)) {
            throw new IllegalArgumentException("currentPlanGateStatus must match currentPlanReport gateStatus");
        }
    }

    private static List<DiscreteTokenDatasetCheckpointResumeGate> gatesFromMetadata(Map<?, ?> metadata) {
        if (metadata.containsKey("gates") && metadata.get("gates") != null) {
            Object value = metadata.get("gates");
            if (value instanceof List<?> list) {
                return immutableGates(list.stream()
                        .map(entry -> {
                            if (entry instanceof Map<?, ?> map) {
                                return DiscreteTokenDatasetCheckpointResumeGate.fromMetadata(map);
                            }
                            throw new IllegalArgumentException("gates entries must be maps");
                        })
                        .toList());
            }
            throw new IllegalArgumentException("metadata field 'gates' must be a list");
        }

        Map<String, Object> currentPlanReportMetadata =
                DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "currentPlanReport");
        DiscreteTokenDatasetPlanReport currentPlanReport = currentPlanReportMetadata.isEmpty()
                ? null
                : DiscreteTokenDatasetPlanReport.fromMetadata(currentPlanReportMetadata);
        DiscreteTokenDatasetCheckpointResumeReport report = new DiscreteTokenDatasetCheckpointResumeReport(
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, "checkpoint")),
                DiscreteTokenDatasetFingerprintMatch.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, "fingerprintMatch")),
                DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, "expectation")),
                currentPlanReport,
                compatibilityModeFromMetadata(metadata),
                DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "policy"));
        return report.gates();
    }

    private static void verifyOptionalGateSummary(
            Map<?, ?> metadata,
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        if (!metadata.containsKey("gateSummary")) {
            return;
        }
        DiscreteTokenDatasetCheckpointResumeGateSummary.fromMetadata(
                DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, "gateSummary"),
                gates);
    }

    private static void verifyOptionalActionArtifacts(
            Map<?, ?> metadata,
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        DiscreteTokenDatasetCheckpointResumeGateSummary summary =
                DiscreteTokenDatasetCheckpointResumeGateSummary.fromGates(gates);
        verifyOptionalMetadata(metadata, "actionPlan", summary.actionPlan().toMetadata());
        verifyOptionalMetadata(metadata, "readinessBadge", summary.readinessBadge().toMetadata());
    }

    private static void verifyOptionalMetadata(Map<?, ?> metadata, String key, Map<String, Object> expected) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            return;
        }
        if (!DiscreteTokenDatasetMetadataSupport.metadataValueMatches(
                expected,
                DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, key),
                true)) {
            throw new IllegalArgumentException("metadata field '" + key + "' does not match resume gates");
        }
    }

    private static List<DiscreteTokenDatasetCheckpointResumeGate> immutableGates(List<?> values) {
        if (values == null || values.isEmpty()) {
            throw new IllegalArgumentException("gates must not be empty");
        }
        return values.stream()
                .map(value -> {
                    if (value instanceof DiscreteTokenDatasetCheckpointResumeGate gate) {
                        return gate;
                    }
                    throw new IllegalArgumentException("gates entries must be checkpoint resume gates");
                })
                .toList();
    }

    private static DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityModeFromMetadata(
            Map<?, ?> metadata) {
        Object value = metadata.containsKey("compatibilityMode")
                ? metadata.get("compatibilityMode")
                : metadata.get("mode");
        return DiscreteTokenDatasetCheckpointResumeCompatibilityMode.fromMetadataValue(value);
    }
}

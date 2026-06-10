package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Resume preflight for a persisted checkpoint manifest and a current dataset plan.
 */
public record DiscreteTokenDatasetCheckpointResumeReport(
        DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
        DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
        DiscreteTokenDatasetCheckpointResumeExpectation expectation,
        DiscreteTokenDatasetPlanReport currentPlanReport,
        DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode,
        Map<String, Object> policyMetadata) {

    public DiscreteTokenDatasetCheckpointResumeReport {
        checkpoint = Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        fingerprintMatch = Objects.requireNonNull(fingerprintMatch, "fingerprintMatch must not be null");
        expectation = Objects.requireNonNull(expectation, "expectation must not be null");
        compatibilityMode = Objects.requireNonNull(compatibilityMode, "compatibilityMode must not be null");
        policyMetadata = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(policyMetadata, "policyMetadata");
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetPlanReport currentPlanReport) {
        this(
                checkpoint,
                fingerprintMatch,
                expectation,
                currentPlanReport,
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT,
                Map.of());
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetPlanReport currentPlanReport,
            Map<String, Object> policyMetadata) {
        this(
                checkpoint,
                fingerprintMatch,
                expectation,
                currentPlanReport,
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT,
                policyMetadata);
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetPlanReport currentPlanReport,
            DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode) {
        this(checkpoint, fingerprintMatch, expectation, currentPlanReport, compatibilityMode, Map.of());
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        this(checkpoint, fingerprintMatch, expectation, null, DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT);
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch) {
        this(checkpoint, fingerprintMatch, DiscreteTokenDatasetCheckpointResumeExpectation.none(), null);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromMetadata(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlan plan) {
        return fromMetadata(checkpointMetadata, plan, DiscreteTokenDatasetCheckpointResumeExpectation.none());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromMetadata(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlan plan,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return fromSnapshot(
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(checkpointMetadata),
                plan,
                expectation);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromMetadata(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlanReport report) {
        return fromMetadata(checkpointMetadata, report, DiscreteTokenDatasetCheckpointResumeExpectation.none());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromMetadata(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlanReport report,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return fromSnapshot(
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(checkpointMetadata),
                report,
                expectation);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlan plan) {
        return fromSnapshot(checkpoint, plan, DiscreteTokenDatasetCheckpointResumeExpectation.none());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlan plan,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        Objects.requireNonNull(plan, "plan must not be null");
        return fromSnapshot(checkpoint, plan.report(), expectation);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlanReport report) {
        return fromSnapshot(checkpoint, report, DiscreteTokenDatasetCheckpointResumeExpectation.none());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlanReport report,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return fromSnapshot(
                checkpoint,
                report,
                expectation,
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlanReport report,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode) {
        Objects.requireNonNull(report, "report must not be null");
        return new DiscreteTokenDatasetCheckpointResumeReport(
                checkpoint,
                checkpoint.verifyReport(report),
                expectation,
                report,
                compatibilityMode);
    }

    public boolean schemaAccepted() {
        return checkpoint.isCurrentSchema();
    }

    public boolean datasetAccepted() {
        return checkpoint.datasetAccepted();
    }

    public boolean fingerprintMatched() {
        return fingerprintMatch.matches();
    }

    public boolean expectationAccepted() {
        return expectation.accepts(checkpoint);
    }

    public boolean currentPlanChecked() {
        return currentPlanReport != null;
    }

    public boolean policyTracked() {
        return !policyMetadata.isEmpty();
    }

    public boolean currentPlanAccepted() {
        return currentPlanReport == null || currentPlanReport.accepted();
    }

    public String currentPlanGateStatus() {
        return currentPlanReport == null ? "not-checked" : currentPlanReport.gateStatus();
    }

    public boolean ready() {
        return schemaAccepted()
                && (datasetAccepted() || compatibilityMode.allowRejectedCheckpointDataset())
                && (fingerprintMatched() || compatibilityMode.allowFingerprintMismatch())
                && expectationAccepted()
                && (currentPlanAccepted() || compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus()));
    }

    public List<DiscreteTokenDatasetCheckpointResumeGate> gates() {
        return List.of(
                checkpointSchemaGate(),
                checkpointDatasetGate(),
                datasetFingerprintGate(),
                resumeExpectationGate(),
                currentPlanGate());
    }

    public DiscreteTokenDatasetCheckpointResumeGateSummary gateSummary() {
        return DiscreteTokenDatasetCheckpointResumeGateSummary.fromGates(gates());
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

    public DiscreteTokenDatasetCheckpointResumeReport requireAllGatesAccepted() {
        gateSummary().requireAllAccepted();
        return this;
    }

    public DiscreteTokenDatasetCheckpointResumeReport requireNoRequiredActions() {
        actionPlan().requireNoRequiredActions();
        return this;
    }

    public List<DiscreteTokenDatasetCheckpointResumeGate> blockedGates() {
        return gates().stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGate::blocked)
                .toList();
    }

    public List<DiscreteTokenDatasetCheckpointResumeGate> warningGates() {
        return gates().stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGate::warning)
                .toList();
    }

    public Map<String, DiscreteTokenDatasetCheckpointResumeGate> gatesById() {
        Map<String, DiscreteTokenDatasetCheckpointResumeGate> index = new LinkedHashMap<>();
        for (DiscreteTokenDatasetCheckpointResumeGate gate : gates()) {
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

    public DiscreteTokenDatasetCheckpointResumeReport requireGateAccepted(String id) {
        requireGate(id).requireAccepted();
        return this;
    }

    public String status() {
        return ready() ? "ready" : "blocked";
    }

    public List<String> rejectionReasons() {
        if (ready()) {
            return List.of();
        }

        List<String> reasons = new ArrayList<>();
        if (!schemaAccepted()) {
            reasons.add("checkpoint manifest schema is "
                    + checkpoint.schemaVersion()
                    + " but expected "
                    + DiscreteTokenDatasetCheckpointManifestSnapshot.SCHEMA_VERSION);
        }
        if (!datasetAccepted() && !compatibilityMode.allowRejectedCheckpointDataset()) {
            reasons.add("checkpoint dataset was not accepted: " + checkpoint.datasetGateStatus());
        }
        if (!fingerprintMatched() && !compatibilityMode.allowFingerprintMismatch()) {
            reasons.add(fingerprintMatch.summary());
        }
        reasons.addAll(expectation.rejectionReasons(checkpoint));
        if (!currentPlanAccepted() && !compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus())) {
            reasons.add("current dataset plan was not accepted: " + currentPlanGateStatus());
        }
        return List.copyOf(reasons);
    }

    public boolean forceAccepted() {
        return compatibilityMode.force() && ready() && !compatibilityWarnings().isEmpty();
    }

    public List<String> compatibilityWarnings() {
        List<String> warnings = new ArrayList<>();
        if (!datasetAccepted() && compatibilityMode.allowRejectedCheckpointDataset()) {
            warnings.add("force mode accepted a checkpoint dataset that was not accepted: "
                    + checkpoint.datasetGateStatus());
        }
        if (!fingerprintMatched() && compatibilityMode.allowFingerprintMismatch()) {
            warnings.add("force mode accepted dataset fingerprint mismatch: " + fingerprintMatch.summary());
        }
        if (!currentPlanAccepted() && compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus())) {
            warnings.add(compatibilityMode.id()
                    + " mode accepted current dataset plan status: "
                    + currentPlanGateStatus());
        }
        return List.copyOf(warnings);
    }

    public String summary() {
        String prefix = "checkpoint resume "
                + status()
                + ": "
                + checkpoint.runId()
                + " step "
                + checkpoint.checkpointStep()
                + " dataset "
                + checkpoint.fingerprint().shortValue();
        if (ready()) {
            return prefix;
        }
        return prefix + " (" + String.join("; ", rejectionReasons()) + ")";
    }

    public void requireReady() {
        if (!ready()) {
            throw new IllegalStateException(summary());
        }
    }

    public DiscreteTokenDatasetCheckpointResumeExplanation explanation() {
        return DiscreteTokenDatasetCheckpointResumeExplanation.from(this);
    }

    public DiscreteTokenDatasetCheckpointResumeReport withPolicyMetadata(Map<String, Object> policyMetadata) {
        return new DiscreteTokenDatasetCheckpointResumeReport(
                checkpoint,
                fingerprintMatch,
                expectation,
                currentPlanReport,
                compatibilityMode,
                policyMetadata);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status());
        metadata.put("ready", ready());
        metadata.put("schemaAccepted", schemaAccepted());
        metadata.put("datasetAccepted", datasetAccepted());
        metadata.put("fingerprintMatched", fingerprintMatched());
        metadata.put("expectationAccepted", expectationAccepted());
        metadata.put("currentPlanChecked", currentPlanChecked());
        metadata.put("currentPlanAccepted", currentPlanAccepted());
        metadata.put("currentPlanGateStatus", currentPlanGateStatus());
        metadata.put("compatibilityMode", compatibilityMode.id());
        metadata.put("forceAccepted", forceAccepted());
        metadata.put("compatibilityWarnings", compatibilityWarnings());
        metadata.put("gateSummary", gateSummary().toMetadata());
        metadata.put("actionPlan", actionPlan().toMetadata());
        metadata.put("readinessBadge", readinessBadge().toMetadata());
        metadata.put("gates", gateMetadata());
        metadata.put("policyTracked", policyTracked());
        metadata.put("runId", checkpoint.runId());
        metadata.put("checkpointStep", checkpoint.checkpointStep());
        metadata.put("schemaVersion", checkpoint.schemaVersion());
        metadata.put("datasetGateStatus", checkpoint.datasetGateStatus());
        metadata.put("rejectionReasons", rejectionReasons());
        metadata.put("expectation", expectation.toMetadata());
        metadata.put("fingerprintMatch", fingerprintMatch.toMetadata());
        if (policyTracked()) {
            metadata.put("policy", policyMetadata);
        }
        if (currentPlanReport != null) {
            metadata.put("currentPlanReport", currentPlanReport.toMetadata());
        }
        metadata.put("checkpoint", checkpoint.toMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private List<Map<String, Object>> gateMetadata() {
        return gates().stream()
                .map(DiscreteTokenDatasetCheckpointResumeGate::toMetadata)
                .toList();
    }

    private DiscreteTokenDatasetCheckpointResumeGate checkpointSchemaGate() {
        boolean accepted = schemaAccepted();
        return gate(
                DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_SCHEMA,
                "manifest",
                accepted,
                accepted ? "info" : "error",
                accepted
                        ? "checkpoint manifest schema matches current schema"
                        : "checkpoint manifest schema mismatch",
                Map.of(
                        "actualSchemaVersion", checkpoint.schemaVersion(),
                        "expectedSchemaVersion", DiscreteTokenDatasetCheckpointManifestSnapshot.SCHEMA_VERSION));
    }

    private DiscreteTokenDatasetCheckpointResumeGate checkpointDatasetGate() {
        boolean accepted = datasetAccepted() || compatibilityMode.allowRejectedCheckpointDataset();
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("checkpointAccepted", datasetAccepted());
        details.put("checkpointGateStatus", checkpoint.datasetGateStatus());
        details.put("compatibilityMode", compatibilityMode.id());
        details.put("compatibilityOverride", !datasetAccepted() && accepted);
        return gate(
                DiscreteTokenDatasetCheckpointResumeGate.CHECKPOINT_DATASET,
                "checkpoint-dataset",
                accepted,
                gateSeverity(datasetAccepted(), accepted),
                datasetAccepted()
                        ? "checkpoint dataset accepted by readiness gate"
                        : accepted
                                ? "compatibility mode accepted checkpoint dataset status: "
                                        + checkpoint.datasetGateStatus()
                                : "checkpoint dataset rejected by readiness gate: "
                                        + checkpoint.datasetGateStatus(),
                details);
    }

    private DiscreteTokenDatasetCheckpointResumeGate datasetFingerprintGate() {
        boolean accepted = fingerprintMatched() || compatibilityMode.allowFingerprintMismatch();
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("matched", fingerprintMatched());
        details.put("expected", fingerprintMatch.expected().toMetadata());
        details.put("actual", fingerprintMatch.actual().toMetadata());
        details.put("mismatchReasons", fingerprintMatch.mismatchReasons());
        details.put("compatibilityMode", compatibilityMode.id());
        details.put("compatibilityOverride", !fingerprintMatched() && accepted);
        return gate(
                DiscreteTokenDatasetCheckpointResumeGate.DATASET_FINGERPRINT,
                "dataset-fingerprint",
                accepted,
                gateSeverity(fingerprintMatched(), accepted),
                fingerprintMatched()
                        ? "dataset fingerprint matched"
                        : accepted
                                ? "compatibility mode accepted dataset fingerprint mismatch"
                                : fingerprintMatch.summary(),
                details);
    }

    private DiscreteTokenDatasetCheckpointResumeGate resumeExpectationGate() {
        boolean accepted = expectationAccepted();
        return gate(
                DiscreteTokenDatasetCheckpointResumeGate.RESUME_EXPECTATION,
                "resume-expectation",
                accepted,
                accepted ? "info" : "error",
                accepted ? "resume expectation accepted" : "resume expectation rejected",
                Map.of(
                        "active", expectation.active(),
                        "rejectionReasons", expectation.rejectionReasons(checkpoint),
                        "expectation", expectation.toMetadata()));
    }

    private DiscreteTokenDatasetCheckpointResumeGate currentPlanGate() {
        boolean accepted = currentPlanAccepted()
                || compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus());
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("checked", currentPlanChecked());
        details.put("accepted", currentPlanAccepted());
        details.put("gateStatus", currentPlanGateStatus());
        details.put("compatibilityMode", compatibilityMode.id());
        details.put("compatibilityOverride", !currentPlanAccepted() && accepted);
        if (currentPlanReport != null) {
            details.put("fingerprint", currentPlanReport.fingerprint().toMetadata());
            details.put("warnings", currentPlanReport.warnings());
            details.put("rejectionReasons", currentPlanReport.readiness().rejectionReasons());
        }
        return gate(
                DiscreteTokenDatasetCheckpointResumeGate.CURRENT_PLAN,
                "current-dataset",
                accepted,
                gateSeverity(currentPlanAccepted(), accepted),
                !currentPlanChecked()
                        ? "current dataset plan was not checked"
                        : currentPlanAccepted()
                                ? "current dataset plan accepted"
                                : accepted
                                        ? "compatibility mode accepted current dataset plan status: "
                                                + currentPlanGateStatus()
                                        : "current dataset plan rejected: " + currentPlanGateStatus(),
                details);
    }

    private static DiscreteTokenDatasetCheckpointResumeGate gate(
            String id,
            String category,
            boolean accepted,
            String severity,
            String summary,
            Map<String, Object> details) {
        return new DiscreteTokenDatasetCheckpointResumeGate(id, category, accepted, severity, summary, details);
    }

    private static String gateSeverity(boolean rawAccepted, boolean effectiveAccepted) {
        if (rawAccepted) {
            return "info";
        }
        return effectiveAccepted ? "warning" : "error";
    }

}

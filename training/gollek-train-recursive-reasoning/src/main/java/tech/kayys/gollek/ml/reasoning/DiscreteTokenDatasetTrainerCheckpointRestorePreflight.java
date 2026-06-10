package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Current-plan resume preflight for a selected recursive-reasoning restore target.
 */
public record DiscreteTokenDatasetTrainerCheckpointRestorePreflight(
        DiscreteTokenDatasetTrainerCheckpointRestorePlan restorePlan,
        DiscreteTokenDatasetCheckpointResumePolicy resumePolicy,
        DiscreteTokenDatasetCheckpointResumeReport resumeReport) {

    public DiscreteTokenDatasetTrainerCheckpointRestorePreflight {
        restorePlan = Objects.requireNonNull(restorePlan, "restorePlan must not be null");
        resumePolicy = Objects.requireNonNull(resumePolicy, "resumePolicy must not be null");
        resumeReport = Objects.requireNonNull(resumeReport, "resumeReport must not be null");
        verifyReportMatchesRestorePlan(restorePlan, resumeReport);
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight evaluate(
            DiscreteTokenDatasetTrainerCheckpointRestorePlan restorePlan,
            DiscreteTokenDatasetPlan currentPlan,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) {
        Objects.requireNonNull(restorePlan, "restorePlan must not be null");
        Objects.requireNonNull(resumePolicy, "resumePolicy must not be null");
        return new DiscreteTokenDatasetTrainerCheckpointRestorePreflight(
                restorePlan,
                resumePolicy,
                resumePolicy.evaluate(restorePlan.manifest(), currentPlan));
    }

    public static DiscreteTokenDatasetTrainerCheckpointRestorePreflight evaluate(
            DiscreteTokenDatasetTrainerCheckpointRestorePlan restorePlan,
            DiscreteTokenDatasetPlanReport currentPlanReport,
            DiscreteTokenDatasetCheckpointResumePolicy resumePolicy) {
        Objects.requireNonNull(restorePlan, "restorePlan must not be null");
        Objects.requireNonNull(resumePolicy, "resumePolicy must not be null");
        return new DiscreteTokenDatasetTrainerCheckpointRestorePreflight(
                restorePlan,
                resumePolicy,
                resumePolicy.evaluate(restorePlan.manifest(), currentPlanReport));
    }

    public boolean ready() {
        return resumeReport.ready();
    }

    public String status() {
        return resumeReport.status();
    }

    public List<String> rejectionReasons() {
        return resumeReport.rejectionReasons();
    }

    public DiscreteTokenDatasetCheckpointResumeExplanation explanation() {
        return resumeReport.explanation();
    }

    public List<DiscreteTokenDatasetCheckpointResumeGate> gates() {
        return resumeReport.gates();
    }

    public DiscreteTokenDatasetCheckpointResumeGateSummary gateSummary() {
        return resumeReport.gateSummary();
    }

    public DiscreteTokenDatasetCheckpointResumeActionPlan actionPlan() {
        return resumeReport.actionPlan();
    }

    public DiscreteTokenDatasetCheckpointResumeReadinessBadge readinessBadge() {
        return resumeReport.readinessBadge();
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> nextActions() {
        return resumeReport.nextActions();
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeAction> primaryAction() {
        return resumeReport.primaryAction();
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> requiredActions() {
        return resumeReport.requiredActions();
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> warningActions() {
        return resumeReport.warningActions();
    }

    public List<String> actionCodes() {
        return resumeReport.actionCodes();
    }

    public List<String> requiredActionCodes() {
        return resumeReport.requiredActionCodes();
    }

    public List<String> warningActionCodes() {
        return resumeReport.warningActionCodes();
    }

    public DiscreteTokenDatasetTrainerCheckpointRestorePreflight requireReady() {
        resumeReport.requireReady();
        return this;
    }

    public DiscreteTokenDatasetTrainerCheckpointRestorePreflight requireAllGatesAccepted() {
        resumeReport.requireAllGatesAccepted();
        return this;
    }

    public DiscreteTokenDatasetTrainerCheckpointRestorePreflight requireNoRequiredActions() {
        resumeReport.requireNoRequiredActions();
        return this;
    }

    public DiscreteTokenDatasetTrainerCheckpointRestorePreflight requireMetadataMatch(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        Map<?, ?> currentResumeReport =
                DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "currentResumeReport");
        DiscreteTokenDatasetCheckpointResumeExplanation.fromMetadata(
                DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "explanation"));
        DiscreteTokenDatasetCheckpointResumeGateSummary.fromMetadata(
                DiscreteTokenDatasetMetadataSupport.requiredMap(currentResumeReport, "gateSummary"),
                resumeReport.gates());
        DiscreteTokenDatasetCheckpointResumeActionPlan.fromMetadata(
                DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "actionPlan"));
        DiscreteTokenDatasetCheckpointResumeReadinessBadge.fromMetadata(
                DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "readinessBadge"));
        DiscreteTokenDatasetCheckpointResumeActionPlan.fromMetadata(
                DiscreteTokenDatasetMetadataSupport.requiredMap(currentResumeReport, "actionPlan"));
        DiscreteTokenDatasetCheckpointResumeReadinessBadge.fromMetadata(
                DiscreteTokenDatasetMetadataSupport.requiredMap(currentResumeReport, "readinessBadge"));

        Map<String, Object> expected = toMetadata();
        for (Map.Entry<String, Object> entry : expected.entrySet()) {
            Object actual = DiscreteTokenDatasetMetadataSupport.required(metadata, entry.getKey());
            if (!DiscreteTokenDatasetMetadataSupport.metadataValueMatches(entry.getValue(), actual)) {
                throw new IllegalArgumentException(
                        "restore preflight field '" + entry.getKey() + "' does not match resume report");
            }
        }
        return this;
    }

    public String summary() {
        return "restore preflight "
                + status()
                + ": "
                + restorePlan.runId()
                + " step "
                + restorePlan.checkpointStep()
                + " from "
                + restorePlan.checkpointDir();
    }

    public Map<String, Object> currentResumeReportMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", resumeReport.status());
        metadata.put("ready", resumeReport.ready());
        metadata.put("runId", resumeReport.checkpoint().runId());
        metadata.put("checkpointStep", resumeReport.checkpoint().checkpointStep());
        metadata.put("schemaAccepted", resumeReport.schemaAccepted());
        metadata.put("datasetAccepted", resumeReport.datasetAccepted());
        metadata.put("fingerprintMatched", resumeReport.fingerprintMatched());
        metadata.put("expectationAccepted", resumeReport.expectationAccepted());
        metadata.put("currentPlanChecked", resumeReport.currentPlanChecked());
        metadata.put("currentPlanAccepted", resumeReport.currentPlanAccepted());
        metadata.put("currentPlanGateStatus", resumeReport.currentPlanGateStatus());
        metadata.put("compatibilityMode", resumeReport.compatibilityMode().id());
        metadata.put("forceAccepted", resumeReport.forceAccepted());
        metadata.put("compatibilityWarnings", resumeReport.compatibilityWarnings());
        metadata.put("gateSummary", resumeReport.gateSummary().toMetadata());
        metadata.put("actionPlan", resumeReport.actionPlan().toMetadata());
        metadata.put("readinessBadge", resumeReport.readinessBadge().toMetadata());
        metadata.put("gates", resumeReport.gates().stream()
                .map(DiscreteTokenDatasetCheckpointResumeGate::toMetadata)
                .toList());
        metadata.put("policyTracked", resumeReport.policyTracked());
        metadata.put("rejectionReasons", resumeReport.rejectionReasons());
        metadata.put("expectation", resumeReport.expectation().toMetadata());
        metadata.put("fingerprintMatch", resumeReport.fingerprintMatch().toMetadata());
        if (resumeReport.currentPlanReport() != null) {
            metadata.put("currentPlanReport", currentPlanReportMetadata(resumeReport.currentPlanReport()));
        }
        if (resumeReport.policyTracked()) {
            metadata.put("policy", resumeReport.policyMetadata());
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public Map<String, Object> restorePlanMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("summary", restorePlan.summary());
        metadata.put("rootDir", restorePlan.rootDir().toString());
        metadata.put("checkpointDir", restorePlan.checkpointDir().toString());
        metadata.put("status", restorePlan.status());
        metadata.put("ready", restorePlan.ready());
        metadata.put("resumeReportPresent", restorePlan.resumeReportPresent());
        metadata.put("resumeReady", restorePlan.resumeReady());
        metadata.put("paths", restorePlan.pathsMetadata());
        metadata.put("identity", restorePlan.identityMetadata());
        metadata.put("checkpoint", restorePlan.checkpointMetadata());
        metadata.put("policy", restorePlan.policy().toMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("summary", summary());
        metadata.put("status", status());
        metadata.put("ready", ready());
        metadata.put("rejectionReasons", rejectionReasons());
        metadata.put("explanation", explanation().toMetadata());
        metadata.put("actionPlan", actionPlan().toMetadata());
        metadata.put("readinessBadge", readinessBadge().toMetadata());
        if (resumeReport.currentPlanReport() != null) {
            metadata.put("currentPlanReport", resumeReport.currentPlanReport().toMetadata());
        }
        metadata.put("restorePlan", restorePlanMetadata());
        metadata.put("resumePolicy", resumePolicy.toMetadata());
        metadata.put("currentResumeReport", currentResumeReportMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static Map<String, Object> currentPlanReportMetadata(DiscreteTokenDatasetPlanReport report) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("fingerprint", report.fingerprint().toMetadata());
        metadata.put("accepted", report.accepted());
        metadata.put("gateStatus", report.gateStatus());
        metadata.put("warnings", report.warnings());
        metadata.put("summary", report.summary());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void verifyReportMatchesRestorePlan(
            DiscreteTokenDatasetTrainerCheckpointRestorePlan restorePlan,
            DiscreteTokenDatasetCheckpointResumeReport resumeReport) {
        DiscreteTokenDatasetCheckpointManifestSnapshot manifest = restorePlan.manifest();
        DiscreteTokenDatasetCheckpointManifestSnapshot reportCheckpoint = resumeReport.checkpoint();
        if (!manifest.schemaVersion().equals(reportCheckpoint.schemaVersion())
                || !manifest.runId().equals(reportCheckpoint.runId())
                || manifest.checkpointStep() != reportCheckpoint.checkpointStep()
                || !manifest.fingerprint().equals(reportCheckpoint.fingerprint())
                || manifest.datasetAccepted() != reportCheckpoint.datasetAccepted()
                || !manifest.datasetGateStatus().equals(reportCheckpoint.datasetGateStatus())) {
            throw new IllegalArgumentException(
                    "restore preflight resume report must describe the selected restore checkpoint");
        }
    }
}

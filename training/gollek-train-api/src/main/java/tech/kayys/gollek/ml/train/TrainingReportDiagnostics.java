package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Evidence-based diagnostics for canonical trainer reports.
 */
public final class TrainingReportDiagnostics {
    public static final Options DEFAULT_OPTIONS = new Options(
            1.25,
            0.05,
            1.0e-8,
            0.01,
            3,
            0.999999,
            1.0e-12,
            1.0,
            1.0e-12,
            10.0,
            1.0e-6);

    private TrainingReportDiagnostics() {
    }

    public enum Severity {
        INFO,
        WARNING,
        CRITICAL
    }

    public record Options(
            double overfitValidationToTrainLossRatio,
            double overfitGapDelta,
            double meaningfulLossDelta,
            double trainLossPlateauRelativeDelta,
            int trainLossPlateauMinEpochs,
            double optimizationZeroGradientFractionThreshold,
            double optimizationTinyNormThreshold,
            double optimizationMaxUpdateToParameterRatio,
            double learningRateTinyThreshold,
            double learningRateSpikeRatioThreshold,
            double learningRateFlatRelativeDeltaThreshold) {
        public Options(
                double overfitValidationToTrainLossRatio,
                double overfitGapDelta,
                double meaningfulLossDelta,
                double trainLossPlateauRelativeDelta,
                int trainLossPlateauMinEpochs) {
            this(
                    overfitValidationToTrainLossRatio,
                    overfitGapDelta,
                    meaningfulLossDelta,
                    trainLossPlateauRelativeDelta,
                    trainLossPlateauMinEpochs,
                    0.999999,
                    1.0e-12,
                    1.0,
                    1.0e-12,
                    10.0,
                    1.0e-6);
        }

        public Options(
                double overfitValidationToTrainLossRatio,
                double overfitGapDelta,
                double meaningfulLossDelta,
                double trainLossPlateauRelativeDelta,
                int trainLossPlateauMinEpochs,
                double optimizationZeroGradientFractionThreshold,
                double optimizationTinyNormThreshold,
                double optimizationMaxUpdateToParameterRatio) {
            this(
                    overfitValidationToTrainLossRatio,
                    overfitGapDelta,
                    meaningfulLossDelta,
                    trainLossPlateauRelativeDelta,
                    trainLossPlateauMinEpochs,
                    optimizationZeroGradientFractionThreshold,
                    optimizationTinyNormThreshold,
                    optimizationMaxUpdateToParameterRatio,
                    1.0e-12,
                    10.0,
                    1.0e-6);
        }

        public Options {
            overfitValidationToTrainLossRatio = positiveOrDefault(overfitValidationToTrainLossRatio, 1.25);
            overfitGapDelta = nonNegativeOrDefault(overfitGapDelta, 0.05);
            meaningfulLossDelta = nonNegativeOrDefault(meaningfulLossDelta, 1.0e-8);
            trainLossPlateauRelativeDelta = nonNegativeOrDefault(trainLossPlateauRelativeDelta, 0.01);
            trainLossPlateauMinEpochs = Math.max(2, trainLossPlateauMinEpochs);
            optimizationZeroGradientFractionThreshold = fractionOrDefault(
                    optimizationZeroGradientFractionThreshold,
                    0.999999);
            optimizationTinyNormThreshold = nonNegativeOrDefault(optimizationTinyNormThreshold, 1.0e-12);
            optimizationMaxUpdateToParameterRatio = positiveOrDefault(optimizationMaxUpdateToParameterRatio, 1.0);
            learningRateTinyThreshold = nonNegativeOrDefault(learningRateTinyThreshold, 1.0e-12);
            learningRateSpikeRatioThreshold = positiveOrDefault(learningRateSpikeRatioThreshold, 10.0);
            learningRateFlatRelativeDeltaThreshold = nonNegativeOrDefault(
                    learningRateFlatRelativeDeltaThreshold,
                    1.0e-6);
        }
    }

    public record Finding(
            Severity severity,
            String code,
            String message,
            Map<String, Object> evidence) {
        public Finding {
            severity = Objects.requireNonNull(severity, "severity must not be null");
            if (code == null || code.isBlank()) {
                throw new IllegalArgumentException("code must not be blank");
            }
            if (message == null || message.isBlank()) {
                throw new IllegalArgumentException("message must not be blank");
            }
            code = code.trim();
            message = message.trim();
            evidence = immutableEvidence(evidence);
        }
    }

    public record Summary(
            boolean available,
            int total,
            Map<Severity, Integer> bySeverity,
            String highestSeverity,
            List<String> codes) {
        public Summary {
            total = Math.max(0, total);
            bySeverity = normalizeCounts(bySeverity);
            highestSeverity = normalizeHighest(highestSeverity, bySeverity);
            codes = immutableStringList(codes);
        }

        public static Summary fromMap(Map<String, ?> summary) {
            Objects.requireNonNull(summary, "summary must not be null");
            Map<Severity, Integer> counts = countsFromObject(summary.get("bySeverity"));
            String highest = stringValue(summary.get("highestSeverity"), highestSeverityName(counts));
            return new Summary(
                    booleanValue(summary.get("available"), true),
                    intValue(summary.get("total"), sumCounts(counts)),
                    counts,
                    highest,
                    listFromObject(summary.get("codes")));
        }

        public int count(Severity severity) {
            Objects.requireNonNull(severity, "severity must not be null");
            return bySeverity.getOrDefault(severity, 0);
        }

        public boolean hasInfo() {
            return count(Severity.INFO) > 0;
        }

        public boolean hasWarnings() {
            return count(Severity.WARNING) > 0;
        }

        public boolean hasCritical() {
            return count(Severity.CRITICAL) > 0;
        }

        public Optional<Severity> highestSeverityValue() {
            Severity severity = severityOrNull(highestSeverity);
            return severity == null ? Optional.empty() : Optional.of(severity);
        }

        public boolean hasSeverityAbove(Severity maxAllowedSeverity) {
            Objects.requireNonNull(maxAllowedSeverity, "maxAllowedSeverity must not be null");
            for (Severity severity : Severity.values()) {
                if (severity.ordinal() > maxAllowedSeverity.ordinal() && count(severity) > 0) {
                    return true;
                }
            }
            return false;
        }

        public boolean passes(Severity maxAllowedSeverity) {
            return !hasSeverityAbove(maxAllowedSeverity);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("available", available);
            summary.put("total", total);
            summary.put("bySeverity", bySeverityToMap(bySeverity));
            summary.put("highestSeverity", highestSeverity);
            summary.put("hasInfo", hasInfo());
            summary.put("hasWarnings", hasWarnings());
            summary.put("hasCritical", hasCritical());
            summary.put("codes", codes);
            return Map.copyOf(summary);
        }
    }

    public record Provenance(
            boolean diagnosticsPersisted,
            boolean diagnosticsSummaryPersisted,
            boolean diagnosticsBackfilled,
            boolean diagnosticsSummaryStale,
            int persistedFindingCount,
            int effectiveFindingCount,
            String persistedHighestSeverity,
            String effectiveHighestSeverity,
            List<String> persistedCodes,
            List<String> effectiveCodes,
            List<String> backfilledCodes) {
        public Provenance {
            persistedFindingCount = Math.max(0, persistedFindingCount);
            effectiveFindingCount = Math.max(0, effectiveFindingCount);
            persistedHighestSeverity = stringValue(persistedHighestSeverity, "NONE");
            effectiveHighestSeverity = stringValue(effectiveHighestSeverity, "NONE");
            persistedCodes = immutableStringList(persistedCodes);
            effectiveCodes = immutableStringList(effectiveCodes);
            backfilledCodes = immutableStringList(backfilledCodes);
            diagnosticsBackfilled = diagnosticsBackfilled || !backfilledCodes.isEmpty();
        }

        public static Provenance fromMap(Map<String, ?> provenance) {
            if (provenance == null || provenance.isEmpty()) {
                return new Provenance(false, false, true, false, 0, 0, "NONE", "NONE", List.of(), List.of(), List.of());
            }
            return new Provenance(
                    booleanValue(provenance.get("diagnosticsPersisted"), false),
                    booleanValue(provenance.get("diagnosticsSummaryPersisted"), false),
                    booleanValue(provenance.get("diagnosticsBackfilled"), false),
                    booleanValue(provenance.get("diagnosticsSummaryStale"), false),
                    TrainingReportValues.intValue(provenance.get("persistedFindingCount"), 0),
                    TrainingReportValues.intValue(provenance.get("effectiveFindingCount"), 0),
                    stringValue(provenance.get("persistedHighestSeverity"), "NONE"),
                    stringValue(provenance.get("effectiveHighestSeverity"), "NONE"),
                    listFromObject(provenance.get("persistedCodes")),
                    listFromObject(provenance.get("effectiveCodes")),
                    listFromObject(provenance.get("backfilledCodes")));
        }

        public String status() {
            if (diagnosticsBackfilled) {
                return "BACKFILLED";
            }
            if (!diagnosticsPersisted) {
                return "DIAGNOSTICS_MISSING";
            }
            if (diagnosticsSummaryStale) {
                return "SUMMARY_STALE";
            }
            if (!diagnosticsSummaryPersisted) {
                return "SUMMARY_RECOMPUTED";
            }
            return "FRESH";
        }

        public boolean fresh() {
            return "FRESH".equals(status());
        }

        public boolean stale() {
            return !fresh();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("status", status());
            map.put("fresh", fresh());
            map.put("stale", stale());
            map.put("diagnosticsPersisted", diagnosticsPersisted);
            map.put("diagnosticsSummaryPersisted", diagnosticsSummaryPersisted);
            map.put("diagnosticsBackfilled", diagnosticsBackfilled);
            map.put("diagnosticsSummaryStale", diagnosticsSummaryStale);
            map.put("persistedFindingCount", persistedFindingCount);
            map.put("effectiveFindingCount", effectiveFindingCount);
            map.put("persistedHighestSeverity", persistedHighestSeverity);
            map.put("effectiveHighestSeverity", effectiveHighestSeverity);
            map.put("persistedCodes", persistedCodes);
            map.put("effectiveCodes", effectiveCodes);
            map.put("backfilledCodes", backfilledCodes);
            return Map.copyOf(map);
        }
    }

    public record GateResult(
            Summary summary,
            Severity maxAllowedSeverity,
            boolean passed,
            List<Finding> failingFindings) {
        public GateResult {
            summary = Objects.requireNonNull(summary, "summary must not be null");
            maxAllowedSeverity = Objects.requireNonNull(maxAllowedSeverity, "maxAllowedSeverity must not be null");
            failingFindings = failingFindings == null ? List.of() : List.copyOf(failingFindings);
            passed = failingFindings.isEmpty();
        }

        public static GateResult fromMap(Map<String, ?> gate) {
            if (gate == null || gate.isEmpty()) {
                return new GateResult(
                        new Summary(true, 0, Map.of(), "NONE", List.of()),
                        Severity.INFO,
                        true,
                        List.of());
            }
            Map<String, ?> summaryMap = mapSectionUntyped(gate, "summary");
            return new GateResult(
                    Summary.fromMap(summaryMap),
                    severityValue(gate.get("maxAllowedSeverity")),
                    booleanValue(gate.get("passed"), true),
                    fromMaps(mapList(gate.get("failingFindings"))));
        }

        public List<String> failingCodes() {
            return failingFindings.stream()
                    .map(Finding::code)
                    .toList();
        }

        public String message() {
            if (passed) {
                return "Training report diagnostics passed with highest severity " + summary.highestSeverity()
                        + " and max allowed severity " + maxAllowedSeverity.name() + ".";
            }
            return "Training report diagnostics failed: highest severity " + summary.highestSeverity()
                    + " exceeds max allowed severity " + maxAllowedSeverity.name()
                    + " for codes " + failingCodes() + ".";
        }

        public void requirePassed() {
            if (!passed) {
                throw new IllegalStateException(message());
            }
        }
    }

    public static List<Finding> analyze(Path reportFile) throws IOException {
        return analyze(TrainingReportReader.readCanonical(reportFile));
    }

    public static List<Finding> analyze(Path reportFile, Options options) throws IOException {
        return analyze(TrainingReportReader.readCanonical(reportFile), options);
    }

    public static List<Finding> analyze(Map<String, ?> report) {
        return analyze(report, DEFAULT_OPTIONS);
    }

    public static List<Finding> withRunHealthFindings(Map<String, ?> report, List<Finding> findings) {
        return withReportHealthFindings(report, findings);
    }

    public static List<Finding> withReportHealthFindings(Map<String, ?> report, List<Finding> findings) {
        Objects.requireNonNull(report, "report must not be null");
        List<Finding> merged = new ArrayList<>(findings == null ? List.of() : findings);
        addRunHealthFinding(merged, report);
        addDataHealthFinding(merged, report);
        return List.copyOf(merged);
    }

    public static List<Finding> analyze(Map<String, ?> report, Options options) {
        Objects.requireNonNull(report, "report must not be null");
        Options resolvedOptions = options == null ? DEFAULT_OPTIONS : options;
        List<Finding> findings = new ArrayList<>();
        addRunHealthFinding(findings, report);
        addDataHealthFinding(findings, report);
        List<Map<String, Object>> history = TrainingReportReader.history(report);
        Map<String, Object> summary = TrainingReportReader.historySummary(report);
        if (history.isEmpty()) {
            findings.add(finding(
                    Severity.INFO,
                    "history.missing",
                    "No epoch history is available in this training report.",
                    Map.of("historyRows", 0)));
            return List.copyOf(findings);
        }

        Map<String, Object> validationLoss = mapSection(summary, "validationLoss");
        if (!available(validationLoss)) {
            findings.add(finding(
                    Severity.INFO,
                    "validation.missing",
                    "No validation loss was recorded, so generalization cannot be diagnosed.",
                    Map.of("historyRows", history.size())));
        } else {
            addValidationLossFinding(findings, validationLoss, resolvedOptions);
            addGeneralizationFinding(findings, summary, resolvedOptions);
        }

        addTrainLossPlateauFinding(findings, mapSection(summary, "trainLoss"), resolvedOptions);
        TrainerLearningRateReportDiagnostics.addFindings(findings, history, summary, resolvedOptions);
        TrainerOptimizationReportDiagnostics.addFindings(findings, history, resolvedOptions);
        return List.copyOf(findings);
    }

    private static void addRunHealthFinding(List<Finding> findings, Map<String, ?> report) {
        TrainingReportRunHealth health = TrainingReportReader.runHealthView(report);
        if (health.gatePassed() && !health.issueDetected() && !health.blockingIssueDetected()) {
            return;
        }

        boolean gateFailed = !health.gatePassed() || health.blockingIssueDetected();
        if (hasRunHealthFinding(findings, gateFailed)) {
            return;
        }
        Map<String, Object> primaryIssue = gateFailed && !health.primaryBlockingIssue().isEmpty()
                ? health.primaryBlockingIssue()
                : health.primaryIssue();
        String primaryMessage = stringValue(primaryIssue.get("message"), health.recommendedAction());
        findings.add(finding(
                gateFailed ? Severity.CRITICAL : Severity.WARNING,
                gateFailed ? "run_health.gate_failed" : "run_health.issue_detected",
                gateFailed
                        ? "Training run health gate failed: " + primaryMessage
                        : "Training run health reported issues: " + primaryMessage,
                runHealthEvidence(health, primaryIssue)));
    }

    private static Map<String, Object> runHealthEvidence(
            TrainingReportRunHealth health,
            Map<String, Object> primaryIssue) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("status", health.status());
        evidence.put("healthy", health.healthy());
        evidence.put("gatePassed", health.gatePassed());
        evidence.put("issueDetected", health.issueDetected());
        evidence.put("issueCount", health.issueCount());
        evidence.put("blockingIssueDetected", health.blockingIssueDetected());
        evidence.put("blockingIssueCount", health.blockingIssueCount());
        evidence.put("recommendedAction", health.recommendedAction());
        evidence.put("issueCodes", health.issueCodes());
        evidence.put("issueSeverities", health.issueSeverities());
        evidence.put("issueCountsByKind", health.issueCountsByKind());
        evidence.put("issueCountsBySeverity", health.issueCountsBySeverity());
        if (!primaryIssue.isEmpty()) {
            evidence.put("primaryIssue", primaryIssue);
            putIfPresent(evidence, "primaryIssueCode", primaryIssue.get("code"));
            putIfPresent(evidence, "primaryIssueSeverity", primaryIssue.get("severity"));
            putIfPresent(evidence, "primaryIssueArtifact", primaryIssue.get("artifact"));
            putIfPresent(evidence, "primaryIssueMessage", primaryIssue.get("message"));
            putIfPresent(evidence, "primaryIssueAction", primaryIssue.get("action"));
        }
        if (!health.primaryBlockingIssue().isEmpty()) {
            evidence.put("primaryBlockingIssue", health.primaryBlockingIssue());
        }
        return evidence;
    }

    private static boolean hasRunHealthFinding(List<Finding> findings, boolean gateFailed) {
        if (hasCode(findings, "run_health.gate_failed")) {
            return true;
        }
        return !gateFailed && hasCode(findings, "run_health.issue_detected");
    }

    private static void addDataHealthFinding(List<Finding> findings, Map<String, ?> report) {
        TrainingReportDataHealth health = TrainingReportReader.dataHealthView(report);
        if (health.gatePassed() && !health.issueDetected()) {
            return;
        }

        boolean gateFailed = !health.gatePassed() || health.errorCount() > 0;
        String code = gateFailed ? "data_health.gate_failed" : "data_health.issue_detected";
        if (hasCode(findings, code) || (gateFailed && hasCode(findings, "data_health.gate_failed"))) {
            return;
        }

        Map<String, Object> primaryIssue = primaryDataHealthIssue(health, gateFailed);
        String primaryMessage = stringValue(primaryIssue.get("message"), firstRecommendedAction(health));
        findings.add(finding(
                gateFailed ? Severity.CRITICAL : Severity.WARNING,
                code,
                gateFailed
                        ? "Trainer data health gate failed: " + primaryMessage
                        : "Trainer data health reported issues: " + primaryMessage,
                dataHealthEvidence(health, primaryIssue)));
    }

    private static Map<String, Object> primaryDataHealthIssue(
            TrainingReportDataHealth health,
            boolean preferBlocking) {
        if (health.issues().isEmpty()) {
            return Map.of();
        }
        if (preferBlocking) {
            for (Map<String, Object> issue : health.issues()) {
                String severity = stringValue(issue.get("severity"), "").toLowerCase();
                if ("error".equals(severity) || TrainingReportValues.booleanValue(issue.get("blocking"))) {
                    return issue;
                }
            }
        }
        return health.issues().get(0);
    }

    private static String firstRecommendedAction(TrainingReportDataHealth health) {
        if (!health.recommendedActions().isEmpty()) {
            return health.recommendedActions().get(0);
        }
        return "inspect trainer data-loader and distribution health before promotion";
    }

    private static Map<String, Object> dataHealthEvidence(
            TrainingReportDataHealth health,
            Map<String, Object> primaryIssue) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("available", health.available());
        evidence.put("healthy", health.healthy());
        evidence.put("gatePassed", health.gatePassed());
        evidence.put("issueDetected", health.issueDetected());
        evidence.put("issueCount", health.issueCount());
        evidence.put("warningCount", health.warningCount());
        evidence.put("errorCount", health.errorCount());
        evidence.put("issueCodes", health.issueCodes());
        evidence.put("issueSeverities", health.issueSeverities());
        evidence.put("recommendedActions", health.recommendedActions());
        evidence.put("loaderPlan", health.loaderPlan().toMap());
        evidence.put("distribution", health.distribution().toMap());
        if (!primaryIssue.isEmpty()) {
            evidence.put("primaryIssue", primaryIssue);
            putIfPresent(evidence, "primaryIssueCode", primaryIssue.get("code"));
            putIfPresent(evidence, "primaryIssueSeverity", primaryIssue.get("severity"));
            putIfPresent(evidence, "primaryIssueMessage", primaryIssue.get("message"));
            putIfPresent(evidence, "primaryIssueAction", primaryIssue.get("action"));
        }
        return evidence;
    }

    private static boolean hasCode(List<Finding> findings, String code) {
        return findings.stream().anyMatch(finding -> code.equals(finding.code()));
    }

    public static GateResult gate(Path reportFile, Severity maxAllowedSeverity) throws IOException {
        return gateFindings(analyze(reportFile), maxAllowedSeverity);
    }

    public static GateResult gate(Map<String, ?> report, Severity maxAllowedSeverity) {
        return gateFindings(analyze(report), maxAllowedSeverity);
    }

    public static GateResult gateFindings(List<Finding> findings, Severity maxAllowedSeverity) {
        Objects.requireNonNull(maxAllowedSeverity, "maxAllowedSeverity must not be null");
        List<Finding> safeFindings = findings == null ? List.of() : List.copyOf(findings);
        List<Finding> failing = safeFindings.stream()
                .filter(finding -> finding.severity().ordinal() > maxAllowedSeverity.ordinal())
                .toList();
        return new GateResult(summarize(safeFindings), maxAllowedSeverity, failing.isEmpty(), failing);
    }

    public static GateResult gateMaps(List<Map<String, Object>> diagnostics, Severity maxAllowedSeverity) {
        return gateFindings(fromMaps(diagnostics), maxAllowedSeverity);
    }

    public static List<Map<String, Object>> toMaps(List<Finding> findings) {
        if (findings == null || findings.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> rows = new ArrayList<>(findings.size());
        for (Finding finding : findings) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("severity", finding.severity().name());
            row.put("code", finding.code());
            row.put("message", finding.message());
            row.put("evidence", finding.evidence());
            rows.add(Map.copyOf(row));
        }
        return List.copyOf(rows);
    }

    public static List<Finding> fromMaps(List<Map<String, Object>> diagnostics) {
        if (diagnostics == null || diagnostics.isEmpty()) {
            return List.of();
        }
        List<Finding> findings = new ArrayList<>(diagnostics.size());
        for (Map<String, Object> diagnostic : diagnostics) {
            findings.add(fromMap(diagnostic));
        }
        return List.copyOf(findings);
    }

    public static Summary summarize(List<Finding> findings) {
        if (findings == null || findings.isEmpty()) {
            return summaryFromCountsTyped(0, Map.of(), List.of());
        }
        Map<String, Integer> counts = new LinkedHashMap<>();
        List<String> codes = new ArrayList<>();
        for (Finding finding : findings) {
            String severity = finding.severity().name();
            counts.put(severity, counts.getOrDefault(severity, 0) + 1);
            codes.add(finding.code());
        }
        return summaryFromCountsTyped(findings.size(), counts, codes);
    }

    public static Map<String, Object> summary(List<Finding> findings) {
        return summarize(findings).toMap();
    }

    public static Summary summarizeFromMaps(List<Map<String, Object>> diagnostics) {
        if (diagnostics == null || diagnostics.isEmpty()) {
            return summaryFromCountsTyped(0, Map.of(), List.of());
        }
        Map<String, Integer> counts = new LinkedHashMap<>();
        List<String> codes = new ArrayList<>();
        for (Map<String, Object> diagnostic : diagnostics) {
            String severity = normalizeSeverity(diagnostic.get("severity"));
            counts.put(severity, counts.getOrDefault(severity, 0) + 1);
            Object code = diagnostic.get("code");
            if (code != null) {
                codes.add(String.valueOf(code));
            }
        }
        return summaryFromCountsTyped(diagnostics.size(), counts, codes);
    }

    public static Map<String, Object> summaryFromMaps(List<Map<String, Object>> diagnostics) {
        return summarizeFromMaps(diagnostics).toMap();
    }

    private static void addValidationLossFinding(
            List<Finding> findings,
            Map<String, Object> validationLoss,
            Options options) {
        OptionalDouble delta = optionalDouble(validationLoss.get("deltaFromFirst"));
        if ("worsened".equals(validationLoss.get("trend"))
                && delta.isPresent()
                && delta.getAsDouble() > options.meaningfulLossDelta()) {
            Map<String, Object> evidence = new LinkedHashMap<>();
            putIfPresent(evidence, "first", validationLoss.get("first"));
            putIfPresent(evidence, "latest", validationLoss.get("latest"));
            putIfPresent(evidence, "deltaFromFirst", validationLoss.get("deltaFromFirst"));
            putIfPresent(evidence, "best", validationLoss.get("best"));
            putIfPresent(evidence, "bestEpoch", validationLoss.get("bestEpoch"));
            findings.add(finding(
                    Severity.WARNING,
                    "validation.loss_worsened",
                    "Validation loss increased across the recorded training history.",
                    evidence));
        }
    }

    private static void addGeneralizationFinding(
            List<Finding> findings,
            Map<String, Object> summary,
            Options options) {
        Map<String, Object> generalization = mapSection(summary, "generalization");
        if (!available(generalization)) {
            return;
        }
        OptionalDouble ratio = optionalDouble(generalization.get("latestValidationToTrainLossRatio"));
        OptionalDouble gapDelta = optionalDouble(generalization.get("gapDeltaFromFirst"));
        boolean gapIncreasing = TrainingReportValues.booleanValue(generalization.get("gapIncreasing"));
        boolean ratioHigh = ratio.isPresent()
                && ratio.getAsDouble() >= options.overfitValidationToTrainLossRatio();
        boolean gapHigh = gapDelta.isPresent()
                && gapDelta.getAsDouble() >= options.overfitGapDelta();
        if (gapIncreasing && ratioHigh && gapHigh) {
            Map<String, Object> evidence = new LinkedHashMap<>();
            putIfPresent(evidence, "latestGap", generalization.get("latestGap"));
            putIfPresent(evidence, "gapDeltaFromFirst", generalization.get("gapDeltaFromFirst"));
            putIfPresent(evidence, "latestValidationToTrainLossRatio",
                    generalization.get("latestValidationToTrainLossRatio"));
            putIfPresent(evidence, "maxGapEpoch", generalization.get("maxGapEpoch"));
            findings.add(finding(
                    Severity.WARNING,
                    "generalization.overfit_risk",
                    "Validation loss is pulling away from training loss.",
                    evidence));
        }
    }

    private static void addTrainLossPlateauFinding(
            List<Finding> findings,
            Map<String, Object> trainLoss,
            Options options) {
        if (!available(trainLoss)) {
            return;
        }
        int count = intValue(trainLoss.get("count"), 0);
        OptionalDouble first = optionalDouble(trainLoss.get("first"));
        OptionalDouble delta = optionalDouble(trainLoss.get("deltaFromFirst"));
        if (count < options.trainLossPlateauMinEpochs()
                || first.isEmpty()
                || delta.isEmpty()) {
            return;
        }
        double scale = Math.max(Math.abs(first.getAsDouble()), 1.0e-12);
        double relativeDelta = Math.abs(delta.getAsDouble()) / scale;
        if (relativeDelta <= options.trainLossPlateauRelativeDelta()) {
            Map<String, Object> evidence = new LinkedHashMap<>();
            evidence.put("count", count);
            putIfPresent(evidence, "first", trainLoss.get("first"));
            putIfPresent(evidence, "latest", trainLoss.get("latest"));
            evidence.put("relativeDeltaFromFirst", relativeDelta);
            findings.add(finding(
                    Severity.INFO,
                    "train.loss_plateau",
                    "Training loss changed only marginally across the recorded history.",
                    evidence));
        }
    }

    private static Finding finding(
            Severity severity,
            String code,
            String message,
            Map<String, Object> evidence) {
        return new Finding(severity, code, message, evidence);
    }

    private static Finding fromMap(Map<String, Object> diagnostic) {
        Objects.requireNonNull(diagnostic, "diagnostic must not be null");
        Severity severity = severityValue(diagnostic.get("severity"));
        String code = stringValue(diagnostic.get("code"), "unknown");
        String message = stringValue(diagnostic.get("message"), "No diagnostic message available.");
        Map<String, Object> evidence = mapSection(diagnostic, "evidence");
        return new Finding(severity, code, message, evidence);
    }

    private static Summary summaryFromCountsTyped(
            int total,
            Map<String, Integer> counts,
            List<String> codes) {
        Map<Severity, Integer> bySeverity = new EnumMap<>(Severity.class);
        for (Severity severity : Severity.values()) {
            bySeverity.put(severity, counts.getOrDefault(severity.name(), 0));
        }
        return new Summary(true, total, bySeverity, highestSeverityName(bySeverity), codes);
    }

    private static String highestSeverityName(Map<Severity, Integer> bySeverity) {
        if (bySeverity.getOrDefault(Severity.CRITICAL, 0) > 0) {
            return Severity.CRITICAL.name();
        }
        if (bySeverity.getOrDefault(Severity.WARNING, 0) > 0) {
            return Severity.WARNING.name();
        }
        if (bySeverity.getOrDefault(Severity.INFO, 0) > 0) {
            return Severity.INFO.name();
        }
        return "NONE";
    }

    private static Map<Severity, Integer> normalizeCounts(Map<Severity, Integer> counts) {
        Map<Severity, Integer> normalized = new EnumMap<>(Severity.class);
        for (Severity severity : Severity.values()) {
            normalized.put(severity, Math.max(0, counts == null ? 0 : counts.getOrDefault(severity, 0)));
        }
        return Map.copyOf(normalized);
    }

    private static Map<Severity, Integer> countsFromObject(Object value) {
        Map<Severity, Integer> counts = new EnumMap<>(Severity.class);
        for (Severity severity : Severity.values()) {
            counts.put(severity, 0);
        }
        if (value instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                Severity severity = severityOrNull(entry.getKey());
                if (severity != null) {
                    counts.put(severity, Math.max(0, intValue(entry.getValue(), 0)));
                }
            }
        }
        return Map.copyOf(counts);
    }

    private static Map<String, Integer> bySeverityToMap(Map<Severity, Integer> counts) {
        Map<String, Integer> bySeverity = new LinkedHashMap<>();
        for (Severity severity : Severity.values()) {
            bySeverity.put(severity.name(), Math.max(0, counts.getOrDefault(severity, 0)));
        }
        return Map.copyOf(bySeverity);
    }

    private static String normalizeHighest(String highestSeverity, Map<Severity, Integer> counts) {
        String text = highestSeverity == null ? "" : highestSeverity.trim().toUpperCase();
        if ("NONE".equals(text)) {
            return "NONE";
        }
        Severity severity = severityOrNull(text);
        return severity == null ? highestSeverityName(counts) : severity.name();
    }

    private static Severity severityOrNull(Object value) {
        if (value instanceof Severity severity) {
            return severity;
        }
        String text = String.valueOf(value).trim().toUpperCase();
        for (Severity severity : Severity.values()) {
            if (severity.name().equals(text)) {
                return severity;
            }
        }
        return null;
    }

    private static int sumCounts(Map<Severity, Integer> counts) {
        int total = 0;
        for (Severity severity : Severity.values()) {
            total += Math.max(0, counts.getOrDefault(severity, 0));
        }
        return total;
    }

    private static boolean booleanValue(Object value, boolean fallback) {
        if (value instanceof Boolean || value instanceof String) {
            return TrainingReportValues.booleanValue(value);
        }
        return fallback;
    }

    private static List<String> immutableStringList(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        List<String> normalized = new ArrayList<>(values.size());
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                normalized.add(value.trim());
            }
        }
        return List.copyOf(normalized);
    }

    private static List<String> listFromObject(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (Object item : iterable) {
            if (item != null) {
                String text = String.valueOf(item).trim();
                if (!text.isEmpty()) {
                    values.add(text);
                }
            }
        }
        return List.copyOf(values);
    }

    private static String normalizeSeverity(Object value) {
        return severityValue(value).name();
    }

    private static Severity severityValue(Object value) {
        String text = String.valueOf(value).trim().toUpperCase();
        for (Severity severity : Severity.values()) {
            if (severity.name().equals(text)) {
                return severity;
            }
        }
        return Severity.INFO;
    }

    private static String stringValue(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? fallback : text;
    }

    private static boolean available(Map<String, Object> section) {
        return TrainingReportValues.booleanValue(section.get("available"));
    }

    private static Map<String, Object> mapSection(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        return immutableEvidence(map);
    }

    private static Map<String, Object> mapSectionUntyped(Map<String, ?> source, String key) {
        Object value = source.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        return immutableEvidence(map);
    }

    private static List<Map<String, Object>> mapList(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object item : iterable) {
            if (item instanceof Map<?, ?> map) {
                rows.add(immutableEvidence(map));
            }
        }
        return List.copyOf(rows);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableEvidence(Map<?, ?> evidence) {
        if (evidence == null || evidence.isEmpty()) {
            return Map.of();
        }
        Object snapshot = TrainingReportSnapshots.immutableSnapshot(evidence);
        if (snapshot instanceof Map<?, ?> map) {
            return (Map<String, Object>) map;
        }
        return Map.of();
    }

    private static void putIfPresent(Map<String, Object> target, String key, Object value) {
        if (value != null) {
            target.put(key, value);
        }
    }

    private static double positiveOrDefault(double value, double fallback) {
        return Double.isFinite(value) && value > 0.0 ? value : fallback;
    }

    private static double nonNegativeOrDefault(double value, double fallback) {
        return Double.isFinite(value) && value >= 0.0 ? value : fallback;
    }

    private static double fractionOrDefault(double value, double fallback) {
        return Double.isFinite(value) && value >= 0.0 && value <= 1.0 ? value : fallback;
    }
}

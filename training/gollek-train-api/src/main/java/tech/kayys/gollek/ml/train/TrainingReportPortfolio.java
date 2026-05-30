package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Leaderboard-style view for selecting the best candidate from multiple training reports.
 */
public record TrainingReportPortfolio(List<Entry> entries) {
    public TrainingReportPortfolio {
        entries = entries == null ? List.of() : List.copyOf(entries);
    }

    public enum PromotionStatus {
        PROMOTE,
        HOLD,
        NO_ELIGIBLE_CANDIDATE
    }

    public record PromotionPolicy(
            TrainingReportDiagnostics.Severity maxCandidateDiagnosticSeverity,
            TrainingReportDiagnostics.Severity maxComparisonFindingSeverity,
            double minimumValidationImprovement,
            boolean requireTrackedMetricImprovement) {
        public PromotionPolicy {
            maxCandidateDiagnosticSeverity = maxCandidateDiagnosticSeverity == null
                    ? TrainingReportDiagnostics.Severity.INFO
                    : maxCandidateDiagnosticSeverity;
            maxComparisonFindingSeverity = maxComparisonFindingSeverity == null
                    ? TrainingReportDiagnostics.Severity.INFO
                    : maxComparisonFindingSeverity;
            minimumValidationImprovement = Double.isFinite(minimumValidationImprovement)
                    && minimumValidationImprovement >= 0.0
                            ? minimumValidationImprovement
                            : TrainingReportComparison.DEFAULT_OPTIONS.lossRegressionTolerance();
        }

        public static PromotionPolicy defaultPolicy() {
            return new PromotionPolicy(
                    TrainingReportDiagnostics.Severity.INFO,
                    TrainingReportDiagnostics.Severity.INFO,
                    TrainingReportComparison.DEFAULT_OPTIONS.lossRegressionTolerance(),
                    true);
        }

        public PromotionPolicy withMaxCandidateDiagnosticSeverity(
                TrainingReportDiagnostics.Severity severity) {
            return new PromotionPolicy(
                    severity,
                    maxComparisonFindingSeverity,
                    minimumValidationImprovement,
                    requireTrackedMetricImprovement);
        }

        public PromotionPolicy withMaxComparisonFindingSeverity(
                TrainingReportDiagnostics.Severity severity) {
            return new PromotionPolicy(
                    maxCandidateDiagnosticSeverity,
                    severity,
                    minimumValidationImprovement,
                    requireTrackedMetricImprovement);
        }

        public PromotionPolicy withMinimumValidationImprovement(double improvement) {
            return new PromotionPolicy(
                    maxCandidateDiagnosticSeverity,
                    maxComparisonFindingSeverity,
                    improvement,
                    requireTrackedMetricImprovement);
        }

        public PromotionPolicy withRequireTrackedMetricImprovement(boolean required) {
            return new PromotionPolicy(
                    maxCandidateDiagnosticSeverity,
                    maxComparisonFindingSeverity,
                    minimumValidationImprovement,
                    required);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("maxCandidateDiagnosticSeverity", maxCandidateDiagnosticSeverity.name());
            map.put("maxComparisonFindingSeverity", maxComparisonFindingSeverity.name());
            map.put("minimumValidationImprovement", minimumValidationImprovement);
            map.put("requireTrackedMetricImprovement", requireTrackedMetricImprovement);
            return Map.copyOf(map);
        }
    }

    public record Entry(
            String name,
            TrainingReport report,
            Path source,
            Long sourceBytes,
            String sourceSha256) {
        public Entry {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("name must not be blank");
            }
            name = name.trim();
            report = Objects.requireNonNull(report, "report must not be null");
            source = source == null ? null : source.toAbsolutePath().normalize();
            if (sourceBytes != null && sourceBytes.longValue() < 0L) {
                throw new IllegalArgumentException("sourceBytes must be non-negative");
            }
            sourceSha256 = normalizeSha256(sourceSha256, "sourceSha256");
        }

        public Entry(String name, TrainingReport report, Path source) {
            this(name, report, source, null, null);
        }

        public OptionalDouble validationScore() {
            OptionalDouble best = report.bestValidationLoss();
            return best.isPresent() ? best : report.latestValidationLoss();
        }

        public boolean hasValidationScore() {
            return validationScore().isPresent();
        }

        public TrainingReportDiagnostics.Summary diagnosticSummary() {
            return report.diagnosticSummary();
        }

        public boolean passesDiagnostics(TrainingReportDiagnostics.Severity maxAllowedSeverity) {
            return report.diagnosticGate(maxAllowedSeverity).passed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("name", name);
            if (source != null) {
                map.put("source", source.toString());
            }
            if (sourceBytes != null) {
                map.put("sourceBytes", sourceBytes);
            }
            if (sourceSha256 != null) {
                map.put("sourceSha256", sourceSha256);
            }
            putOptional(map, "validationScore", validationScore());
            putOptional(map, "bestValidationLoss", report.bestValidationLoss());
            putOptional(map, "latestValidationLoss", report.latestValidationLoss());
            putOptional(map, "latestTrainLoss", report.latestTrainLoss());
            map.put("epochCount", report.epochCount());
            map.put("durationMs", report.durationMs());
            map.put("highestDiagnosticSeverity", report.highestDiagnosticSeverity());
            map.put("dataHealth", TrainingReportDataHealthSummary.from(report.dataHealth()).toMap());
            return Map.copyOf(map);
        }
    }

    public record PromotionDecision(
            Entry baseline,
            Optional<Entry> candidate,
            Optional<TrainingReportComparison> comparison,
            PromotionPolicy policy,
            PromotionStatus status,
            List<String> reasons) {
        public PromotionDecision {
            baseline = Objects.requireNonNull(baseline, "baseline must not be null");
            candidate = candidate == null ? Optional.empty() : candidate;
            comparison = comparison == null ? Optional.empty() : comparison;
            policy = policy == null ? PromotionPolicy.defaultPolicy() : policy;
            status = Objects.requireNonNull(status, "status must not be null");
            reasons = reasons == null ? List.of() : List.copyOf(reasons);
        }

        public TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity() {
            return policy.maxCandidateDiagnosticSeverity();
        }

        public boolean promotable() {
            return status == PromotionStatus.PROMOTE;
        }

        public void requirePromotable() {
            if (!promotable()) {
                throw new IllegalStateException(message());
            }
        }

        public String message() {
            if (promotable()) {
                return "Candidate " + candidate.map(Entry::name).orElse("<none>")
                        + " can replace baseline " + baseline.name() + ".";
            }
            return "Hold baseline " + baseline.name() + ": " + String.join("; ", reasons) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("baseline", baseline.name());
            candidate.ifPresent(entry -> map.put("candidate", entry.name()));
            map.put("baselineReport", baseline.toMap());
            candidate.ifPresent(entry -> map.put("candidateReport", entry.toMap()));
            map.put("maxAllowedDiagnosticSeverity", maxAllowedDiagnosticSeverity().name());
            map.put("policy", policy.toMap());
            map.put("status", status.name());
            map.put("promotable", promotable());
            map.put("reasons", reasons);
            comparison.ifPresent(value -> map.put("comparison", value.toMap()));
            return Map.copyOf(map);
        }
    }

    public record PromotionCandidateReview(
            Entry baseline,
            Entry candidate,
            TrainingReportComparison comparison,
            TrainingReportDiagnostics.GateResult diagnosticGate,
            PromotionPolicy policy,
            PromotionStatus status,
            List<String> reasons) {
        public PromotionCandidateReview {
            baseline = Objects.requireNonNull(baseline, "baseline must not be null");
            candidate = Objects.requireNonNull(candidate, "candidate must not be null");
            comparison = Objects.requireNonNull(comparison, "comparison must not be null");
            diagnosticGate = Objects.requireNonNull(diagnosticGate, "diagnosticGate must not be null");
            policy = policy == null ? PromotionPolicy.defaultPolicy() : policy;
            status = Objects.requireNonNull(status, "status must not be null");
            if (status == PromotionStatus.NO_ELIGIBLE_CANDIDATE) {
                throw new IllegalArgumentException("candidate review cannot use NO_ELIGIBLE_CANDIDATE status");
            }
            reasons = reasons == null ? List.of() : List.copyOf(reasons);
        }

        public boolean diagnosticsPassed() {
            return diagnosticGate.passed();
        }

        public boolean promotable() {
            return status == PromotionStatus.PROMOTE;
        }

        public PromotionDecision toDecision() {
            return new PromotionDecision(
                    baseline,
                    Optional.of(candidate),
                    Optional.of(comparison),
                    policy,
                    status,
                    reasons);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("baseline", baseline.name());
            map.put("candidate", candidate.name());
            map.put("baselineReport", baseline.toMap());
            map.put("candidateReport", candidate.toMap());
            map.put("status", status.name());
            map.put("promotable", promotable());
            map.put("diagnosticsPassed", diagnosticsPassed());
            map.put("diagnosticGate", diagnosticGateToMap(diagnosticGate));
            map.put("policy", policy.toMap());
            map.put("reasons", reasons);
            map.put("comparison", comparison.toMap());
            return Map.copyOf(map);
        }
    }

    public record PromotionReview(
            Entry baseline,
            PromotionPolicy policy,
            List<PromotionCandidateReview> candidates) {
        public PromotionReview {
            baseline = Objects.requireNonNull(baseline, "baseline must not be null");
            policy = policy == null ? PromotionPolicy.defaultPolicy() : policy;
            candidates = candidates == null ? List.of() : List.copyOf(candidates);
        }

        public Optional<PromotionCandidateReview> candidate(String name) {
            if (name == null || name.isBlank()) {
                return Optional.empty();
            }
            return candidates.stream()
                    .filter(candidate -> candidate.candidate().name().equals(name))
                    .findFirst();
        }

        public List<PromotionCandidateReview> eligibleCandidates() {
            return candidates.stream()
                    .filter(PromotionCandidateReview::diagnosticsPassed)
                    .toList();
        }

        public List<PromotionCandidateReview> promotableCandidates() {
            return candidates.stream()
                    .filter(PromotionCandidateReview::promotable)
                    .toList();
        }

        public List<PromotionCandidateReview> heldCandidates() {
            return candidates.stream()
                    .filter(candidate -> !candidate.promotable())
                    .toList();
        }

        public PromotionDecision decision() {
            Optional<PromotionCandidateReview> promotable = promotableCandidates().stream().findFirst();
            if (promotable.isPresent()) {
                return promotable.orElseThrow().toDecision();
            }

            Optional<PromotionCandidateReview> firstEligibleHold = eligibleCandidates().stream().findFirst();
            if (firstEligibleHold.isPresent()) {
                return firstEligibleHold.orElseThrow().toDecision();
            }

            return new PromotionDecision(
                    baseline,
                    Optional.empty(),
                    Optional.empty(),
                    policy,
                    PromotionStatus.NO_ELIGIBLE_CANDIDATE,
                    List.of("No candidate passed diagnostics at or below "
                            + policy.maxCandidateDiagnosticSeverity().name()));
        }

        public Map<String, Object> toMap() {
            PromotionDecision decision = decision();
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("baseline", baseline.name());
            map.put("baselineReport", baseline.toMap());
            map.put("policy", policy.toMap());
            map.put("candidateCount", candidates.size());
            map.put("eligibleCount", eligibleCandidates().size());
            map.put("promotableCount", promotableCandidates().size());
            map.put("heldCount", heldCandidates().size());
            map.put("decision", decision.toMap());
            map.put("candidates", candidates.stream().map(PromotionCandidateReview::toMap).toList());
            return Map.copyOf(map);
        }
    }

    public static TrainingReportPortfolio fromFiles(Map<String, Path> reportFiles) throws IOException {
        Objects.requireNonNull(reportFiles, "reportFiles must not be null");
        List<Entry> entries = new ArrayList<>();
        for (Map.Entry<String, Path> entry : reportFiles.entrySet()) {
            Path reportFile = Objects.requireNonNull(entry.getValue(), "report file path must not be null")
                    .toAbsolutePath()
                    .normalize();
            entries.add(new Entry(
                    entry.getKey(),
                    TrainingReportReader.readReport(reportFile),
                    reportFile,
                    Files.size(reportFile),
                    TrainerCheckpointIO.sha256Hex(reportFile)));
        }
        return new TrainingReportPortfolio(entries);
    }

    public static TrainingReportPortfolio fromReports(Map<String, TrainingReport> reports) {
        Objects.requireNonNull(reports, "reports must not be null");
        List<Entry> entries = new ArrayList<>();
        for (Map.Entry<String, TrainingReport> entry : reports.entrySet()) {
            entries.add(new Entry(entry.getKey(), entry.getValue(), null));
        }
        return new TrainingReportPortfolio(entries);
    }

    public Optional<Entry> entry(String name) {
        if (name == null || name.isBlank()) {
            return Optional.empty();
        }
        return entries.stream()
                .filter(entry -> entry.name().equals(name))
                .findFirst();
    }

    public List<Entry> rankedByValidationScore() {
        return entries.stream()
                .sorted(ENTRY_RANKING)
                .toList();
    }

    public List<Entry> rankedPassingDiagnostics(TrainingReportDiagnostics.Severity maxAllowedSeverity) {
        Objects.requireNonNull(maxAllowedSeverity, "maxAllowedSeverity must not be null");
        return entries.stream()
                .filter(entry -> entry.passesDiagnostics(maxAllowedSeverity))
                .sorted(ENTRY_RANKING)
                .toList();
    }

    public Optional<Entry> bestByValidationScore() {
        return rankedByValidationScore().stream().findFirst();
    }

    public Optional<Entry> bestPassingDiagnostics(TrainingReportDiagnostics.Severity maxAllowedSeverity) {
        return rankedPassingDiagnostics(maxAllowedSeverity).stream().findFirst();
    }

    public TrainingReportComparison compare(String baselineName, String candidateName) {
        Entry baseline = entry(baselineName)
                .orElseThrow(() -> new IllegalArgumentException("Unknown baseline report: " + baselineName));
        Entry candidate = entry(candidateName)
                .orElseThrow(() -> new IllegalArgumentException("Unknown candidate report: " + candidateName));
        return TrainingReportComparison.compare(baseline.report(), candidate.report());
    }

    public PromotionDecision promotionDecision(
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity) {
        Objects.requireNonNull(maxAllowedDiagnosticSeverity, "maxAllowedDiagnosticSeverity must not be null");
        return promotionDecision(
                baselineName,
                PromotionPolicy.defaultPolicy()
                        .withMaxCandidateDiagnosticSeverity(maxAllowedDiagnosticSeverity));
    }

    public PromotionDecision promotionDecision(String baselineName, PromotionPolicy policy) {
        return promotionReview(baselineName, policy).decision();
    }

    public PromotionReview promotionReview(
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity) {
        Objects.requireNonNull(maxAllowedDiagnosticSeverity, "maxAllowedDiagnosticSeverity must not be null");
        return promotionReview(
                baselineName,
                PromotionPolicy.defaultPolicy()
                        .withMaxCandidateDiagnosticSeverity(maxAllowedDiagnosticSeverity));
    }

    public PromotionReview promotionReview(String baselineName, PromotionPolicy policy) {
        PromotionPolicy resolvedPolicy = policy == null ? PromotionPolicy.defaultPolicy() : policy;
        Entry baseline = entry(baselineName)
                .orElseThrow(() -> new IllegalArgumentException("Unknown baseline report: " + baselineName));
        List<PromotionCandidateReview> reviews = rankedByValidationScore().stream()
                .filter(entry -> !entry.name().equals(baseline.name()))
                .map(candidate -> promotionCandidateReview(baseline, candidate, resolvedPolicy))
                .toList();
        return new PromotionReview(baseline, resolvedPolicy, reviews);
    }

    public Map<String, TrainingReportComparison> compareAllAgainst(String baselineName) {
        Entry baseline = entry(baselineName)
                .orElseThrow(() -> new IllegalArgumentException("Unknown baseline report: " + baselineName));
        Map<String, TrainingReportComparison> comparisons = new LinkedHashMap<>();
        for (Entry candidate : entries) {
            if (!candidate.name().equals(baseline.name())) {
                comparisons.put(candidate.name(), TrainingReportComparison.compare(baseline.report(), candidate.report()));
            }
        }
        return Map.copyOf(comparisons);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("total", entries.size());
        map.put("entries", entries.stream().map(Entry::toMap).toList());
        bestByValidationScore().ifPresent(best -> map.put("bestByValidationScore", best.name()));
        return Map.copyOf(map);
    }

    public TrainingReportPortfolioExport export() {
        return TrainingReportPortfolioExport.fromPortfolio(this);
    }

    public TrainingReportPortfolioExport exportAgainst(String baselineName) {
        return TrainingReportPortfolioExport.fromPortfolio(this, baselineName);
    }

    private static final Comparator<Entry> ENTRY_RANKING = Comparator
            .comparing((Entry entry) -> entry.hasValidationScore() ? 0 : 1)
            .thenComparingDouble(entry -> entry.validationScore().orElse(Double.POSITIVE_INFINITY))
            .thenComparingInt(entry -> severityRank(entry.report().highestDiagnosticSeverity()))
            .thenComparingLong(entry -> entry.report().durationMs() > 0L ? entry.report().durationMs() : Long.MAX_VALUE)
            .thenComparing(Entry::name);

    private static int severityRank(String severity) {
        if (severity == null || severity.isBlank() || "NONE".equalsIgnoreCase(severity)) {
            return -1;
        }
        for (TrainingReportDiagnostics.Severity value : TrainingReportDiagnostics.Severity.values()) {
            if (value.name().equalsIgnoreCase(severity)) {
                return value.ordinal();
            }
        }
        return TrainingReportDiagnostics.Severity.CRITICAL.ordinal() + 1;
    }

    private static List<String> promotionHoldReasons(
            Entry baseline,
            Entry candidate,
            TrainingReportComparison comparison,
            PromotionPolicy policy) {
        List<String> reasons = new ArrayList<>();
        if (!beatsValidationScore(baseline, candidate, policy.minimumValidationImprovement())) {
            reasons.add("Candidate validation score does not beat baseline validation score by at least "
                    + policy.minimumValidationImprovement());
        }
        TrainingReportDiagnostics.GateResult comparisonGate = comparison.gate(policy.maxComparisonFindingSeverity());
        if (!comparisonGate.passed()) {
            reasons.add("Candidate comparison has findings: " + comparisonGate.failingCodes());
        }
        if (policy.requireTrackedMetricImprovement() && comparison.improvedMetrics().isEmpty()) {
            reasons.add("Candidate does not improve any tracked metric");
        }
        return List.copyOf(reasons);
    }

    private static PromotionCandidateReview promotionCandidateReview(
            Entry baseline,
            Entry candidate,
            PromotionPolicy policy) {
        TrainingReportComparison comparison = TrainingReportComparison.compare(baseline.report(), candidate.report());
        TrainingReportDiagnostics.GateResult diagnosticGate =
                candidate.report().diagnosticGate(policy.maxCandidateDiagnosticSeverity());
        List<String> reasons = new ArrayList<>();
        if (!diagnosticGate.passed()) {
            reasons.add("Candidate diagnostics exceed " + policy.maxCandidateDiagnosticSeverity().name()
                    + ": " + diagnosticGate.failingCodes());
        }
        reasons.addAll(promotionHoldReasons(baseline, candidate, comparison, policy));
        return new PromotionCandidateReview(
                baseline,
                candidate,
                comparison,
                diagnosticGate,
                policy,
                reasons.isEmpty() ? PromotionStatus.PROMOTE : PromotionStatus.HOLD,
                reasons);
    }

    private static Map<String, Object> diagnosticGateToMap(TrainingReportDiagnostics.GateResult gate) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("summary", gate.summary().toMap());
        map.put("maxAllowedSeverity", gate.maxAllowedSeverity().name());
        map.put("passed", gate.passed());
        map.put("failingCodes", gate.failingCodes());
        map.put("failingFindings", TrainingReportDiagnostics.toMaps(gate.failingFindings()));
        return Map.copyOf(map);
    }

    private static boolean beatsValidationScore(Entry baseline, Entry candidate, double tolerance) {
        OptionalDouble candidateScore = candidate.validationScore();
        if (candidateScore.isEmpty()) {
            return false;
        }
        OptionalDouble baselineScore = baseline.validationScore();
        if (baselineScore.isEmpty()) {
            return true;
        }
        return candidateScore.getAsDouble() < baselineScore.getAsDouble() - Math.max(0.0, tolerance);
    }

    private static void putOptional(Map<String, Object> target, String key, OptionalDouble value) {
        if (value != null && value.isPresent()) {
            target.put(key, value.getAsDouble());
        }
    }

    private static String normalizeSha256(String value, String fieldName) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String normalized = value.trim().toLowerCase(java.util.Locale.ROOT);
        if (normalized.length() != 64) {
            throw new IllegalArgumentException(fieldName + " must be a 64-character SHA-256 hex digest");
        }
        for (int index = 0; index < normalized.length(); index++) {
            char character = normalized.charAt(index);
            if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) {
                throw new IllegalArgumentException(fieldName + " must be a SHA-256 hex digest");
            }
        }
        return normalized;
    }
}

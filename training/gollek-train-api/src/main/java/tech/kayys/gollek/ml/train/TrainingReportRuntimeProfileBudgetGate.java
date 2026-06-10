package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * CI-friendly gate for trainer runtime-profile performance budgets.
 */
public final class TrainingReportRuntimeProfileBudgetGate {
    private TrainingReportRuntimeProfileBudgetGate() {
    }

    public record Policy(
            double maxPrimaryGroupPercent,
            double maxPrimaryHotspotPercent,
            double maxPrimaryHotspotTotalMillis,
            double maxInputBalancePercent,
            double maxOptimizerBalancePercent,
            double maxValidationBalancePercent,
            double maxWallClockOverheadPercent,
            double maxWallClockOverheadMillis) {
        public Policy {
            maxPrimaryGroupPercent = positiveOrDefault(maxPrimaryGroupPercent, 80.0);
            maxPrimaryHotspotPercent = positiveOrDefault(maxPrimaryHotspotPercent, 60.0);
            maxPrimaryHotspotTotalMillis = positiveOrDefault(maxPrimaryHotspotTotalMillis, Double.POSITIVE_INFINITY);
            maxInputBalancePercent = positiveOrDefault(maxInputBalancePercent, 60.0);
            maxOptimizerBalancePercent = positiveOrDefault(maxOptimizerBalancePercent, 50.0);
            maxValidationBalancePercent = positiveOrDefault(maxValidationBalancePercent, 60.0);
            maxWallClockOverheadPercent = positiveOrDefault(maxWallClockOverheadPercent, 35.0);
            maxWallClockOverheadMillis = positiveOrDefault(maxWallClockOverheadMillis, Double.POSITIVE_INFINITY);
        }

        public Policy(
                double maxPrimaryGroupPercent,
                double maxPrimaryHotspotPercent,
                double maxPrimaryHotspotTotalMillis) {
            this(
                    maxPrimaryGroupPercent,
                    maxPrimaryHotspotPercent,
                    maxPrimaryHotspotTotalMillis,
                    60.0,
                    50.0,
                    60.0,
                    35.0,
                    Double.POSITIVE_INFINITY);
        }

        public Policy(
                double maxPrimaryGroupPercent,
                double maxPrimaryHotspotPercent,
                double maxPrimaryHotspotTotalMillis,
                double maxInputBalancePercent,
                double maxOptimizerBalancePercent,
                double maxValidationBalancePercent) {
            this(
                    maxPrimaryGroupPercent,
                    maxPrimaryHotspotPercent,
                    maxPrimaryHotspotTotalMillis,
                    maxInputBalancePercent,
                    maxOptimizerBalancePercent,
                    maxValidationBalancePercent,
                    35.0,
                    Double.POSITIVE_INFINITY);
        }

        public static Policy defaults() {
            return new Policy(80.0, 60.0, Double.POSITIVE_INFINITY, 60.0, 50.0, 60.0, 35.0,
                    Double.POSITIVE_INFINITY);
        }

        public static Policy strict() {
            return new Policy(70.0, 45.0, 250.0, 45.0, 35.0, 45.0, 25.0, 100.0);
        }

        public static Policy permissive() {
            return new Policy(95.0, 90.0, Double.POSITIVE_INFINITY, 85.0, 80.0, 85.0, 60.0,
                    Double.POSITIVE_INFINITY);
        }

        public static Policy fromMap(Map<String, ?> policy) {
            if (policy == null || policy.isEmpty()) {
                return defaults();
            }
            return new Policy(
                    doubleEntry(policy, "maxPrimaryGroupPercent", defaults().maxPrimaryGroupPercent()),
                    doubleEntry(policy, "maxPrimaryHotspotPercent", defaults().maxPrimaryHotspotPercent()),
                    doubleEntry(policy, "maxPrimaryHotspotTotalMillis", defaults().maxPrimaryHotspotTotalMillis()),
                    doubleEntry(policy, "maxInputBalancePercent", defaults().maxInputBalancePercent()),
                    doubleEntry(policy, "maxOptimizerBalancePercent", defaults().maxOptimizerBalancePercent()),
                    doubleEntry(policy, "maxValidationBalancePercent", defaults().maxValidationBalancePercent()),
                    doubleEntry(policy, "maxWallClockOverheadPercent", defaults().maxWallClockOverheadPercent()),
                    doubleEntry(policy, "maxWallClockOverheadMillis", defaults().maxWallClockOverheadMillis()));
        }

        public Policy withMaxPrimaryGroupPercent(double threshold) {
            return new Policy(
                    threshold,
                    maxPrimaryHotspotPercent,
                    maxPrimaryHotspotTotalMillis,
                    maxInputBalancePercent,
                    maxOptimizerBalancePercent,
                    maxValidationBalancePercent,
                    maxWallClockOverheadPercent,
                    maxWallClockOverheadMillis);
        }

        public Policy withMaxPrimaryHotspotPercent(double threshold) {
            return new Policy(
                    maxPrimaryGroupPercent,
                    threshold,
                    maxPrimaryHotspotTotalMillis,
                    maxInputBalancePercent,
                    maxOptimizerBalancePercent,
                    maxValidationBalancePercent,
                    maxWallClockOverheadPercent,
                    maxWallClockOverheadMillis);
        }

        public Policy withMaxPrimaryHotspotTotalMillis(double threshold) {
            return new Policy(
                    maxPrimaryGroupPercent,
                    maxPrimaryHotspotPercent,
                    threshold,
                    maxInputBalancePercent,
                    maxOptimizerBalancePercent,
                    maxValidationBalancePercent,
                    maxWallClockOverheadPercent,
                    maxWallClockOverheadMillis);
        }

        public Policy withMaxInputBalancePercent(double threshold) {
            return new Policy(
                    maxPrimaryGroupPercent,
                    maxPrimaryHotspotPercent,
                    maxPrimaryHotspotTotalMillis,
                    threshold,
                    maxOptimizerBalancePercent,
                    maxValidationBalancePercent,
                    maxWallClockOverheadPercent,
                    maxWallClockOverheadMillis);
        }

        public Policy withMaxOptimizerBalancePercent(double threshold) {
            return new Policy(
                    maxPrimaryGroupPercent,
                    maxPrimaryHotspotPercent,
                    maxPrimaryHotspotTotalMillis,
                    maxInputBalancePercent,
                    threshold,
                    maxValidationBalancePercent,
                    maxWallClockOverheadPercent,
                    maxWallClockOverheadMillis);
        }

        public Policy withMaxValidationBalancePercent(double threshold) {
            return new Policy(
                    maxPrimaryGroupPercent,
                    maxPrimaryHotspotPercent,
                    maxPrimaryHotspotTotalMillis,
                    maxInputBalancePercent,
                    maxOptimizerBalancePercent,
                    threshold,
                    maxWallClockOverheadPercent,
                    maxWallClockOverheadMillis);
        }

        public Policy withMaxWallClockOverheadPercent(double threshold) {
            return new Policy(
                    maxPrimaryGroupPercent,
                    maxPrimaryHotspotPercent,
                    maxPrimaryHotspotTotalMillis,
                    maxInputBalancePercent,
                    maxOptimizerBalancePercent,
                    maxValidationBalancePercent,
                    threshold,
                    maxWallClockOverheadMillis);
        }

        public Policy withMaxWallClockOverheadMillis(double threshold) {
            return new Policy(
                    maxPrimaryGroupPercent,
                    maxPrimaryHotspotPercent,
                    maxPrimaryHotspotTotalMillis,
                    maxInputBalancePercent,
                    maxOptimizerBalancePercent,
                    maxValidationBalancePercent,
                    maxWallClockOverheadPercent,
                    threshold);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("maxPrimaryGroupPercent", maxPrimaryGroupPercent);
            map.put("maxPrimaryHotspotPercent", maxPrimaryHotspotPercent);
            map.put("maxPrimaryHotspotTotalMillis", maxPrimaryHotspotTotalMillis);
            map.put("maxInputBalancePercent", maxInputBalancePercent);
            map.put("maxOptimizerBalancePercent", maxOptimizerBalancePercent);
            map.put("maxValidationBalancePercent", maxValidationBalancePercent);
            map.put("maxWallClockOverheadPercent", maxWallClockOverheadPercent);
            map.put("maxWallClockOverheadMillis", maxWallClockOverheadMillis);
            return Map.copyOf(map);
        }
    }

    public record Finding(
            String code,
            String severity,
            String message,
            String action,
            Map<String, Object> evidence) {
        public Finding {
            code = code == null || code.isBlank() ? "runtime-profile-budget" : code.trim();
            severity = severity == null || severity.isBlank() ? "warning" : severity.trim();
            message = message == null ? "" : message.trim();
            action = action == null ? "" : action.trim();
            evidence = evidence == null ? Map.of() : Map.copyOf(evidence);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("code", code);
            map.put("severity", severity);
            map.put("message", message);
            map.put("action", action);
            map.put("evidence", evidence);
            return Map.copyOf(map);
        }
    }

    public record Result(
            Policy policy,
            boolean available,
            List<Finding> findings,
            Map<String, Object> runtimeProfile,
            Map<String, Object> actionPlan) {
        public Result {
            policy = policy == null ? Policy.defaults() : policy;
            findings = findings == null ? List.of() : List.copyOf(findings);
            runtimeProfile = runtimeProfile == null ? Map.of() : Map.copyOf(runtimeProfile);
            actionPlan = actionPlan == null ? Map.of() : Map.copyOf(actionPlan);
        }

        public boolean passed() {
            return findings.isEmpty();
        }

        public String message() {
            if (!available) {
                return "Runtime profile is not available.";
            }
            if (passed()) {
                return "Runtime profile budget gate passed.";
            }
            return "Runtime profile budget gate found " + findings.size() + " budget warning(s).";
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("available", available);
            map.put("passed", passed());
            map.put("message", message());
            map.put("policy", policy.toMap());
            map.put("findingCount", findings.size());
            map.put("findings", findings.stream().map(Finding::toMap).toList());
            map.put("runtimeProfile", runtimeProfile);
            map.put("actionPlan", actionPlan);
            return Map.copyOf(map);
        }

        public String markdown() {
            return TrainingReportRuntimeProfileBudgetGateMarkdown.render(this);
        }

        public String junitXml() {
            return TrainingReportRuntimeProfileBudgetGateJUnitXml.render(this);
        }
    }

    public static Result evaluate(TrainingReport report) {
        return evaluate(report, Policy.defaults());
    }

    public static Result evaluate(TrainingReport report, Policy policy) {
        Objects.requireNonNull(report, "report must not be null");
        Policy resolvedPolicy = policy == null ? Policy.defaults() : policy;
        TrainingReportRuntimeProfile profile = report.runtimeProfile();
        TrainingReportRuntimeProfileActionPlan plan = report.runtimeProfileActionPlan();
        if (!profile.available()) {
            return new Result(resolvedPolicy, false, List.of(), profile.toMap(), plan.toMap());
        }
        List<Finding> findings = new ArrayList<>();
        addPrimaryGroupFinding(findings, profile, plan, resolvedPolicy);
        addPrimaryHotspotPercentFinding(findings, profile, plan, resolvedPolicy);
        addPrimaryHotspotMillisFinding(findings, profile, plan, resolvedPolicy);
        addBalanceFindings(findings, profile, plan, resolvedPolicy);
        addWallClockOverheadFinding(findings, profile, plan, resolvedPolicy);
        return new Result(resolvedPolicy, true, findings, profile.toMap(), plan.toMap());
    }

    public static Result evaluate(Map<String, ?> report) {
        return evaluate(TrainingReport.of(report), Policy.defaults());
    }

    public static Result evaluate(Map<String, ?> report, Policy policy) {
        return evaluate(TrainingReport.of(report), policy);
    }

    private static void addPrimaryGroupFinding(
            List<Finding> findings,
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfileActionPlan plan,
            Policy policy) {
        profile.primaryGroup().ifPresent(group -> group.percentTotal().ifPresent(percent -> {
            if (percent <= policy.maxPrimaryGroupPercent()) {
                return;
            }
            findings.add(new Finding(
                    "runtime-profile-primary-group-budget",
                    "warning",
                    "Primary runtime group `" + group.name() + "` uses " + format(percent)
                            + "% of measured trainer time.",
                    firstAction(plan, group.name(), "Reduce work in the `" + group.name()
                            + "` runtime group before tuning unrelated phases."),
                    evidence("group", group.name(), percent, group.totalMillis().orElse(Double.NaN))));
        }));
    }

    private static void addPrimaryHotspotPercentFinding(
            List<Finding> findings,
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfileActionPlan plan,
            Policy policy) {
        profile.primaryHotspot().ifPresent(hotspot -> hotspot.percentTotal().ifPresent(percent -> {
            if (percent <= policy.maxPrimaryHotspotPercent()) {
                return;
            }
            findings.add(new Finding(
                    "runtime-profile-primary-hotspot-percent-budget",
                    "warning",
                    "Primary runtime hotspot `" + hotspot.phase() + "` uses " + format(percent)
                            + "% of measured trainer time.",
                    firstAction(plan, hotspot.phase(), "Optimize `" + hotspot.phase()
                            + "` before broad trainer rewrites."),
                    evidence("phase", hotspot.phase(), percent, hotspot.totalMillis().orElse(Double.NaN))));
        }));
    }

    private static void addPrimaryHotspotMillisFinding(
            List<Finding> findings,
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfileActionPlan plan,
            Policy policy) {
        if (!Double.isFinite(policy.maxPrimaryHotspotTotalMillis())) {
            return;
        }
        profile.primaryHotspot().ifPresent(hotspot -> hotspot.totalMillis().ifPresent(totalMillis -> {
            if (totalMillis <= policy.maxPrimaryHotspotTotalMillis()) {
                return;
            }
            findings.add(new Finding(
                    "runtime-profile-primary-hotspot-millis-budget",
                    "warning",
                    "Primary runtime hotspot `" + hotspot.phase() + "` took " + format(totalMillis)
                            + " ms, above the configured budget.",
                    firstAction(plan, hotspot.phase(), "Benchmark and optimize `" + hotspot.phase()
                            + "` until it fits the runtime budget."),
                    evidence("phase", hotspot.phase(), hotspot.percentTotal().orElse(Double.NaN), totalMillis)));
        }));
    }

    private static void addBalanceFindings(
            List<Finding> findings,
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfileActionPlan plan,
            Policy policy) {
        TrainingReportRuntimeProfileBalanceAssessment assessment =
                TrainingReportRuntimeProfileBalanceAssessment.assess(
                        profile.balance(),
                        new TrainingReportRuntimeProfileBalanceAssessment.Thresholds(
                                policy.maxInputBalancePercent(),
                                policy.maxOptimizerBalancePercent(),
                                policy.maxValidationBalancePercent(),
                                Double.POSITIVE_INFINITY));
        if (!assessment.available()) {
            return;
        }
        addBalanceFinding(
                findings,
                plan,
                assessment,
                TrainingReportRuntimeProfile.BalanceBucket.INPUT,
                "runtime-profile-input-balance-budget",
                "Input pipeline",
                "Enable `DataLoader.prefetch(...)` or reduce synchronous decoding, augmentation, and collation work.");
        addBalanceFinding(
                findings,
                plan,
                assessment,
                TrainingReportRuntimeProfile.BalanceBucket.OPTIMIZER,
                "runtime-profile-optimizer-balance-budget",
                "Optimizer",
                "Reduce full-tensor diagnostics, clipping passes, or optimizer state overhead before changing model code.");
        addBalanceFinding(
                findings,
                plan,
                assessment,
                TrainingReportRuntimeProfile.BalanceBucket.VALIDATION,
                "runtime-profile-validation-balance-budget",
                "Validation",
                "Reduce validation frequency, cache validation preprocessing, or inspect validation-only metrics.");
    }

    private static void addWallClockOverheadFinding(
            List<Finding> findings,
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfileActionPlan plan,
            Policy policy) {
        TrainingReportRuntimeProfile.WallClock wallClock = profile.wallClock();
        TrainingReportRuntimeWallClockAssessment assessment =
                TrainingReportRuntimeWallClockAssessment.assess(wallClock);
        if (!assessment.available()
                || !assessment.exceedsBudget(
                        policy.maxWallClockOverheadPercent(),
                        policy.maxWallClockOverheadMillis())) {
            return;
        }
        findings.add(new Finding(
                "runtime-profile-wall-clock-overhead-budget",
                assessment.severity().name().toLowerCase(java.util.Locale.ROOT),
                "Wall-clock overhead in `" + assessment.scopeKey() + "` is "
                        + format(assessment.overhead().overheadPercent().orElse(Double.NaN)) + "% / "
                        + format(assessment.overhead().overheadMillis().orElse(Double.NaN))
                        + " ms above the configured budget.",
                firstAction(plan, assessment.scopeKey(),
                        "Inspect trainer orchestration around `" + assessment.scopeKey()
                                + "` before tuning tensor kernels."),
                assessment.budgetEvidence(
                        wallClock.totalMillis(),
                        policy.maxWallClockOverheadPercent(),
                        policy.maxWallClockOverheadMillis())));
    }

    private static void addBalanceFinding(
            List<Finding> findings,
            TrainingReportRuntimeProfileActionPlan plan,
            TrainingReportRuntimeProfileBalanceAssessment assessment,
            TrainingReportRuntimeProfile.BalanceBucket bucket,
            String code,
            String label,
            String fallbackAction) {
        assessment.percentTotal(bucket).ifPresent(percent -> {
            double threshold = thresholdFor(assessment.thresholds(), bucket);
            if (!assessment.exceeds(bucket, threshold)) {
                return;
            }
            String bucketName = bucket.name().toLowerCase(java.util.Locale.ROOT);
            findings.add(new Finding(
                    code,
                    "warning",
                    label + " balance uses " + format(percent) + "% of measured trainer time.",
                    firstAction(plan, bucketName, fallbackAction),
                    balanceEvidence(bucketName, percent, assessment.totalMillis(bucket).orElse(Double.NaN), threshold)));
        });
    }

    private static double thresholdFor(
            TrainingReportRuntimeProfileBalanceAssessment.Thresholds thresholds,
            TrainingReportRuntimeProfile.BalanceBucket bucket) {
        return switch (bucket) {
            case INPUT -> thresholds.inputBoundPercent();
            case OPTIMIZER -> thresholds.optimizerBoundPercent();
            case VALIDATION -> thresholds.validationBoundPercent();
            default -> Double.POSITIVE_INFINITY;
        };
    }

    private static String firstAction(
            TrainingReportRuntimeProfileActionPlan plan,
            String targetName,
            String fallback) {
        return plan.targets().stream()
                .filter(target -> target.name().equals(targetName))
                .flatMap(target -> target.actions().stream())
                .findFirst()
                .orElse(fallback);
    }

    private static Map<String, Object> balanceEvidence(
            String bucket,
            double percentTotal,
            double totalMillis,
            double threshold) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("bucket", bucket);
        evidence.put("percentTotal", percentTotal);
        evidence.put("thresholdPercent", threshold);
        if (Double.isFinite(totalMillis)) {
            evidence.put("totalMillis", totalMillis);
        }
        return Map.copyOf(evidence);
    }

    private static Map<String, Object> evidence(
            String targetType,
            String targetName,
            double percentTotal,
            double totalMillis) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put(targetType, targetName);
        if (Double.isFinite(percentTotal)) {
            evidence.put("percentTotal", percentTotal);
        }
        if (Double.isFinite(totalMillis)) {
            evidence.put("totalMillis", totalMillis);
        }
        return Map.copyOf(evidence);
    }

    private static double doubleEntry(Map<String, ?> source, String key, double fallback) {
        var parsed = optionalDouble(source.get(key));
        return parsed.isPresent() ? parsed.getAsDouble() : fallback;
    }

    private static double positiveOrDefault(double value, double fallback) {
        return (Double.isFinite(value) && value > 0.0) || value == Double.POSITIVE_INFINITY
                ? value
                : fallback;
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(java.util.Locale.ROOT, "%.3f", value) : "n/a";
    }
}

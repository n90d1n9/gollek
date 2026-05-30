package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class TrainerDataDistributionHealthRules {
    private static final String KIND = "data-distribution";
    private static final double CLASS_IMBALANCE_WARNING_RATIO = 10.0;
    private static final double MULTI_LABEL_IMBALANCE_WARNING_RATIO = 10.0;
    private static final double CLASSIFICATION_DRIFT_TVD_WARNING = 0.20;
    private static final double CLASSIFICATION_DRIFT_MAX_DELTA_WARNING = 0.25;
    private static final double MULTI_LABEL_DRIFT_MEAN_DELTA_WARNING = 0.10;
    private static final double MULTI_LABEL_DRIFT_MAX_DELTA_WARNING = 0.25;

    private TrainerDataDistributionHealthRules() {
    }

    static List<TrainerHealthIssue> evaluate(Map<String, Object> metadata) {
        List<TrainerHealthIssue> issues = new ArrayList<>();
        addDistributionIssues(metadata, issues, "trainDataDistribution", "train", true);
        addDistributionIssues(metadata, issues, "validationDataDistribution", "validation", false);
        addDriftIssues(metadata, issues);
        return List.copyOf(issues);
    }

    private static void addDistributionIssues(
            Map<String, Object> metadata,
            List<TrainerHealthIssue> issues,
            String prefix,
            String phase,
            boolean required) {
        if (!booleanValue(metadata.get(prefix + ".available"))) {
            if (required) {
                issues.add(warning(
                        code(phase, "distribution-unavailable"),
                        phase,
                        phase + " data distribution diagnostics are unavailable: "
                                + stringValue(metadata.get(prefix + ".skipReason"), "unavailable"),
                        "enable diagnostics on a tensor-backed training loader before promotion checks",
                        evidence(prefix, metadata)));
            }
            return;
        }

        int sampleCount = intValue(metadata.get(prefix + ".sampleCount"), -1);
        if (sampleCount <= 0) {
            issues.add(warning(
                    code(phase, "empty-distribution"),
                    phase,
                    phase + " distribution diagnostics found no samples",
                    "provide non-empty " + phase + " data before relying on distribution diagnostics",
                    evidence(prefix, metadata)));
            return;
        }

        String kind = stringValue(metadata.get(prefix + ".kind"), "unknown");
        if ("multi-label".equals(kind)) {
            addMultiLabelIssues(metadata, issues, prefix, phase);
        } else if ("classification".equals(kind)) {
            addClassificationIssues(metadata, issues, prefix, phase);
        }
    }

    private static void addClassificationIssues(
            Map<String, Object> metadata,
            List<TrainerHealthIssue> issues,
            String prefix,
            String phase) {
        int missingClassCount = intValue(metadata.get(prefix + ".missingClassCount"), 0);
        if (missingClassCount > 0) {
            issues.add(warning(
                    code(phase, "missing-classes"),
                    phase,
                    phase + " distribution is missing " + missingClassCount + " class(es)",
                    "rebalance the split, add stratified sampling, or confirm the missing classes are intentional",
                    evidence(prefix, metadata)));
        }

        double imbalanceRatio = optionalDouble(metadata.get(prefix + ".imbalanceRatio")).orElse(1.0);
        if (imbalanceRatio >= CLASS_IMBALANCE_WARNING_RATIO) {
            issues.add(warning(
                    code(phase, "class-imbalance"),
                    phase,
                    phase + " distribution has class imbalance ratio " + compact(imbalanceRatio),
                    "use balanced class weights, class-balanced sampling, or collect more minority examples",
                    evidence(prefix, metadata)));
        }
    }

    private static void addMultiLabelIssues(
            Map<String, Object> metadata,
            List<TrainerHealthIssue> issues,
            String prefix,
            String phase) {
        int zeroPositiveLabels = intValue(metadata.get(prefix + ".zeroPositiveLabelCount"), 0);
        if (zeroPositiveLabels > 0) {
            issues.add(warning(
                    code(phase, "zero-positive-labels"),
                    phase,
                    phase + " multi-label distribution has " + zeroPositiveLabels + " label(s) with no positives",
                    "rebalance labels or remove labels that cannot be learned from this split",
                    evidence(prefix, metadata)));
        }

        int allPositiveLabels = intValue(metadata.get(prefix + ".allPositiveLabelCount"), 0);
        if (allPositiveLabels > 0) {
            issues.add(warning(
                    code(phase, "all-positive-labels"),
                    phase,
                    phase + " multi-label distribution has " + allPositiveLabels + " label(s) always positive",
                    "verify label extraction or split data so each label has both positive and negative examples",
                    evidence(prefix, metadata)));
        }

        double imbalanceRatio = optionalDouble(metadata.get(prefix + ".positiveImbalanceRatio")).orElse(1.0);
        if (imbalanceRatio >= MULTI_LABEL_IMBALANCE_WARNING_RATIO) {
            issues.add(warning(
                    code(phase, "positive-label-imbalance"),
                    phase,
                    phase + " multi-label positive imbalance ratio is " + compact(imbalanceRatio),
                    "use positive class weights or a label-balanced sampler for sparse labels",
                    evidence(prefix, metadata)));
        }
    }

    private static void addDriftIssues(Map<String, Object> metadata, List<TrainerHealthIssue> issues) {
        if (!booleanValue(metadata.get("dataDistributionDrift.available"))) {
            if (booleanValue(metadata.get("dataDistributionDrift.skipped"))) {
                return;
            }
            issues.add(warning(
                    "data-distribution-drift-unavailable",
                    "drift",
                    "data distribution drift diagnostics failed",
                    "inspect dataDistributionDrift.error and rerun diagnostics with compatible loaders",
                    evidence("dataDistributionDrift", metadata)));
            return;
        }

        String kind = stringValue(metadata.get("dataDistributionDrift.kind"), "unknown");
        if ("multi-label".equals(kind)) {
            addMultiLabelDriftIssues(metadata, issues);
        } else if ("classification".equals(kind)) {
            addClassificationDriftIssues(metadata, issues);
        }
    }

    private static void addClassificationDriftIssues(
            Map<String, Object> metadata,
            List<TrainerHealthIssue> issues) {
        double tvd = optionalDouble(metadata.get("dataDistributionDrift.totalVariationDistance")).orElse(0.0);
        double maxDelta = optionalDouble(metadata.get("dataDistributionDrift.maxAbsoluteFractionDelta")).orElse(0.0);
        if (tvd >= CLASSIFICATION_DRIFT_TVD_WARNING || maxDelta >= CLASSIFICATION_DRIFT_MAX_DELTA_WARNING) {
            issues.add(warning(
                    "data-distribution-classification-drift",
                    "drift",
                    "train/validation class distribution drift is high"
                            + " (TVD=" + compact(tvd) + ", max delta=" + compact(maxDelta) + ")",
                    "use stratified splitting or rebalance validation data before comparing generalization",
                    evidence("dataDistributionDrift", metadata)));
        }

        if (!listValue(metadata.get("dataDistributionDrift.candidateMissingClasses")).isEmpty()) {
            issues.add(warning(
                    "data-distribution-validation-missing-classes",
                    "validation",
                    "validation distribution is missing classes present in training",
                    "use stratified validation data so metrics cover the same label space as training",
                    evidence("dataDistributionDrift", metadata)));
        }
    }

    private static void addMultiLabelDriftIssues(
            Map<String, Object> metadata,
            List<TrainerHealthIssue> issues) {
        double meanDelta = optionalDouble(
                metadata.get("dataDistributionDrift.meanAbsolutePositiveFractionDelta")).orElse(0.0);
        double maxDelta = optionalDouble(
                metadata.get("dataDistributionDrift.maxAbsolutePositiveFractionDelta")).orElse(0.0);
        if (meanDelta >= MULTI_LABEL_DRIFT_MEAN_DELTA_WARNING || maxDelta >= MULTI_LABEL_DRIFT_MAX_DELTA_WARNING) {
            issues.add(warning(
                    "data-distribution-multilabel-drift",
                    "drift",
                    "train/validation positive-label drift is high"
                            + " (mean delta=" + compact(meanDelta) + ", max delta=" + compact(maxDelta) + ")",
                    "rebalance validation labels or report metrics per label before promotion",
                    evidence("dataDistributionDrift", metadata)));
        }

        if (!listValue(metadata.get("dataDistributionDrift.candidateZeroPositiveLabels")).isEmpty()) {
            issues.add(warning(
                    "data-distribution-validation-zero-positive-labels",
                    "validation",
                    "validation distribution has labels with no positive examples",
                    "add validation examples for every trained label or exclude unsupported labels from metrics",
                    evidence("dataDistributionDrift", metadata)));
        }
    }

    private static TrainerHealthIssue warning(
            String code,
            String artifact,
            String message,
            String action,
            Map<String, Object> evidence) {
        return TrainerHealthIssue.warning(KIND, code, artifact, message, action, evidence);
    }

    private static Map<String, Object> evidence(String prefix, Map<String, Object> metadata) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        copy(metadata, evidence, prefix + ".available");
        copy(metadata, evidence, prefix + ".skipReason");
        copy(metadata, evidence, prefix + ".errorType");
        copy(metadata, evidence, prefix + ".error");
        copy(metadata, evidence, prefix + ".kind");
        copy(metadata, evidence, prefix + ".sampleCount");
        copy(metadata, evidence, prefix + ".batchCount");
        copy(metadata, evidence, prefix + ".numClasses");
        copy(metadata, evidence, prefix + ".labelCount");
        copy(metadata, evidence, prefix + ".missingClassCount");
        copy(metadata, evidence, prefix + ".imbalanceRatio");
        copy(metadata, evidence, prefix + ".normalizedEntropy");
        copy(metadata, evidence, prefix + ".zeroPositiveLabelCount");
        copy(metadata, evidence, prefix + ".allPositiveLabelCount");
        copy(metadata, evidence, prefix + ".positiveImbalanceRatio");
        copy(metadata, evidence, prefix + ".totalVariationDistance");
        copy(metadata, evidence, prefix + ".maxAbsoluteFractionDelta");
        copy(metadata, evidence, prefix + ".meanAbsolutePositiveFractionDelta");
        copy(metadata, evidence, prefix + ".candidateMissingClasses");
        copy(metadata, evidence, prefix + ".candidateZeroPositiveLabels");
        return Map.copyOf(evidence);
    }

    private static void copy(Map<String, Object> source, Map<String, Object> target, String key) {
        if (source.containsKey(key)) {
            target.put(key, source.get(key));
        }
    }

    private static List<?> listValue(Object value) {
        return value instanceof List<?> list ? list : List.of();
    }

    private static String code(String phase, String suffix) {
        return "data-distribution-" + phase + "-" + suffix;
    }

    private static String compact(double value) {
        return String.format(java.util.Locale.ROOT, "%.3f", value);
    }
}

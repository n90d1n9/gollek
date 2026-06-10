package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;

/**
 * Converts aggregate runtime-balance metadata into trainer performance recommendations.
 */
final class TrainingReportRuntimeProfileBalanceAdvisor {
    private TrainingReportRuntimeProfileBalanceAdvisor() {
    }

    static List<TrainingReportRecommendation> recommendations(Map<String, ?> report) {
        TrainingReportRuntimeProfileBalanceAssessment assessment =
                TrainingReportRuntimeProfileBalanceAssessment.assess(
                        TrainingReportRuntimeProfile.fromMetadata(TrainingReportReader.metadata(report)).balance());
        if (!assessment.available()) {
            return List.of();
        }
        if (assessment.inputBound()) {
            return List.of(inputBoundRecommendation(assessment));
        }
        if (assessment.optimizerBound()) {
            return List.of(optimizerBoundRecommendation(assessment));
        }
        if (assessment.validationBound()) {
            return List.of(validationBoundRecommendation(assessment));
        }
        if (assessment.trainComputeBound()) {
            return List.of(trainComputeBoundRecommendation(assessment));
        }
        return List.of();
    }

    private static TrainingReportRecommendation inputBoundRecommendation(
            TrainingReportRuntimeProfileBalanceAssessment assessment) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.HIGH,
                TrainingReportRecommendation.Category.DATA_HEALTH,
                TrainingReportDiagnostics.Severity.WARNING,
                "runtime_profile.balance.input_bound",
                "Reduce input pipeline wait before tuning model math",
                "The runtime balance profile shows input loading consumes at least half of measured trainer time.",
                List.of(
                        "Enable `DataLoader.prefetch(...)` or increase the prefetch buffer before changing model architecture.",
                        "Move decoding, augmentation, dynamic padding, or target conversion out of synchronous loader iteration.",
                        "Compare `runtimeProfile.balance.input.percentTotal` after prefetching to confirm compute overlap improved."),
                assessment.recommendationEvidence());
    }

    private static TrainingReportRecommendation optimizerBoundRecommendation(
            TrainingReportRuntimeProfileBalanceAssessment assessment) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.MEDIUM,
                TrainingReportRecommendation.Category.OPTIMIZATION,
                TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.balance.optimizer_bound",
                "Reduce optimizer-side overhead",
                "The runtime balance profile shows optimizer work is a large share of measured trainer time.",
                List.of(
                        "Inspect optimizer step, gradient diagnostics, clipping, and scheduler timings before changing model code.",
                        "Sample expensive parameter or gradient diagnostics during large runs instead of scanning every step.",
                        "Compare optimizer balance before and after optimizer or diagnostics changes."),
                assessment.recommendationEvidence());
    }

    private static TrainingReportRecommendation validationBoundRecommendation(
            TrainingReportRuntimeProfileBalanceAssessment assessment) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.MEDIUM,
                TrainingReportRecommendation.Category.VALIDATION,
                TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.balance.validation_bound",
                "Reduce validation-loop overhead",
                "The runtime balance profile shows validation consumes a large share of measured trainer time.",
                List.of(
                        "Review validation frequency and validation batch count before tuning training-loop compute.",
                        "Cache validation-only preprocessing or metric setup that does not change between batches.",
                        "Run a short validation-only profile to decide whether forward, loss, metrics, or input dominates."),
                assessment.recommendationEvidence());
    }

    private static TrainingReportRecommendation trainComputeBoundRecommendation(
            TrainingReportRuntimeProfileBalanceAssessment assessment) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.MEDIUM,
                TrainingReportRecommendation.Category.TRAINING_DYNAMICS,
                TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.balance.train_compute_bound",
                "Tune train compute with backend-aware benchmarks",
                "The runtime balance profile shows train compute dominates measured trainer time.",
                List.of(
                        "Compare CPU, Metal, and other available backends with the same model, batch size, and data.",
                        "Inspect forward/backward phase hotspots before changing optimizer or input pipeline settings.",
                        "Benchmark a smaller representative batch to separate tensor-kernel cost from data and reporting cost."),
                assessment.recommendationEvidence());
    }
}

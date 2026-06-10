package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds trainer recommendations from accelerator placement and fallback metadata.
 */
final class TrainingReportAccelerationAdvisor {
    private TrainingReportAccelerationAdvisor() {
    }

    static List<TrainingReportRecommendation> recommendations(Map<String, ?> report) {
        TrainingReportAcceleration acceleration = TrainingReportReader.accelerationView(report);
        if (!acceleration.available()) {
            return List.of();
        }
        if (acceleration.requestedAcceleratorUnavailable() || acceleration.requestedAcceleratorFellBack()) {
            return List.of(acceleratorFallbackRecommendation(acceleration));
        }
        if (acceleration.acceleratedWorkMissing()) {
            return List.of(acceleratedWorkMissingRecommendation(acceleration));
        }
        return List.of();
    }

    private static TrainingReportRecommendation acceleratorFallbackRecommendation(
            TrainingReportAcceleration acceleration) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.HIGH,
                TrainingReportRecommendation.Category.OPTIMIZATION,
                TrainingReportDiagnostics.Severity.WARNING,
                "acceleration.requested_backend_fallback",
                "Fix trainer accelerator fallback before tuning model quality",
                "The trainer requested an accelerator, but execution resolved to another backend.",
                List.of(
                        "Install or enable the requested backend, then rerun a short trainer smoke job.",
                        "If the fallback is intentional, set the trainer device to the observed backend to make reports explicit.",
                        "Compare throughput again after the requested backend is active before changing batch size or architecture."),
                evidence(acceleration));
    }

    private static TrainingReportRecommendation acceleratedWorkMissingRecommendation(
            TrainingReportAcceleration acceleration) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.MEDIUM,
                TrainingReportRecommendation.Category.OPTIMIZATION,
                TrainingReportDiagnostics.Severity.INFO,
                "acceleration.no_accelerated_matmul_delta",
                "Verify accelerated training kernels are being exercised",
                "The execution backend reported acceleration, but the trainer did not record new accelerated matmul calls.",
                List.of(
                        "Run a small profile with matrix-heavy batches and confirm accelerated matmul counters increase.",
                        "Check whether this model path uses operations that bypass the accelerated backend.",
                        "Inspect host-device transfer boundaries before assuming the accelerator is the bottleneck."),
                evidence(acceleration));
    }

    private static Map<String, Object> evidence(TrainingReportAcceleration acceleration) {
        Map<String, Object> evidence = new LinkedHashMap<>(acceleration.toMap());
        acceleration.acceleratedMatmulCallsDelta()
                .ifPresent(value -> evidence.put("acceleratedMatmulCallsDelta", value));
        return Map.copyOf(evidence);
    }
}

package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;

import java.util.Map;

/**
 * Type-safe view over parameter-update diagnostic sampling metadata.
 */
public record TrainingReportParameterUpdateDiagnosticsPolicy(
        boolean enabled,
        boolean sampled,
        int intervalSteps) {
    public TrainingReportParameterUpdateDiagnosticsPolicy {
        intervalSteps = Math.max(1, intervalSteps);
    }

    public static TrainingReportParameterUpdateDiagnosticsPolicy disabled() {
        return new TrainingReportParameterUpdateDiagnosticsPolicy(false, false, 1);
    }

    public static TrainingReportParameterUpdateDiagnosticsPolicy fromMetadata(Map<String, ?> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return disabled();
        }
        return new TrainingReportParameterUpdateDiagnosticsPolicy(
                booleanValue(metadata.get("parameterUpdateDiagnosticsEnabled")),
                booleanValue(metadata.get("parameterUpdateDiagnosticsSampled")),
                intValue(metadata.get("parameterUpdateDiagnosticsIntervalSteps"), 1));
    }

    public Map<String, Object> toMap() {
        return Map.of(
                "enabled", enabled,
                "sampled", sampled,
                "intervalSteps", intervalSteps);
    }
}

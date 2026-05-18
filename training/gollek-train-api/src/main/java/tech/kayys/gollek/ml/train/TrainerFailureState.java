package tech.kayys.gollek.ml.train;

import java.util.Map;

/**
 * Records the first terminal trainer data/metric failure and emits stable summary metadata.
 */
final class TrainerFailureState {
    private volatile boolean nonFiniteDetected;
    private volatile String nonFinitePhase;
    private volatile String nonFiniteKind;
    private volatile double nonFiniteValue = Double.NaN;
    private volatile String nonFiniteMessage;
    private volatile boolean nonFiniteOptimizerStepSkipped;
    private volatile boolean invalidBatchDetected;
    private volatile String invalidBatchPhase;
    private volatile String invalidBatchReason;
    private volatile String invalidBatchMessage;
    private volatile boolean invalidBatchOptimizerStepSkipped;
    private volatile boolean invalidLossShapeDetected;
    private volatile String invalidLossShapePhase;
    private volatile String invalidLossShapeMessage;
    private volatile String invalidLossShape;
    private volatile long invalidLossShapeElementCount;
    private volatile boolean invalidLossShapeOptimizerStepSkipped;
    private volatile boolean invalidMetricDetected;
    private volatile String invalidMetricPhase;
    private volatile String invalidMetricName;
    private volatile double invalidMetricValue = Double.NaN;
    private volatile String invalidMetricMessage;
    private volatile String invalidMetricKind;
    private volatile String invalidMetricDetailPath;
    private volatile String invalidMetricErrorType;
    private volatile RuntimeException invalidMetricCause;
    private volatile boolean invalidMetricOptimizerStepSkipped;

    void reset() {
        resetNonFinite();
        resetInvalidBatch();
        resetInvalidLossShape();
        resetInvalidMetric();
    }

    boolean terminalFailureDetected() {
        return nonFiniteDetected || invalidBatchDetected || invalidLossShapeDetected || invalidMetricDetected;
    }

    boolean nonFiniteDetected() {
        return nonFiniteDetected;
    }

    String nonFiniteMessage() {
        return nonFiniteMessage;
    }

    void throwIfNonFiniteDetected() {
        if (nonFiniteDetected) {
            throw new IllegalArgumentException(nonFiniteMessage);
        }
    }

    void throwIfInvalidMetricDetected() {
        if (invalidMetricDetected) {
            RuntimeException cause = invalidMetricCause;
            if (cause != null) {
                throw new IllegalArgumentException(invalidMetricMessage, cause);
            }
            throw new IllegalArgumentException(invalidMetricMessage);
        }
    }

    String recordNonFinite(
            String phase,
            String kind,
            double value,
            String label,
            boolean optimizerStepSkipped) {
        String safePhase = safe(phase, "unknown");
        String safeKind = safe(kind, "value");
        String message = safePhase + " " + label + " must be finite, got " + value;
        if (!nonFiniteDetected) {
            nonFiniteDetected = true;
            nonFinitePhase = safePhase;
            nonFiniteKind = safeKind;
            nonFiniteValue = value;
            nonFiniteMessage = message;
            nonFiniteOptimizerStepSkipped = optimizerStepSkipped;
        }
        return nonFiniteMessage;
    }

    String recordInvalidBatch(
            String phase,
            String reason,
            String message,
            boolean optimizerStepSkipped) {
        String safePhase = safe(phase, "unknown");
        String safeReason = safe(reason, "invalid");
        String safeMessage = message == null || message.isBlank()
                ? safePhase + " batch is invalid"
                : message;
        if (!invalidBatchDetected) {
            invalidBatchDetected = true;
            invalidBatchPhase = safePhase;
            invalidBatchReason = safeReason;
            invalidBatchMessage = safeMessage;
            invalidBatchOptimizerStepSkipped = optimizerStepSkipped;
        }
        return invalidBatchMessage;
    }

    String recordInvalidLossShape(
            String phase,
            String shape,
            long elements,
            boolean optimizerStepSkipped) {
        String safePhase = safe(phase, "unknown");
        String message = safePhase + " loss tensor must contain exactly one value before backward, got shape "
                + shape + " with " + elements + " values";
        if (!invalidLossShapeDetected) {
            invalidLossShapeDetected = true;
            invalidLossShapePhase = safePhase;
            invalidLossShape = shape;
            invalidLossShapeElementCount = elements;
            invalidLossShapeMessage = message;
            invalidLossShapeOptimizerStepSkipped = optimizerStepSkipped;
        }
        return invalidLossShapeMessage;
    }

    String recordInvalidMetricValue(
            String phase,
            String metricName,
            double value,
            boolean optimizerStepSkipped) {
        String safePhase = safe(phase, "unknown");
        String safeMetricName = safe(metricName, "metric");
        String message = safePhase + " metric " + safeMetricName + " must be finite, got " + value;
        if (!invalidMetricDetected) {
            invalidMetricDetected = true;
            invalidMetricPhase = safePhase;
            invalidMetricName = safeMetricName;
            invalidMetricValue = value;
            invalidMetricMessage = message;
            invalidMetricKind = "value";
            invalidMetricDetailPath = null;
            invalidMetricErrorType = null;
            invalidMetricCause = null;
            invalidMetricOptimizerStepSkipped = optimizerStepSkipped;
        }
        return invalidMetricMessage;
    }

    String recordInvalidMetricDetail(
            String phase,
            String metricName,
            String detailPath,
            double value) {
        String safePhase = safe(phase, "unknown");
        String safeMetricName = safe(metricName, "metric");
        String safeDetailPath = safe(detailPath, "details");
        String message = safePhase + " metric " + safeMetricName + " detail "
                + safeDetailPath + " must be finite, got " + value;
        if (!invalidMetricDetected) {
            invalidMetricDetected = true;
            invalidMetricPhase = safePhase;
            invalidMetricName = safeMetricName;
            invalidMetricValue = value;
            invalidMetricMessage = message;
            invalidMetricKind = "detail";
            invalidMetricDetailPath = safeDetailPath;
            invalidMetricErrorType = null;
            invalidMetricCause = null;
            invalidMetricOptimizerStepSkipped = false;
        }
        return invalidMetricMessage;
    }

    String recordInvalidMetricName(String phase, String metricName) {
        String safePhase = safe(phase, "unknown");
        String safeMetricName = safe(metricName, "metric");
        String message = safePhase + " metric name must be unique, duplicate: " + safeMetricName;
        if (!invalidMetricDetected) {
            invalidMetricDetected = true;
            invalidMetricPhase = safePhase;
            invalidMetricName = safeMetricName;
            invalidMetricValue = Double.NaN;
            invalidMetricMessage = message;
            invalidMetricKind = "name";
            invalidMetricDetailPath = null;
            invalidMetricErrorType = null;
            invalidMetricCause = null;
            invalidMetricOptimizerStepSkipped = false;
        }
        return invalidMetricMessage;
    }

    String recordInvalidMetricFailure(
            String phase,
            String metricName,
            String kind,
            String detailPath,
            RuntimeException error) {
        String safePhase = safe(phase, "unknown");
        String safeMetricName = safe(metricName, "metric");
        String safeKind = safe(kind, "value");
        String safeDetailPath = detailPath == null || detailPath.isBlank() ? null : detailPath;
        String errorMessage = error.getMessage() == null || error.getMessage().isBlank()
                ? error.getClass().getSimpleName()
                : error.getMessage();
        String target = "detail".equals(safeKind)
                ? " detail " + safeDetailPath
                : "";
        String message = safePhase + " metric " + safeMetricName + target
                + " failed to produce a value: " + errorMessage;
        if (!invalidMetricDetected) {
            invalidMetricDetected = true;
            invalidMetricPhase = safePhase;
            invalidMetricName = safeMetricName;
            invalidMetricValue = Double.NaN;
            invalidMetricMessage = message;
            invalidMetricKind = safeKind;
            invalidMetricDetailPath = safeDetailPath;
            invalidMetricErrorType = error.getClass().getSimpleName();
            invalidMetricCause = error;
            invalidMetricOptimizerStepSkipped = false;
        }
        return invalidMetricMessage;
    }

    void putMetadata(Map<String, Object> metadata) {
        metadata.put("nonFiniteGuardEnabled", true);
        metadata.put("nonFiniteDetected", nonFiniteDetected);
        if (nonFiniteDetected) {
            metadata.put("nonFinitePhase", nonFinitePhase);
            metadata.put("nonFiniteKind", nonFiniteKind);
            metadata.put("nonFiniteValue", nonFiniteValue);
            metadata.put("nonFiniteMessage", nonFiniteMessage);
            metadata.put("nonFiniteOptimizerStepSkipped", nonFiniteOptimizerStepSkipped);
            metadata.put("stopReason", "non-finite-" + nonFinitePhase + "-" + nonFiniteKind);
        }
        metadata.put("batchDataGuardEnabled", true);
        metadata.put("invalidBatchDetected", invalidBatchDetected);
        if (invalidBatchDetected) {
            metadata.put("invalidBatchPhase", invalidBatchPhase);
            metadata.put("invalidBatchReason", invalidBatchReason);
            metadata.put("invalidBatchMessage", invalidBatchMessage);
            metadata.put("invalidBatchOptimizerStepSkipped", invalidBatchOptimizerStepSkipped);
            metadata.put("stopReason", "invalid-batch-" + invalidBatchPhase + "-" + invalidBatchReason);
        }
        metadata.put("lossShapeGuardEnabled", true);
        metadata.put("invalidLossShapeDetected", invalidLossShapeDetected);
        if (invalidLossShapeDetected) {
            metadata.put("invalidLossShapePhase", invalidLossShapePhase);
            metadata.put("invalidLossShape", invalidLossShape);
            metadata.put("invalidLossShapeElementCount", invalidLossShapeElementCount);
            metadata.put("invalidLossShapeMessage", invalidLossShapeMessage);
            metadata.put("invalidLossShapeOptimizerStepSkipped", invalidLossShapeOptimizerStepSkipped);
            metadata.put("stopReason", "invalid-loss-shape-" + invalidLossShapePhase);
        }
        metadata.put("metricFiniteGuardEnabled", true);
        metadata.put("invalidMetricDetected", invalidMetricDetected);
        if (invalidMetricDetected) {
            metadata.put("failed", true);
            metadata.put("invalidMetricPhase", invalidMetricPhase);
            metadata.put("invalidMetricName", invalidMetricName);
            metadata.put("invalidMetricValue", invalidMetricValue);
            metadata.put("invalidMetricMessage", invalidMetricMessage);
            metadata.put("invalidMetricKind", invalidMetricKind);
            if (invalidMetricDetailPath != null) {
                metadata.put("invalidMetricDetailPath", invalidMetricDetailPath);
            }
            if (invalidMetricErrorType != null) {
                metadata.put("invalidMetricErrorType", invalidMetricErrorType);
            }
            metadata.put("invalidMetricOptimizerStepSkipped", invalidMetricOptimizerStepSkipped);
            metadata.put("stopReason", "invalid-metric-" + invalidMetricPhase + "-" + invalidMetricName);
        }
    }

    private void resetNonFinite() {
        nonFiniteDetected = false;
        nonFinitePhase = null;
        nonFiniteKind = null;
        nonFiniteValue = Double.NaN;
        nonFiniteMessage = null;
        nonFiniteOptimizerStepSkipped = false;
    }

    private void resetInvalidBatch() {
        invalidBatchDetected = false;
        invalidBatchPhase = null;
        invalidBatchReason = null;
        invalidBatchMessage = null;
        invalidBatchOptimizerStepSkipped = false;
    }

    private void resetInvalidLossShape() {
        invalidLossShapeDetected = false;
        invalidLossShapePhase = null;
        invalidLossShapeMessage = null;
        invalidLossShape = null;
        invalidLossShapeElementCount = 0L;
        invalidLossShapeOptimizerStepSkipped = false;
    }

    private void resetInvalidMetric() {
        invalidMetricDetected = false;
        invalidMetricPhase = null;
        invalidMetricName = null;
        invalidMetricValue = Double.NaN;
        invalidMetricMessage = null;
        invalidMetricKind = null;
        invalidMetricDetailPath = null;
        invalidMetricErrorType = null;
        invalidMetricCause = null;
        invalidMetricOptimizerStepSkipped = false;
    }

    private static String safe(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }
}

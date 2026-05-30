package tech.kayys.gollek.train.data.multimodal;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Aggregate audit report for a train/validation/test multimodal split.
 */
public record MultimodalThreeWaySplitReport(
        int trainSampleCount,
        int validationSampleCount,
        int testSampleCount,
        MultimodalSplitReport trainValidationReport,
        MultimodalSplitReport trainTestReport,
        MultimodalSplitReport validationTestReport) {
    public MultimodalThreeWaySplitReport {
        if (trainSampleCount < 1 || validationSampleCount < 1 || testSampleCount < 1) {
            throw new IllegalArgumentException("train, validation, and test sample counts must all be positive");
        }
        trainValidationReport = Objects.requireNonNull(
                trainValidationReport,
                "trainValidationReport must not be null");
        trainTestReport = Objects.requireNonNull(trainTestReport, "trainTestReport must not be null");
        validationTestReport = Objects.requireNonNull(
                validationTestReport,
                "validationTestReport must not be null");
    }

    public int totalSampleCount() {
        return Math.addExact(Math.addExact(trainSampleCount, validationSampleCount), testSampleCount);
    }

    public boolean isLeakageFree() {
        return leakingSourcePaths().isEmpty();
    }

    public Set<String> leakingSourcePaths() {
        TreeSet<String> paths = new TreeSet<>();
        paths.addAll(trainValidationReport.overlappingSourcePaths());
        paths.addAll(trainTestReport.overlappingSourcePaths());
        paths.addAll(validationTestReport.overlappingSourcePaths());
        return Set.copyOf(paths);
    }

    public double maxSampleSignatureShareDelta() {
        return Math.max(
                trainValidationReport.maxSampleSignatureShareDelta(),
                Math.max(
                        trainTestReport.maxSampleSignatureShareDelta(),
                        validationTestReport.maxSampleSignatureShareDelta()));
    }

    public double maxMimeTypeShareDelta() {
        return Math.max(
                trainValidationReport.maxMimeTypeShareDelta(),
                Math.max(
                        trainTestReport.maxMimeTypeShareDelta(),
                        validationTestReport.maxMimeTypeShareDelta()));
    }

    public boolean isSignatureBalanced(double tolerance) {
        MultimodalSplitReport.requireTolerance(tolerance);
        return maxSampleSignatureShareDelta() <= tolerance;
    }

    public boolean isMimeTypeBalanced(double tolerance) {
        MultimodalSplitReport.requireTolerance(tolerance);
        return maxMimeTypeShareDelta() <= tolerance;
    }

    public void throwIfInvalid(double signatureTolerance, double mimeTypeTolerance) {
        MultimodalSplitReport.requireTolerance(signatureTolerance);
        MultimodalSplitReport.requireTolerance(mimeTypeTolerance);
        List<String> failures = new ArrayList<>();
        if (!isLeakageFree()) {
            failures.add("source leakage: " + leakingSourcePaths());
        }
        if (!isSignatureBalanced(signatureTolerance)) {
            failures.add("signature drift " + maxSampleSignatureShareDelta()
                    + " exceeds tolerance " + signatureTolerance);
        }
        if (!isMimeTypeBalanced(mimeTypeTolerance)) {
            failures.add("mime-type drift " + maxMimeTypeShareDelta()
                    + " exceeds tolerance " + mimeTypeTolerance);
        }
        if (!failures.isEmpty()) {
            throw new IllegalStateException("multimodal train/validation/test audit failed: " + failures);
        }
    }
}

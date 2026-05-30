package tech.kayys.gollek.train.data.multimodal;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * Aggregate audit report for a full multimodal cross-validation run.
 */
public record MultimodalCrossValidationReport(List<MultimodalSplitReport> foldReports) {
    public MultimodalCrossValidationReport {
        if (foldReports == null || foldReports.isEmpty()) {
            throw new IllegalArgumentException("foldReports must not be null or empty");
        }
        foldReports = Collections.unmodifiableList(new ArrayList<>(foldReports));
        for (MultimodalSplitReport report : foldReports) {
            if (report == null) {
                throw new IllegalArgumentException("foldReports must not contain null");
            }
        }
    }

    public int foldCount() {
        return foldReports.size();
    }

    public int totalValidationSamples() {
        int total = 0;
        for (MultimodalSplitReport report : foldReports) {
            total = Math.addExact(total, report.validationReport().sampleCount());
        }
        return total;
    }

    public boolean isLeakageFree() {
        return foldsWithSourceLeakage().isEmpty();
    }

    public List<Integer> foldsWithSourceLeakage() {
        List<Integer> folds = new ArrayList<>();
        for (int i = 0; i < foldReports.size(); i++) {
            if (foldReports.get(i).hasSourceLeakage()) {
                folds.add(i);
            }
        }
        return Collections.unmodifiableList(folds);
    }

    public Set<String> leakingSourcePaths() {
        TreeSet<String> paths = new TreeSet<>();
        for (MultimodalSplitReport report : foldReports) {
            paths.addAll(report.overlappingSourcePaths());
        }
        return Collections.unmodifiableSet(paths);
    }

    public double maxSampleSignatureShareDelta() {
        double max = 0.0;
        for (MultimodalSplitReport report : foldReports) {
            max = Math.max(max, report.maxSampleSignatureShareDelta());
        }
        return max;
    }

    public double maxMimeTypeShareDelta() {
        double max = 0.0;
        for (MultimodalSplitReport report : foldReports) {
            max = Math.max(max, report.maxMimeTypeShareDelta());
        }
        return max;
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
            failures.add("source leakage in folds " + foldsWithSourceLeakage() + ": " + leakingSourcePaths());
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
            throw new IllegalStateException("multimodal cross-validation audit failed: " + failures);
        }
    }
}

package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Non-throwing compatibility report for replaying a persisted multimodal split manifest.
 */
public record MultimodalManifestAuditReport(
        int expectedSampleCount,
        int actualSampleCount,
        List<SampleMismatch> sampleMismatches) {
    public MultimodalManifestAuditReport {
        if (expectedSampleCount < 0) {
            throw new IllegalArgumentException("expectedSampleCount must be non-negative");
        }
        if (actualSampleCount < 0) {
            throw new IllegalArgumentException("actualSampleCount must be non-negative");
        }
        sampleMismatches = List.copyOf(
                Objects.requireNonNull(sampleMismatches, "sampleMismatches must not be null"));
    }

    static MultimodalManifestAuditReport inspect(
            int expectedSampleCount,
            List<MultimodalSplitManifest.SampleFingerprint> expectedSamples,
            Dataset<? extends List<MultimodalContent>> dataset) {
        Objects.requireNonNull(expectedSamples, "expectedSamples must not be null");
        Objects.requireNonNull(dataset, "dataset must not be null");

        int checkedSamples = Math.min(Math.min(expectedSampleCount, expectedSamples.size()), dataset.size());
        List<SampleMismatch> mismatches = new ArrayList<>();
        for (int index = 0; index < checkedSamples; index++) {
            MultimodalSplitManifest.SampleFingerprint expected = expectedSamples.get(index);
            MultimodalSplitManifest.SampleFingerprint actual =
                    MultimodalSplitManifest.fingerprint(index, dataset.get(index));
            if (!expected.equals(actual)) {
                mismatches.add(SampleMismatch.from(expected, actual));
            }
        }
        return new MultimodalManifestAuditReport(expectedSampleCount, dataset.size(), mismatches);
    }

    public boolean matches() {
        return !hasSampleCountMismatch() && sampleMismatches.isEmpty();
    }

    public boolean hasSampleCountMismatch() {
        return expectedSampleCount != actualSampleCount;
    }

    public int checkedSampleCount() {
        return Math.min(expectedSampleCount, actualSampleCount);
    }

    public int missingSampleCount() {
        return Math.max(0, expectedSampleCount - actualSampleCount);
    }

    public int extraSampleCount() {
        return Math.max(0, actualSampleCount - expectedSampleCount);
    }

    public Optional<SampleMismatch> firstMismatch() {
        return sampleMismatches.stream().findFirst();
    }

    public void throwIfInvalid() {
        if (matches()) {
            return;
        }
        StringBuilder message = new StringBuilder("multimodal manifest does not match dataset: ")
                .append(summary());
        firstMismatch().ifPresent(mismatch -> message
                .append(", firstMismatchIndex=")
                .append(mismatch.index()));
        throw new IllegalArgumentException(message.toString());
    }

    public String summary() {
        return "expectedSamples=" + expectedSampleCount
                + ", actualSamples=" + actualSampleCount
                + ", missingSamples=" + missingSampleCount()
                + ", extraSamples=" + extraSampleCount()
                + ", fingerprintMismatches=" + sampleMismatches.size();
    }

    public record SampleMismatch(
            int index,
            String expectedSignature,
            String actualSignature,
            String expectedDigest,
            String actualDigest,
            List<String> expectedSourcePaths,
            List<String> actualSourcePaths,
            List<String> expectedMimeTypes,
            List<String> actualMimeTypes) {
        public SampleMismatch {
            if (index < 0) {
                throw new IllegalArgumentException("index must be non-negative");
            }
            expectedSignature = Objects.requireNonNull(expectedSignature, "expectedSignature must not be null");
            actualSignature = Objects.requireNonNull(actualSignature, "actualSignature must not be null");
            expectedDigest = Objects.requireNonNull(expectedDigest, "expectedDigest must not be null");
            actualDigest = Objects.requireNonNull(actualDigest, "actualDigest must not be null");
            expectedSourcePaths = List.copyOf(
                    Objects.requireNonNull(expectedSourcePaths, "expectedSourcePaths must not be null"));
            actualSourcePaths = List.copyOf(
                    Objects.requireNonNull(actualSourcePaths, "actualSourcePaths must not be null"));
            expectedMimeTypes = List.copyOf(
                    Objects.requireNonNull(expectedMimeTypes, "expectedMimeTypes must not be null"));
            actualMimeTypes = List.copyOf(
                    Objects.requireNonNull(actualMimeTypes, "actualMimeTypes must not be null"));
        }

        static SampleMismatch from(
                MultimodalSplitManifest.SampleFingerprint expected,
                MultimodalSplitManifest.SampleFingerprint actual) {
            return new SampleMismatch(
                    expected.index(),
                    expected.signature(),
                    actual.signature(),
                    expected.digest(),
                    actual.digest(),
                    expected.sourcePaths(),
                    actual.sourcePaths(),
                    expected.mimeTypes(),
                    actual.mimeTypes());
        }

        public boolean signatureChanged() {
            return !expectedSignature.equals(actualSignature);
        }

        public boolean contentChanged() {
            return !expectedDigest.equals(actualDigest);
        }

        public boolean sourcePathsChanged() {
            return !expectedSourcePaths.equals(actualSourcePaths);
        }

        public boolean mimeTypesChanged() {
            return !expectedMimeTypes.equals(actualMimeTypes);
        }
    }
}

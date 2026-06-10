package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Resume/checkpoint guard for comparing expected and actual dataset fingerprints.
 */
public record DiscreteTokenDatasetFingerprintMatch(
        DiscreteTokenDatasetFingerprint expected,
        DiscreteTokenDatasetFingerprint actual) {

    public DiscreteTokenDatasetFingerprintMatch {
        expected = Objects.requireNonNull(expected, "expected must not be null");
        actual = Objects.requireNonNull(actual, "actual must not be null");
    }

    public static DiscreteTokenDatasetFingerprintMatch verify(
            DiscreteTokenDatasetFingerprint expected,
            DiscreteTokenDatasetFingerprint actual) {
        return new DiscreteTokenDatasetFingerprintMatch(expected, actual);
    }

    public static DiscreteTokenDatasetFingerprintMatch verify(
            DiscreteTokenDatasetFingerprint expected,
            DiscreteTokenDatasetPlan plan) {
        Objects.requireNonNull(plan, "plan must not be null");
        return verify(expected, plan.fingerprint());
    }

    public static DiscreteTokenDatasetFingerprintMatch verify(
            DiscreteTokenDatasetFingerprint expected,
            DiscreteTokenDatasetPlanReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return verify(expected, report.fingerprint());
    }

    public static DiscreteTokenDatasetFingerprintMatch verifyMetadata(
            Map<?, ?> expectedMetadata,
            DiscreteTokenDatasetPlan plan) {
        return verify(DiscreteTokenDatasetFingerprint.fromMetadata(expectedMetadata), plan);
    }

    public static DiscreteTokenDatasetFingerprintMatch fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        DiscreteTokenDatasetFingerprintMatch match = verify(
                DiscreteTokenDatasetFingerprint.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "expected")),
                DiscreteTokenDatasetFingerprint.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "actual")));
        verifyOptionalStatus(metadata, match);
        return match;
    }

    public static DiscreteTokenDatasetFingerprintMatch verifyMetadata(
            Map<?, ?> expectedMetadata,
            DiscreteTokenDatasetPlanReport report) {
        return verify(DiscreteTokenDatasetFingerprint.fromMetadata(expectedMetadata), report);
    }

    public static DiscreteTokenDatasetFingerprintMatch verifyMetadataSection(
            Map<?, ?> expectedMetadata,
            DiscreteTokenDatasetPlan plan) {
        return verifyMetadataSection(expectedMetadata, DiscreteTokenDatasetFingerprint.METADATA_KEY, plan);
    }

    public static DiscreteTokenDatasetFingerprintMatch verifyMetadataSection(
            Map<?, ?> expectedMetadata,
            String key,
            DiscreteTokenDatasetPlan plan) {
        return verify(DiscreteTokenDatasetFingerprint.fromMetadataSection(expectedMetadata, key), plan);
    }

    public static DiscreteTokenDatasetFingerprintMatch verifyMetadataSection(
            Map<?, ?> expectedMetadata,
            DiscreteTokenDatasetPlanReport report) {
        return verifyMetadataSection(expectedMetadata, DiscreteTokenDatasetFingerprint.METADATA_KEY, report);
    }

    public static DiscreteTokenDatasetFingerprintMatch verifyMetadataSection(
            Map<?, ?> expectedMetadata,
            String key,
            DiscreteTokenDatasetPlanReport report) {
        return verify(DiscreteTokenDatasetFingerprint.fromMetadataSection(expectedMetadata, key), report);
    }

    public boolean matches() {
        return expected.equals(actual);
    }

    public String status() {
        return matches() ? "matched" : "mismatched";
    }

    public List<String> mismatchReasons() {
        if (matches()) {
            return List.of();
        }

        List<String> reasons = new ArrayList<>();
        if (!expected.algorithm().equals(actual.algorithm())) {
            reasons.add("fingerprint algorithm differs");
        }
        if (!expected.value().equals(actual.value())) {
            reasons.add("fingerprint value differs");
        }
        if (expected.exampleCount() != actual.exampleCount()) {
            reasons.add("fingerprint example count differs");
        }
        return List.copyOf(reasons);
    }

    public String summary() {
        if (matches()) {
            return "dataset fingerprint matched: " + actual.shortValue();
        }
        return "dataset fingerprint mismatched: expected "
                + expected.shortValue()
                + " but actual "
                + actual.shortValue()
                + " ("
                + String.join("; ", mismatchReasons())
                + ")";
    }

    public void requireMatch() {
        if (!matches()) {
            throw new IllegalStateException(summary());
        }
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status());
        metadata.put("matches", matches());
        metadata.put("expected", expected.toMetadata());
        metadata.put("actual", actual.toMetadata());
        metadata.put("mismatchReasons", mismatchReasons());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void verifyOptionalStatus(Map<?, ?> metadata, DiscreteTokenDatasetFingerprintMatch match) {
        if (metadata.containsKey("status") && metadata.get("status") != null) {
            Object value = metadata.get("status");
            if (!(value instanceof CharSequence text) || !match.status().equals(text.toString())) {
                throw new IllegalArgumentException("metadata field 'status' does not match fingerprint comparison");
            }
        }
        if (metadata.containsKey("matches") && metadata.get("matches") != null) {
            Object value = metadata.get("matches");
            if (!(value instanceof Boolean flag) || flag.booleanValue() != match.matches()) {
                throw new IllegalArgumentException("metadata field 'matches' does not match fingerprint comparison");
            }
        }
    }

}

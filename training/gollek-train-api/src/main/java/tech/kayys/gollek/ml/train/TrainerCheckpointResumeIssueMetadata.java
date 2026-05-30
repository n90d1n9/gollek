package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds the structured trainer summary fields that explain checkpoint resume issues.
 */
final class TrainerCheckpointResumeIssueMetadata {
    private TrainerCheckpointResumeIssueMetadata() {
    }

    static Snapshot snapshot(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        List<String> missing = immutableList(missingArtifacts);
        List<String> mismatches = immutableList(compatibilityMismatches);
        List<String> manifestEntryMissing = immutableList(manifestEntryMissingArtifacts);
        return new Snapshot(
                missing,
                mismatches,
                manifestEntryMissing,
                !missing.isEmpty() || !mismatches.isEmpty() || !manifestEntryMissing.isEmpty(),
                issueKinds(missing, mismatches, manifestEntryMissing),
                affectedArtifacts(missing, mismatches, manifestEntryMissing),
                issueDetails(missing, mismatches, manifestEntryMissing),
                issueCodes(missing, mismatches, manifestEntryMissing),
                issueSeverities(missing, mismatches, manifestEntryMissing),
                recommendedActions(missing, mismatches, manifestEntryMissing),
                issueCountsByKind(missing, mismatches, manifestEntryMissing),
                issueSeveritiesByKind(missing, mismatches, manifestEntryMissing),
                issueArtifactsByKind(missing, mismatches, manifestEntryMissing));
    }

    private static List<String> issueKinds(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        List<String> kinds = new ArrayList<>(3);
        if (!missingArtifacts.isEmpty()) {
            kinds.add(TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.kind());
        }
        if (!compatibilityMismatches.isEmpty()) {
            kinds.add(TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH.kind());
        }
        if (!manifestEntryMissingArtifacts.isEmpty()) {
            kinds.add(TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.kind());
        }
        return List.copyOf(kinds);
    }

    private static Map<String, Integer> issueCountsByKind(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        addCount(counts, TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT, missingArtifacts.size());
        addCount(counts, TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH, compatibilityMismatches.size());
        addCount(counts, TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING, manifestEntryMissingArtifacts.size());
        return stableMap(counts);
    }

    private static Map<String, List<String>> issueArtifactsByKind(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        Map<String, List<String>> artifacts = new LinkedHashMap<>();
        addArtifactGroup(artifacts, TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT, missingArtifacts);
        addArtifactGroup(
                artifacts,
                TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH,
                compatibilityMismatches.stream()
                        .map(TrainerCheckpointResumeIssueMetadata::compatibilityMismatchArtifact)
                        .toList());
        addArtifactGroup(artifacts, TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING, manifestEntryMissingArtifacts);
        return stableMap(artifacts);
    }

    private static List<String> affectedArtifacts(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        List<String> artifacts = new ArrayList<>();
        missingArtifacts.forEach(artifact -> addUniqueNonBlank(artifacts, artifact));
        compatibilityMismatches.forEach(mismatch -> addUniqueNonBlank(
                artifacts,
                compatibilityMismatchArtifact(mismatch)));
        manifestEntryMissingArtifacts.forEach(artifact -> addUniqueNonBlank(artifacts, artifact));
        return List.copyOf(artifacts);
    }

    private static List<TrainerCheckpointResumeIssue> issueDetails(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        List<TrainerCheckpointResumeIssue> issues = new ArrayList<>();
        missingArtifacts.forEach(artifact -> issues.add(issue(
                TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT,
                artifact,
                TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.artifactMessage(artifact))));
        compatibilityMismatches.forEach(mismatch -> issues.add(issue(
                TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH,
                compatibilityMismatchArtifact(mismatch),
                compatibilityMismatchMessage(mismatch))));
        manifestEntryMissingArtifacts.forEach(artifact -> issues.add(issue(
                TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING,
                artifact,
                TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.artifactMessage(artifact))));
        return List.copyOf(issues);
    }

    private static List<String> issueCodes(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        List<String> codes = new ArrayList<>(3);
        if (!missingArtifacts.isEmpty()) {
            addUniqueNonBlank(codes, TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.code());
        }
        if (!compatibilityMismatches.isEmpty()) {
            addUniqueNonBlank(codes, TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH.code());
        }
        if (!manifestEntryMissingArtifacts.isEmpty()) {
            addUniqueNonBlank(codes, TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.code());
        }
        return List.copyOf(codes);
    }

    private static List<String> recommendedActions(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        List<String> actions = new ArrayList<>(3);
        if (!missingArtifacts.isEmpty()) {
            addUniqueNonBlank(actions, TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.action());
        }
        if (!compatibilityMismatches.isEmpty()) {
            addUniqueNonBlank(actions, TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH.action());
        }
        if (!manifestEntryMissingArtifacts.isEmpty()) {
            addUniqueNonBlank(actions, TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.action());
        }
        return List.copyOf(actions);
    }

    private static List<String> issueSeverities(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        List<String> severities = new ArrayList<>(3);
        if (!missingArtifacts.isEmpty()) {
            addUniqueNonBlank(severities, TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT.severity());
        }
        if (!compatibilityMismatches.isEmpty()) {
            addUniqueNonBlank(severities, TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH.severity());
        }
        if (!manifestEntryMissingArtifacts.isEmpty()) {
            addUniqueNonBlank(severities, TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING.severity());
        }
        return List.copyOf(severities);
    }

    private static Map<String, String> issueSeveritiesByKind(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts) {
        Map<String, String> severities = new LinkedHashMap<>();
        addSeverity(severities, TrainerCheckpointResumeIssueKind.MISSING_ARTIFACT, missingArtifacts.size());
        addSeverity(severities, TrainerCheckpointResumeIssueKind.COMPATIBILITY_MISMATCH, compatibilityMismatches.size());
        addSeverity(severities, TrainerCheckpointResumeIssueKind.MANIFEST_ENTRY_MISSING, manifestEntryMissingArtifacts.size());
        return stableMap(severities);
    }

    private static Map<String, Boolean> issueBlockingByKind(
            List<TrainerCheckpointResumeIssue> issues) {
        Map<String, Boolean> blocking = new LinkedHashMap<>();
        issues.forEach(issue -> blocking.put(issue.kindValue(), issue.blocking()));
        return stableMap(blocking);
    }

    private static String compatibilityMismatchArtifact(String mismatch) {
        if (mismatch == null) {
            return null;
        }
        int separator = mismatch.indexOf(": ");
        return (separator > 0 ? mismatch.substring(0, separator) : mismatch).trim();
    }

    private static String compatibilityMismatchMessage(String mismatch) {
        if (mismatch == null) {
            return null;
        }
        int separator = mismatch.indexOf(": ");
        return (separator > 0 ? mismatch.substring(separator + 2) : mismatch).trim();
    }

    private static TrainerCheckpointResumeIssue issue(
            TrainerCheckpointResumeIssueKind kind,
            String artifact,
            String message) {
        return new TrainerCheckpointResumeIssue(kind, artifact, message);
    }

    private static void addCount(Map<String, Integer> counts, TrainerCheckpointResumeIssueKind kind, int count) {
        if (count > 0) {
            counts.put(kind.kind(), count);
        }
    }

    private static void addSeverity(Map<String, String> severities, TrainerCheckpointResumeIssueKind kind, int count) {
        if (count > 0) {
            severities.put(kind.kind(), kind.severity());
        }
    }

    private static void addArtifactGroup(
            Map<String, List<String>> groups,
            TrainerCheckpointResumeIssueKind kind,
            List<String> artifacts) {
        List<String> values = new ArrayList<>();
        artifacts.forEach(artifact -> addUniqueNonBlank(values, artifact));
        if (!values.isEmpty()) {
            groups.put(kind.kind(), List.copyOf(values));
        }
    }

    private static void addUniqueNonBlank(List<String> values, String value) {
        if (value == null || value.isBlank()) {
            return;
        }
        String normalized = value.trim();
        if (!values.contains(normalized)) {
            values.add(normalized);
        }
    }

    private static List<String> immutableList(List<String> values) {
        return values == null || values.isEmpty() ? List.of() : List.copyOf(values);
    }

    private static <K, V> Map<K, V> stableMap(Map<K, V> values) {
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }

    record Snapshot(
            List<String> missingArtifacts,
            List<String> compatibilityMismatches,
            List<String> manifestEntryMissingArtifacts,
            boolean issueDetected,
            List<String> issueKinds,
            List<String> affectedArtifacts,
            List<TrainerCheckpointResumeIssue> issueDetails,
            List<String> issueCodes,
            List<String> issueSeverities,
            List<String> recommendedActions,
            Map<String, Integer> issueCountsByKind,
            Map<String, String> issueSeveritiesByKind,
            Map<String, List<String>> issueArtifactsByKind) {
        int issueCount() {
            return issueDetails.size();
        }

        List<Map<String, Object>> issues() {
            return issueDetails.stream()
                    .map(TrainerCheckpointResumeIssue::toMetadataMap)
                    .toList();
        }

        List<Map<String, Object>> blockingIssues() {
            return issueDetails.stream()
                    .filter(TrainerCheckpointResumeIssue::blocking)
                    .map(TrainerCheckpointResumeIssue::toMetadataMap)
                    .toList();
        }

        boolean hasCompatibilityMismatches() {
            return !compatibilityMismatches.isEmpty();
        }

        boolean hasManifestEntryMissingArtifacts() {
            return !manifestEntryMissingArtifacts.isEmpty();
        }

        boolean primaryIssueAvailable() {
            return !issueDetails.isEmpty();
        }

        boolean blockingIssueDetected() {
            return issueDetails.stream().anyMatch(TrainerCheckpointResumeIssue::blocking);
        }

        int blockingIssueCount() {
            return (int) issueDetails.stream()
                    .filter(TrainerCheckpointResumeIssue::blocking)
                    .count();
        }

        List<String> blockingIssueKinds() {
            List<String> kinds = new ArrayList<>();
            issueDetails.stream()
                    .filter(TrainerCheckpointResumeIssue::blocking)
                    .map(TrainerCheckpointResumeIssue::kindValue)
                    .forEach(kind -> addUniqueNonBlank(kinds, kind));
            return List.copyOf(kinds);
        }

        Map<String, Boolean> issueBlockingByKind() {
            return TrainerCheckpointResumeIssueMetadata.issueBlockingByKind(issueDetails);
        }

        String primaryIssueKind() {
            return primaryIssueValue("kind");
        }

        String primaryIssueCode() {
            return primaryIssueValue("code");
        }

        String primaryIssueSeverity() {
            return primaryIssueValue("severity");
        }

        boolean primaryIssueBlocking() {
            return primaryIssueAvailable() && issueDetails.getFirst().blocking();
        }

        boolean primaryBlockingIssueAvailable() {
            return primaryBlockingIssue() != null;
        }

        String primaryBlockingIssueKind() {
            return primaryBlockingIssueValue("kind");
        }

        String primaryBlockingIssueCode() {
            return primaryBlockingIssueValue("code");
        }

        String primaryBlockingIssueSeverity() {
            return primaryBlockingIssueValue("severity");
        }

        String primaryBlockingAffectedArtifact() {
            return primaryBlockingIssueValue("artifact");
        }

        String primaryBlockingIssueMessage() {
            return primaryBlockingIssueValue("message");
        }

        String primaryBlockingRecommendedAction() {
            return primaryBlockingIssueValue("action");
        }

        String primaryAffectedArtifact() {
            return primaryIssueValue("artifact");
        }

        String primaryIssueMessage() {
            return primaryIssueValue("message");
        }

        String primaryRecommendedAction() {
            return primaryIssueValue("action");
        }

        private String primaryIssueValue(String key) {
            if (issueDetails.isEmpty()) {
                return null;
            }
            String value = issueDetails.getFirst().value(key);
            return value != null && !value.isBlank() ? value : null;
        }

        private String primaryBlockingIssueValue(String key) {
            TrainerCheckpointResumeIssue issue = primaryBlockingIssue();
            if (issue == null) {
                return null;
            }
            String value = issue.value(key);
            return value != null && !value.isBlank() ? value : null;
        }

        private TrainerCheckpointResumeIssue primaryBlockingIssue() {
            return issueDetails.stream()
                    .filter(TrainerCheckpointResumeIssue::blocking)
                    .findFirst()
                    .orElse(null);
        }
    }
}

package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Read-side inventory of recursive-reasoning checkpoint directories under one root.
 */
public record DiscreteTokenDatasetTrainerCheckpointInventory(
        Path rootDir,
        List<DiscreteTokenDatasetTrainerCheckpointSnapshot> checkpoints,
        List<ReadFailure> failures) {

    public DiscreteTokenDatasetTrainerCheckpointInventory {
        rootDir = Objects.requireNonNull(rootDir, "rootDir must not be null");
        checkpoints = sortedSnapshots(checkpoints);
        failures = sortedFailures(failures);
    }

    public static DiscreteTokenDatasetTrainerCheckpointInventory scan(Path rootDir) throws IOException {
        Objects.requireNonNull(rootDir, "rootDir must not be null");
        if (!Files.exists(rootDir)) {
            throw new NoSuchFileException(rootDir.toString());
        }

        List<DiscreteTokenDatasetTrainerCheckpointSnapshot> checkpoints = new ArrayList<>();
        List<ReadFailure> failures = new ArrayList<>();
        for (Path checkpointDir : candidateCheckpointDirs(rootDir)) {
            try {
                checkpoints.add(DiscreteTokenDatasetTrainerCheckpointSnapshot.read(checkpointDir));
            } catch (IOException | RuntimeException error) {
                failures.add(ReadFailure.from(checkpointDir, error));
            }
        }
        return new DiscreteTokenDatasetTrainerCheckpointInventory(rootDir, checkpoints, failures);
    }

    public int checkpointCount() {
        return checkpoints.size();
    }

    public int failureCount() {
        return failures.size();
    }

    public int readyCount() {
        return (int) checkpoints.stream().filter(DiscreteTokenDatasetTrainerCheckpointSnapshot::ready).count();
    }

    public int blockedCount() {
        return (int) checkpoints.stream()
                .filter(checkpoint -> !checkpoint.ready())
                .count();
    }

    public int manifestOnlyCount() {
        return (int) checkpoints.stream()
                .filter(checkpoint -> "manifest-only".equals(checkpoint.status()))
                .count();
    }

    public int resumeReportCount() {
        return (int) checkpoints.stream()
                .filter(DiscreteTokenDatasetTrainerCheckpointSnapshot::resumeReportPresent)
                .count();
    }

    public List<DiscreteTokenDatasetTrainerCheckpointSnapshot> readyCheckpoints() {
        return checkpoints.stream()
                .filter(DiscreteTokenDatasetTrainerCheckpointSnapshot::ready)
                .toList();
    }

    public List<DiscreteTokenDatasetTrainerCheckpointSnapshot> blockedCheckpoints() {
        return checkpoints.stream()
                .filter(checkpoint -> !checkpoint.ready())
                .toList();
    }

    public List<DiscreteTokenDatasetTrainerCheckpointSnapshot> manifestOnlyCheckpoints() {
        return checkpoints.stream()
                .filter(checkpoint -> "manifest-only".equals(checkpoint.status()))
                .toList();
    }

    public Optional<DiscreteTokenDatasetTrainerCheckpointSnapshot> latestCheckpoint() {
        return checkpoints.stream().max(checkpointComparator());
    }

    public Optional<DiscreteTokenDatasetTrainerCheckpointSnapshot> latestReadyCheckpoint() {
        return readyCheckpoints().stream().max(checkpointComparator());
    }

    public DiscreteTokenDatasetTrainerCheckpointSnapshot requireLatestReadyCheckpoint() {
        return latestReadyCheckpoint().orElseThrow(() -> new IllegalStateException(
                "no ready recursive-reasoning checkpoint found under " + rootDir));
    }

    public Optional<DiscreteTokenDatasetTrainerCheckpointSnapshot> selectCheckpoint(
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) {
        Objects.requireNonNull(policy, "policy must not be null");
        return policy.select(this);
    }

    public DiscreteTokenDatasetTrainerCheckpointInspectionReport inspectCheckpoints(
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) {
        Objects.requireNonNull(policy, "policy must not be null");
        return DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(this, policy);
    }

    public Optional<DiscreteTokenDatasetTrainerCheckpointRestorePlan> selectRestorePlan(
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) {
        return inspectCheckpoints(policy).selectedRestorePlan();
    }

    public DiscreteTokenDatasetTrainerCheckpointRestorePlan requireRestorePlan(
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) {
        return inspectCheckpoints(policy).requireRestorePlan();
    }

    public DiscreteTokenDatasetTrainerCheckpointSnapshot requireCheckpoint(
            DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy) {
        Objects.requireNonNull(policy, "policy must not be null");
        return policy.require(this);
    }

    public void requireNoFailures() {
        if (!failures.isEmpty()) {
            throw new IllegalStateException("checkpoint inventory has "
                    + failures.size()
                    + " read failure(s): "
                    + failures.get(0).summary());
        }
    }

    public String summary() {
        return "checkpoint inventory "
                + checkpointCount()
                + " checkpoint(s), "
                + readyCount()
                + " ready, "
                + blockedCount()
                + " blocked, "
                + manifestOnlyCount()
                + " manifest-only, "
                + failureCount()
                + " failure(s)";
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("rootDir", rootDir.toString());
        metadata.put("checkpointCount", checkpointCount());
        metadata.put("failureCount", failureCount());
        metadata.put("readyCount", readyCount());
        metadata.put("blockedCount", blockedCount());
        metadata.put("manifestOnlyCount", manifestOnlyCount());
        metadata.put("resumeReportCount", resumeReportCount());
        metadata.put("summary", summary());
        metadata.put("checkpoints", checkpoints.stream()
                .map(DiscreteTokenDatasetTrainerCheckpointSnapshot::toMetadata)
                .toList());
        metadata.put("failures", failures.stream()
                .map(ReadFailure::toMetadata)
                .toList());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static List<Path> candidateCheckpointDirs(Path rootDir) throws IOException {
        List<Path> candidates = new ArrayList<>();
        if (Files.isRegularFile(DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(rootDir))) {
            candidates.add(rootDir);
        }
        if (!Files.isDirectory(rootDir)) {
            return List.copyOf(candidates);
        }
        try (Stream<Path> children = Files.list(rootDir)) {
            children
                    .filter(Files::isDirectory)
                    .filter(child -> Files.isRegularFile(
                            DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(child)))
                    .sorted(Comparator.comparing(Path::toString))
                    .forEach(candidates::add);
        }
        return List.copyOf(candidates);
    }

    private static Comparator<DiscreteTokenDatasetTrainerCheckpointSnapshot> checkpointComparator() {
        return Comparator
                .comparingLong((DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint) ->
                        checkpoint.manifest().createdAtEpochMillis())
                .thenComparing(checkpoint -> checkpoint.checkpointDir().toString());
    }

    private static List<DiscreteTokenDatasetTrainerCheckpointSnapshot> sortedSnapshots(
            List<DiscreteTokenDatasetTrainerCheckpointSnapshot> snapshots) {
        if (snapshots == null || snapshots.isEmpty()) {
            return List.of();
        }
        return snapshots.stream()
                .map(snapshot -> Objects.requireNonNull(snapshot, "checkpoint snapshot must not be null"))
                .sorted(checkpointComparator())
                .toList();
    }

    private static List<ReadFailure> sortedFailures(List<ReadFailure> failures) {
        if (failures == null || failures.isEmpty()) {
            return List.of();
        }
        return failures.stream()
                .map(failure -> Objects.requireNonNull(failure, "read failure must not be null"))
                .sorted(Comparator.comparing(failure -> failure.checkpointDir().toString()))
                .toList();
    }

    public record ReadFailure(Path checkpointDir, String errorType, String message) {
        public ReadFailure {
            checkpointDir = Objects.requireNonNull(checkpointDir, "checkpointDir must not be null");
            errorType = requireText(errorType, "errorType");
            message = requireText(message, "message");
        }

        public static ReadFailure from(Path checkpointDir, Exception error) {
            Objects.requireNonNull(error, "error must not be null");
            return new ReadFailure(
                    checkpointDir,
                    error.getClass().getSimpleName(),
                    error.getMessage() == null ? error.getClass().getName() : error.getMessage());
        }

        public String summary() {
            return checkpointDir + " -> " + errorType + ": " + message;
        }

        public Map<String, Object> toMetadata() {
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("checkpointDir", checkpointDir.toString());
            metadata.put("errorType", errorType);
            metadata.put("message", message);
            metadata.put("summary", summary());
            return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        }
    }

    private static String requireText(String value, String name) {
        value = Objects.requireNonNull(value, name + " must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }
}

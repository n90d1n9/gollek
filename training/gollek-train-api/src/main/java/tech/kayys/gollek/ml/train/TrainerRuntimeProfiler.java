package tech.kayys.gollek.ml.train;

import java.util.EnumMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Lightweight phase profiler for trainer hot paths.
 */
final class TrainerRuntimeProfiler {
    private final EnumMap<Phase, PhaseCounters> phases = new EnumMap<>(Phase.class);
    private final EnumMap<Scope, PhaseCounters> scopes = new EnumMap<>(Scope.class);
    private long sequence;

    <T> T time(Phase phase, Supplier<T> action) {
        Objects.requireNonNull(action, "action must not be null");
        long startedAt = System.nanoTime();
        try {
            return action.get();
        } finally {
            record(phase, System.nanoTime() - startedAt);
        }
    }

    void time(Phase phase, Runnable action) {
        Objects.requireNonNull(action, "action must not be null");
        long startedAt = System.nanoTime();
        try {
            action.run();
        } finally {
            record(phase, System.nanoTime() - startedAt);
        }
    }

    <T> T time(Scope scope, Supplier<T> action) {
        Objects.requireNonNull(action, "action must not be null");
        long startedAt = System.nanoTime();
        try {
            return action.get();
        } finally {
            record(scope, System.nanoTime() - startedAt);
        }
    }

    void time(Scope scope, Runnable action) {
        Objects.requireNonNull(action, "action must not be null");
        long startedAt = System.nanoTime();
        try {
            action.run();
        } finally {
            record(scope, System.nanoTime() - startedAt);
        }
    }

    synchronized void record(Phase phase, long elapsedNanos) {
        Objects.requireNonNull(phase, "phase must not be null");
        phases.computeIfAbsent(phase, ignored -> new PhaseCounters())
                .add(Math.max(0L, elapsedNanos), ++sequence);
    }

    synchronized void record(Scope scope, long elapsedNanos) {
        Objects.requireNonNull(scope, "scope must not be null");
        scopes.computeIfAbsent(scope, ignored -> new PhaseCounters())
                .add(Math.max(0L, elapsedNanos), ++sequence);
    }

    synchronized PhaseSnapshot snapshot(Phase phase) {
        Objects.requireNonNull(phase, "phase must not be null");
        PhaseCounters counters = phases.get(phase);
        return counters == null ? PhaseSnapshot.empty() : counters.snapshot();
    }

    synchronized PhaseSnapshot snapshot(Scope scope) {
        Objects.requireNonNull(scope, "scope must not be null");
        PhaseCounters counters = scopes.get(scope);
        return counters == null ? PhaseSnapshot.empty() : counters.snapshot();
    }

    synchronized Map<Phase, PhaseSnapshot> snapshot() {
        Map<Phase, PhaseSnapshot> snapshots = new EnumMap<>(Phase.class);
        for (Phase phase : Phase.values()) {
            PhaseSnapshot snapshot = snapshot(phase);
            if (snapshot.count() > 0L) {
                snapshots.put(phase, snapshot);
            }
        }
        return Map.copyOf(snapshots);
    }

    synchronized Map<Scope, PhaseSnapshot> scopeSnapshot() {
        Map<Scope, PhaseSnapshot> snapshots = new EnumMap<>(Scope.class);
        for (Scope scope : Scope.values()) {
            PhaseSnapshot snapshot = snapshot(scope);
            if (snapshot.count() > 0L) {
                snapshots.put(scope, snapshot);
            }
        }
        return Map.copyOf(snapshots);
    }

    synchronized void reset() {
        phases.clear();
        scopes.clear();
        sequence = 0L;
    }

    synchronized Map<String, Object> toMetadata(String prefix) {
        String normalizedPrefix = prefix == null || prefix.isBlank() ? "trainerProfile" : prefix.trim();
        Map<String, Object> metadata = new LinkedHashMap<>();
        Map<Phase, PhaseSnapshot> snapshots = snapshot();
        Map<Scope, PhaseSnapshot> scopeSnapshots = scopeSnapshot();
        long totalProfiledNanos = totalNanos(snapshots);
        metadata.putAll(TrainerRuntimeProfileBalanceMetadata.from(normalizedPrefix, snapshots));
        metadata.putAll(TrainerRuntimeWallClockMetadata.from(normalizedPrefix, scopeSnapshots, snapshots));
        for (Map.Entry<Phase, PhaseSnapshot> entry : snapshots.entrySet()) {
            String key = normalizedPrefix + "." + entry.getKey().metadataKey();
            PhaseSnapshot snapshot = entry.getValue();
            metadata.put(key + ".count", snapshot.count());
            metadata.put(key + ".totalMillis", snapshot.totalMillis());
            metadata.put(key + ".percentTotal", percentTotal(snapshot, totalProfiledNanos));
            metadata.put(key + ".averageMillis", snapshot.averageMillis());
            metadata.put(key + ".minMillis", snapshot.minMillis());
            metadata.put(key + ".maxMillis", snapshot.maxMillis());
            metadata.put(key + ".lastMillis", snapshot.lastMillis());
            metadata.put(key + ".stddevMillis", snapshot.stddevMillis());
        }
        putGroups(metadata, normalizedPrefix, snapshots, totalProfiledNanos);
        putHotspots(metadata, normalizedPrefix, snapshots, totalProfiledNanos, 5);
        return Map.copyOf(metadata);
    }

    private static void putGroups(
            Map<String, Object> metadata,
            String prefix,
            Map<Phase, PhaseSnapshot> snapshots,
            long totalProfiledNanos) {
        Map<String, PhaseCounters> grouped = new LinkedHashMap<>();
        for (Map.Entry<Phase, PhaseSnapshot> entry : snapshots.entrySet()) {
            PhaseSnapshot snapshot = entry.getValue();
            grouped.computeIfAbsent(entry.getKey().group(), ignored -> new PhaseCounters())
                    .add(snapshot);
        }
        List<Map.Entry<String, PhaseSnapshot>> ranked = grouped.entrySet().stream()
                .map(entry -> Map.entry(entry.getKey(), entry.getValue().snapshot()))
                .sorted(Comparator
                        .<Map.Entry<String, PhaseSnapshot>>comparingLong(entry -> entry.getValue().totalNanos())
                        .reversed()
                        .thenComparing(Map.Entry::getKey))
                .toList();
        List<Map<String, Object>> groups = new ArrayList<>();
        for (Map.Entry<String, PhaseSnapshot> entry : ranked) {
            groups.add(group(entry.getKey(), entry.getValue(), totalProfiledNanos));
        }
        metadata.put(prefix + ".groups", List.copyOf(groups));
        metadata.put(prefix + ".groupCount", groups.size());
        if (!groups.isEmpty()) {
            Map<String, Object> primary = groups.get(0);
            metadata.put(prefix + ".primaryGroup.name", primary.get("name"));
            metadata.put(prefix + ".primaryGroup.totalMillis", primary.get("totalMillis"));
            metadata.put(prefix + ".primaryGroup.percentTotal", primary.get("percentTotal"));
            metadata.put(prefix + ".primaryGroup.averageMillis", primary.get("averageMillis"));
            metadata.put(prefix + ".primaryGroup.minMillis", primary.get("minMillis"));
            metadata.put(prefix + ".primaryGroup.lastMillis", primary.get("lastMillis"));
            metadata.put(prefix + ".primaryGroup.stddevMillis", primary.get("stddevMillis"));
        }
    }

    private static void putHotspots(
            Map<String, Object> metadata,
            String prefix,
            Map<Phase, PhaseSnapshot> snapshots,
            long totalProfiledNanos,
            int limit) {
        List<Map.Entry<Phase, PhaseSnapshot>> ranked = new ArrayList<>(snapshots.entrySet());
        ranked.sort(Comparator
                .<Map.Entry<Phase, PhaseSnapshot>>comparingLong(entry -> entry.getValue().totalNanos())
                .reversed()
                .thenComparing(entry -> entry.getKey().metadataKey()));
        List<Map<String, Object>> hotspots = new ArrayList<>();
        for (Map.Entry<Phase, PhaseSnapshot> entry : ranked) {
            if (hotspots.size() >= limit) {
                break;
            }
            hotspots.add(hotspot(entry.getKey(), entry.getValue(), totalProfiledNanos));
        }
        metadata.put(prefix + ".hotspots", List.copyOf(hotspots));
        metadata.put(prefix + ".hotspotCount", hotspots.size());
        if (!hotspots.isEmpty()) {
            Map<String, Object> primary = hotspots.get(0);
            metadata.put(prefix + ".primaryHotspot.phase", primary.get("phase"));
            metadata.put(prefix + ".primaryHotspot.totalMillis", primary.get("totalMillis"));
            metadata.put(prefix + ".primaryHotspot.percentTotal", primary.get("percentTotal"));
            metadata.put(prefix + ".primaryHotspot.averageMillis", primary.get("averageMillis"));
            metadata.put(prefix + ".primaryHotspot.minMillis", primary.get("minMillis"));
            metadata.put(prefix + ".primaryHotspot.lastMillis", primary.get("lastMillis"));
            metadata.put(prefix + ".primaryHotspot.stddevMillis", primary.get("stddevMillis"));
        }
    }

    private static Map<String, Object> hotspot(Phase phase, PhaseSnapshot snapshot, long totalProfiledNanos) {
        Map<String, Object> hotspot = new LinkedHashMap<>();
        hotspot.put("phase", phase.metadataKey());
        hotspot.put("count", snapshot.count());
        hotspot.put("totalMillis", snapshot.totalMillis());
        hotspot.put("percentTotal", percentTotal(snapshot, totalProfiledNanos));
        hotspot.put("averageMillis", snapshot.averageMillis());
        hotspot.put("minMillis", snapshot.minMillis());
        hotspot.put("maxMillis", snapshot.maxMillis());
        hotspot.put("lastMillis", snapshot.lastMillis());
        hotspot.put("stddevMillis", snapshot.stddevMillis());
        return Map.copyOf(hotspot);
    }

    private static Map<String, Object> group(String name, PhaseSnapshot snapshot, long totalProfiledNanos) {
        Map<String, Object> group = new LinkedHashMap<>();
        group.put("name", name);
        group.put("count", snapshot.count());
        group.put("totalMillis", snapshot.totalMillis());
        group.put("percentTotal", percentTotal(snapshot, totalProfiledNanos));
        group.put("averageMillis", snapshot.averageMillis());
        group.put("minMillis", snapshot.minMillis());
        group.put("maxMillis", snapshot.maxMillis());
        group.put("lastMillis", snapshot.lastMillis());
        group.put("stddevMillis", snapshot.stddevMillis());
        return Map.copyOf(group);
    }

    private static long totalNanos(Map<Phase, PhaseSnapshot> snapshots) {
        long total = 0L;
        for (PhaseSnapshot snapshot : snapshots.values()) {
            total = Math.addExact(total, snapshot.totalNanos());
        }
        return total;
    }

    private static double percentTotal(PhaseSnapshot snapshot, long totalProfiledNanos) {
        return totalProfiledNanos <= 0L ? 0.0 : (snapshot.totalNanos() * 100.0) / totalProfiledNanos;
    }

    enum Scope {
        TRAIN_BATCH("trainBatch", List.of(
                Phase.TRAIN_BATCH_ADAPT,
                Phase.TRAIN_VALIDATE_BATCH,
                Phase.TRAIN_ZERO_GRAD,
                Phase.TRAIN_FORWARD,
                Phase.TRAIN_VALIDATE_PREDICTION,
                Phase.TRAIN_LOSS,
                Phase.TRAIN_VALIDATE_LOSS,
                Phase.TRAIN_METRICS,
                Phase.TRAIN_BACKWARD,
                Phase.TRAIN_THROUGHPUT)),
        VALIDATION_BATCH("validationBatch", List.of(
                Phase.VALIDATION_BATCH_ADAPT,
                Phase.VALIDATION_VALIDATE_BATCH,
                Phase.VALIDATION_FORWARD,
                Phase.VALIDATION_VALIDATE_PREDICTION,
                Phase.VALIDATION_LOSS,
                Phase.VALIDATION_VALIDATE_LOSS,
                Phase.VALIDATION_METRICS,
                Phase.VALIDATION_THROUGHPUT)),
        OPTIMIZER_STEP("optimizerStep", List.of(
                Phase.OPTIMIZER_GRADIENT_ACCUMULATION_SCALE,
                Phase.OPTIMIZER_AMP_UNSCALE_CHECK,
                Phase.OPTIMIZER_AMP_OVERFLOW_UPDATE,
                Phase.OPTIMIZER_GRADIENT_DIAGNOSTICS_BEFORE_CLIP,
                Phase.OPTIMIZER_GRADIENT_VALIDATE_BEFORE_CLIP,
                Phase.OPTIMIZER_GRADIENT_NORM_CLIP,
                Phase.OPTIMIZER_GRADIENT_VALUE_CLIP,
                Phase.OPTIMIZER_GRADIENT_DIAGNOSTICS_AFTER_CLIP,
                Phase.OPTIMIZER_GRADIENT_VALIDATE_AFTER_CLIP,
                Phase.OPTIMIZER_PARAMETER_SNAPSHOT,
                Phase.OPTIMIZER_STEP,
                Phase.OPTIMIZER_PARAMETER_UPDATE_DIAGNOSTICS,
                Phase.OPTIMIZER_PARAMETER_DIAGNOSTICS,
                Phase.OPTIMIZER_PARAMETER_VALIDATE,
                Phase.OPTIMIZER_SCHEDULER_STEP,
                Phase.OPTIMIZER_AMP_UPDATE,
                Phase.OPTIMIZER_ZERO_GRAD));

        private final String metadataKey;
        private final List<Phase> profiledPhases;

        Scope(String metadataKey, List<Phase> profiledPhases) {
            this.metadataKey = metadataKey;
            this.profiledPhases = List.copyOf(profiledPhases);
        }

        String metadataKey() {
            return metadataKey;
        }

        List<Phase> profiledPhases() {
            return profiledPhases;
        }
    }

    enum Phase {
        INPUT_TRAIN_ITERATOR("input.train.iterator"),
        INPUT_TRAIN_HAS_NEXT("input.train.hasNext"),
        INPUT_TRAIN_NEXT("input.train.next"),
        INPUT_VALIDATION_ITERATOR("input.validation.iterator"),
        INPUT_VALIDATION_HAS_NEXT("input.validation.hasNext"),
        INPUT_VALIDATION_NEXT("input.validation.next"),
        TRAIN_BATCH_ADAPT("train.batchAdapt"),
        TRAIN_VALIDATE_BATCH("train.validateBatch"),
        TRAIN_ZERO_GRAD("train.zeroGrad"),
        TRAIN_FORWARD("train.forward"),
        TRAIN_VALIDATE_PREDICTION("train.validatePrediction"),
        TRAIN_LOSS("train.loss"),
        TRAIN_VALIDATE_LOSS("train.validateLoss"),
        TRAIN_METRICS("train.metrics"),
        TRAIN_BACKWARD("train.backward"),
        TRAIN_THROUGHPUT("train.throughput"),
        VALIDATION_BATCH_ADAPT("validation.batchAdapt"),
        VALIDATION_VALIDATE_BATCH("validation.validateBatch"),
        VALIDATION_FORWARD("validation.forward"),
        VALIDATION_VALIDATE_PREDICTION("validation.validatePrediction"),
        VALIDATION_LOSS("validation.loss"),
        VALIDATION_VALIDATE_LOSS("validation.validateLoss"),
        VALIDATION_METRICS("validation.metrics"),
        VALIDATION_THROUGHPUT("validation.throughput"),
        OPTIMIZER_GRADIENT_ACCUMULATION_SCALE("optimizer.gradientAccumulationScale"),
        OPTIMIZER_AMP_UNSCALE_CHECK("optimizer.ampUnscaleCheck"),
        OPTIMIZER_AMP_OVERFLOW_UPDATE("optimizer.ampOverflowUpdate"),
        OPTIMIZER_GRADIENT_DIAGNOSTICS_BEFORE_CLIP("optimizer.gradientDiagnosticsBeforeClip"),
        OPTIMIZER_GRADIENT_VALIDATE_BEFORE_CLIP("optimizer.gradientValidateBeforeClip"),
        OPTIMIZER_GRADIENT_NORM_CLIP("optimizer.gradientNormClip"),
        OPTIMIZER_GRADIENT_VALUE_CLIP("optimizer.gradientValueClip"),
        OPTIMIZER_GRADIENT_DIAGNOSTICS_AFTER_CLIP("optimizer.gradientDiagnosticsAfterClip"),
        OPTIMIZER_GRADIENT_VALIDATE_AFTER_CLIP("optimizer.gradientValidateAfterClip"),
        OPTIMIZER_PARAMETER_SNAPSHOT("optimizer.parameterSnapshot"),
        OPTIMIZER_STEP("optimizer.step"),
        OPTIMIZER_PARAMETER_UPDATE_DIAGNOSTICS("optimizer.parameterUpdateDiagnostics"),
        OPTIMIZER_PARAMETER_DIAGNOSTICS("optimizer.parameterDiagnostics"),
        OPTIMIZER_PARAMETER_VALIDATE("optimizer.parameterValidate"),
        OPTIMIZER_SCHEDULER_STEP("optimizer.schedulerStep"),
        OPTIMIZER_AMP_UPDATE("optimizer.ampUpdate"),
        OPTIMIZER_ZERO_GRAD("optimizer.zeroGrad");

        private final String metadataKey;

        Phase(String metadataKey) {
            this.metadataKey = metadataKey;
        }

        String metadataKey() {
            return metadataKey;
        }

        String group() {
            int dot = metadataKey.indexOf('.');
            return dot < 1 ? metadataKey : metadataKey.substring(0, dot);
        }
    }

    record PhaseSnapshot(
            long count,
            long totalNanos,
            long minNanos,
            long maxNanos,
            long lastNanos,
            long lastSequence,
            double sumSquaresNanos) {
        static PhaseSnapshot empty() {
            return new PhaseSnapshot(0L, 0L, 0L, 0L, 0L, 0L, 0.0);
        }

        double totalMillis() {
            return totalNanos / 1_000_000.0;
        }

        double averageMillis() {
            return count == 0L ? 0.0 : totalMillis() / count;
        }

        double minMillis() {
            return minNanos / 1_000_000.0;
        }

        double maxMillis() {
            return maxNanos / 1_000_000.0;
        }

        double lastMillis() {
            return lastNanos / 1_000_000.0;
        }

        double stddevMillis() {
            if (count <= 1L) {
                return 0.0;
            }
            double averageNanos = (double) totalNanos / count;
            double varianceNanos = Math.max(0.0, (sumSquaresNanos / count) - (averageNanos * averageNanos));
            return Math.sqrt(varianceNanos) / 1_000_000.0;
        }
    }

    private static final class PhaseCounters {
        private long count;
        private long totalNanos;
        private long minNanos = Long.MAX_VALUE;
        private long maxNanos;
        private long lastNanos;
        private long lastSequence;
        private double sumSquaresNanos;

        private void add(long elapsedNanos, long sequence) {
            count++;
            totalNanos = Math.addExact(totalNanos, elapsedNanos);
            minNanos = Math.min(minNanos, elapsedNanos);
            maxNanos = Math.max(maxNanos, elapsedNanos);
            lastNanos = elapsedNanos;
            lastSequence = sequence;
            sumSquaresNanos += (double) elapsedNanos * elapsedNanos;
        }

        private void add(PhaseSnapshot snapshot) {
            count = Math.addExact(count, snapshot.count());
            totalNanos = Math.addExact(totalNanos, snapshot.totalNanos());
            if (snapshot.count() > 0L) {
                minNanos = Math.min(minNanos, snapshot.minNanos());
                if (snapshot.lastSequence() >= lastSequence) {
                    lastNanos = snapshot.lastNanos();
                    lastSequence = snapshot.lastSequence();
                }
            }
            maxNanos = Math.max(maxNanos, snapshot.maxNanos());
            sumSquaresNanos += snapshot.sumSquaresNanos();
        }

        private PhaseSnapshot snapshot() {
            long safeMinNanos = count == 0L ? 0L : minNanos;
            return new PhaseSnapshot(count, totalNanos, safeMinNanos, maxNanos, lastNanos, lastSequence, sumSquaresNanos);
        }
    }
}

package tech.kayys.gollek.ml.train;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Builds wall-clock runtime metadata that separates measured trainer scope time
 * from already-profiled sub-phase time.
 */
final class TrainerRuntimeWallClockMetadata {
    private TrainerRuntimeWallClockMetadata() {
    }

    static Map<String, Object> from(
            String prefix,
            Map<TrainerRuntimeProfiler.Scope, TrainerRuntimeProfiler.PhaseSnapshot> scopes,
            Map<TrainerRuntimeProfiler.Phase, TrainerRuntimeProfiler.PhaseSnapshot> phases) {
        Objects.requireNonNull(scopes, "scopes must not be null");
        Objects.requireNonNull(phases, "phases must not be null");
        String normalizedPrefix = prefix == null || prefix.isBlank() ? "runtimeProfile" : prefix.trim();
        String wallPrefix = normalizedPrefix + ".wall";
        Map<String, Object> metadata = new LinkedHashMap<>();
        long totalScopeNanos = totalScopeNanos(scopes);
        metadata.put(wallPrefix + ".totalMillis", millis(totalScopeNanos));
        metadata.put(wallPrefix + ".scopeCount", scopes.size());

        ScopeOverhead primaryOverhead = scopes.entrySet().stream()
                .map(entry -> scopeOverhead(entry.getKey(), entry.getValue(), phases))
                .max(Comparator
                        .comparingLong(ScopeOverhead::overheadNanos)
                        .thenComparing(overhead -> overhead.scope().metadataKey()))
                .orElse(ScopeOverhead.empty());

        for (TrainerRuntimeProfiler.Scope scope : TrainerRuntimeProfiler.Scope.values()) {
            TrainerRuntimeProfiler.PhaseSnapshot snapshot =
                    scopes.getOrDefault(scope, TrainerRuntimeProfiler.PhaseSnapshot.empty());
            ScopeOverhead overhead = scopeOverhead(scope, snapshot, phases);
            String key = wallPrefix + "." + scope.metadataKey();
            metadata.put(key + ".count", snapshot.count());
            metadata.put(key + ".totalMillis", snapshot.totalMillis());
            metadata.put(key + ".averageMillis", snapshot.averageMillis());
            metadata.put(key + ".minMillis", snapshot.minMillis());
            metadata.put(key + ".maxMillis", snapshot.maxMillis());
            metadata.put(key + ".lastMillis", snapshot.lastMillis());
            metadata.put(key + ".stddevMillis", snapshot.stddevMillis());
            metadata.put(key + ".profiledMillis", millis(overhead.profiledNanos()));
            metadata.put(key + ".overheadMillis", millis(overhead.overheadNanos()));
            metadata.put(key + ".overheadPercent", percent(overhead.overheadNanos(), snapshot.totalNanos()));
        }

        metadata.put(wallPrefix + ".primaryOverhead.scope", primaryOverhead.scope().metadataKey());
        metadata.put(wallPrefix + ".primaryOverhead.totalMillis", millis(primaryOverhead.scopeNanos()));
        metadata.put(wallPrefix + ".primaryOverhead.profiledMillis", millis(primaryOverhead.profiledNanos()));
        metadata.put(wallPrefix + ".primaryOverhead.overheadMillis", millis(primaryOverhead.overheadNanos()));
        metadata.put(wallPrefix + ".primaryOverhead.overheadPercent",
                percent(primaryOverhead.overheadNanos(), primaryOverhead.scopeNanos()));
        return Map.copyOf(metadata);
    }

    private static ScopeOverhead scopeOverhead(
            TrainerRuntimeProfiler.Scope scope,
            TrainerRuntimeProfiler.PhaseSnapshot scopeSnapshot,
            Map<TrainerRuntimeProfiler.Phase, TrainerRuntimeProfiler.PhaseSnapshot> phases) {
        long profiledNanos = 0L;
        for (TrainerRuntimeProfiler.Phase phase : scope.profiledPhases()) {
            profiledNanos = Math.addExact(
                    profiledNanos,
                    phases.getOrDefault(phase, TrainerRuntimeProfiler.PhaseSnapshot.empty()).totalNanos());
        }
        long overheadNanos = Math.max(0L, scopeSnapshot.totalNanos() - profiledNanos);
        return new ScopeOverhead(scope, scopeSnapshot.totalNanos(), profiledNanos, overheadNanos);
    }

    private static long totalScopeNanos(
            Map<TrainerRuntimeProfiler.Scope, TrainerRuntimeProfiler.PhaseSnapshot> scopes) {
        long total = 0L;
        for (TrainerRuntimeProfiler.PhaseSnapshot snapshot : scopes.values()) {
            total = Math.addExact(total, snapshot.totalNanos());
        }
        return total;
    }

    private static double millis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private static double percent(long nanos, long totalNanos) {
        return totalNanos <= 0L ? 0.0 : (nanos * 100.0) / totalNanos;
    }

    private record ScopeOverhead(
            TrainerRuntimeProfiler.Scope scope,
            long scopeNanos,
            long profiledNanos,
            long overheadNanos) {
        private static ScopeOverhead empty() {
            return new ScopeOverhead(TrainerRuntimeProfiler.Scope.TRAIN_BATCH, 0L, 0L, 0L);
        }
    }
}

package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * Compact, type-safe efficiency summary derived from trainer runtime-profile metadata.
 */
public record TrainingReportRuntimeEfficiency(
        Status status,
        boolean available,
        OptionalDouble measuredMillis,
        OptionalDouble wallMillis,
        OptionalDouble accountedPercent,
        String overheadScope,
        OptionalDouble overheadMillis,
        OptionalDouble overheadPercent,
        String bottleneck,
        OptionalDouble bottleneckPercent,
        String primaryHotspot) {
    public TrainingReportRuntimeEfficiency {
        status = status == null ? Status.NO_PROFILE : status;
        measuredMillis = measuredMillis == null ? OptionalDouble.empty() : measuredMillis;
        wallMillis = wallMillis == null ? OptionalDouble.empty() : wallMillis;
        accountedPercent = accountedPercent == null ? OptionalDouble.empty() : accountedPercent;
        overheadScope = overheadScope == null || overheadScope.isBlank() ? "none" : overheadScope.trim();
        overheadMillis = overheadMillis == null ? OptionalDouble.empty() : overheadMillis;
        overheadPercent = overheadPercent == null ? OptionalDouble.empty() : overheadPercent;
        bottleneck = bottleneck == null || bottleneck.isBlank() ? "none" : bottleneck.trim();
        bottleneckPercent = bottleneckPercent == null ? OptionalDouble.empty() : bottleneckPercent;
        primaryHotspot = primaryHotspot == null || primaryHotspot.isBlank() ? "none" : primaryHotspot.trim();
        available = available
                || measuredMillis.isPresent()
                || wallMillis.isPresent()
                || accountedPercent.isPresent()
                || overheadMillis.isPresent()
                || overheadPercent.isPresent()
                || bottleneckPercent.isPresent()
                || !"none".equals(overheadScope)
                || !"none".equals(bottleneck)
                || !"none".equals(primaryHotspot);
    }

    public static TrainingReportRuntimeEfficiency empty() {
        return new TrainingReportRuntimeEfficiency(
                Status.NO_PROFILE,
                false,
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                "none",
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                "none",
                OptionalDouble.empty(),
                "none");
    }

    public static TrainingReportRuntimeEfficiency from(TrainingReportRuntimeProfile profile) {
        if (profile == null || !profile.available()) {
            return empty();
        }
        OptionalDouble measuredMillis = measuredMillis(profile);
        OptionalDouble wallMillis = profile.wallClock().totalMillis();
        OptionalDouble accountedPercent = accountedPercent(measuredMillis, wallMillis);
        TrainingReportRuntimeWallClockAssessment wallClockAssessment =
                TrainingReportRuntimeWallClockAssessment.assess(profile.wallClock());
        String bottleneck = bottleneck(profile);
        OptionalDouble bottleneckPercent = bottleneckPercent(profile);
        String primaryHotspot = profile.primaryHotspot()
                .map(TrainingReportRuntimeProfile.Hotspot::phase)
                .filter(phase -> !phase.isBlank())
                .orElse("none");
        return new TrainingReportRuntimeEfficiency(
                status(wallClockAssessment.overhead().overheadPercent(), bottleneckPercent, accountedPercent),
                true,
                measuredMillis,
                wallMillis,
                accountedPercent,
                wallClockAssessment.scopeKey(),
                wallClockAssessment.overhead().overheadMillis(),
                wallClockAssessment.overhead().overheadPercent(),
                bottleneck,
                bottleneckPercent,
                primaryHotspot);
    }

    public boolean needsOptimization() {
        return status == Status.NEEDS_OPTIMIZATION;
    }

    public boolean watch() {
        return status == Status.WATCH;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("available", available);
        map.put("status", status.name());
        put(map, "measuredMillis", measuredMillis);
        put(map, "wallMillis", wallMillis);
        put(map, "accountedPercent", accountedPercent);
        map.put("overheadScope", overheadScope);
        put(map, "overheadMillis", overheadMillis);
        put(map, "overheadPercent", overheadPercent);
        map.put("bottleneck", bottleneck);
        put(map, "bottleneckPercent", bottleneckPercent);
        map.put("primaryHotspot", primaryHotspot);
        return Map.copyOf(map);
    }

    private static Status status(
            OptionalDouble overheadPercent,
            OptionalDouble bottleneckPercent,
            OptionalDouble accountedPercent) {
        double overhead = overheadPercent.orElse(0.0);
        double bottleneck = bottleneckPercent.orElse(0.0);
        double accounted = accountedPercent.orElse(100.0);
        if (overhead >= 40.0 || bottleneck >= 80.0) {
            return Status.NEEDS_OPTIMIZATION;
        }
        if (overhead >= 25.0 || bottleneck >= 60.0 || accounted < 75.0) {
            return Status.WATCH;
        }
        return Status.EFFICIENT;
    }

    private static OptionalDouble measuredMillis(TrainingReportRuntimeProfile profile) {
        OptionalDouble balanceTotal = profile.balance().totalMillis();
        if (balanceTotal.isPresent()) {
            return balanceTotal;
        }
        double sum = profile.groups().stream()
                .map(TrainingReportRuntimeProfile.Group::totalMillis)
                .filter(OptionalDouble::isPresent)
                .mapToDouble(OptionalDouble::orElseThrow)
                .sum();
        if (sum > 0.0) {
            return OptionalDouble.of(sum);
        }
        return profile.primaryGroup().flatMap(group -> group.totalMillis().stream().boxed().findFirst())
                .map(OptionalDouble::of)
                .orElseGet(OptionalDouble::empty);
    }

    private static OptionalDouble accountedPercent(OptionalDouble measuredMillis, OptionalDouble wallMillis) {
        if (measuredMillis.isEmpty() || wallMillis.isEmpty() || wallMillis.orElseThrow() <= 0.0) {
            return OptionalDouble.empty();
        }
        return OptionalDouble.of(measuredMillis.orElseThrow() / wallMillis.orElseThrow() * 100.0);
    }

    private static String bottleneck(TrainingReportRuntimeProfile profile) {
        TrainingReportRuntimeProfile.BalanceBucket bucket = profile.balance().dominantBucket();
        if (bucket != TrainingReportRuntimeProfile.BalanceBucket.NONE) {
            return bucket.name().toLowerCase(Locale.ROOT);
        }
        return profile.primaryGroup()
                .map(TrainingReportRuntimeProfile.Group::name)
                .filter(name -> !name.isBlank())
                .orElse("none");
    }

    private static OptionalDouble bottleneckPercent(TrainingReportRuntimeProfile profile) {
        OptionalDouble balancePercent = profile.balance().dominantPercent();
        if (balancePercent.isPresent()) {
            return balancePercent;
        }
        return profile.primaryGroup().flatMap(group -> group.percentTotal().stream().boxed().findFirst())
                .map(OptionalDouble::of)
                .orElseGet(OptionalDouble::empty);
    }

    private static void put(Map<String, Object> map, String key, OptionalDouble value) {
        value.ifPresent(number -> map.put(key, number));
    }

    /**
     * Human-readable runtime-efficiency state for trainer report summaries.
     */
    public enum Status {
        NO_PROFILE,
        EFFICIENT,
        WATCH,
        NEEDS_OPTIMIZATION
    }
}

package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalLong;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;

/**
 * Type-safe view over trainer runtime profile metadata.
 */
public record TrainingReportRuntimeProfile(
        Balance balance,
        WallClock wallClock,
        int groupCount,
        Optional<Group> primaryGroup,
        List<Group> groups,
        int hotspotCount,
        Optional<Hotspot> primaryHotspot,
        List<Hotspot> hotspots) {
    public TrainingReportRuntimeProfile {
        balance = balance == null ? Balance.empty() : balance;
        wallClock = wallClock == null ? WallClock.empty() : wallClock;
        groupCount = Math.max(0, groupCount);
        primaryGroup = primaryGroup == null ? Optional.empty() : primaryGroup;
        groups = groups == null ? List.of() : List.copyOf(groups);
        hotspotCount = Math.max(0, hotspotCount);
        primaryHotspot = primaryHotspot == null ? Optional.empty() : primaryHotspot;
        hotspots = hotspots == null ? List.of() : List.copyOf(hotspots);
    }

    public TrainingReportRuntimeProfile(
            int groupCount,
            Optional<Group> primaryGroup,
            List<Group> groups,
            int hotspotCount,
            Optional<Hotspot> primaryHotspot,
            List<Hotspot> hotspots) {
        this(Balance.empty(), WallClock.empty(), groupCount, primaryGroup, groups, hotspotCount, primaryHotspot, hotspots);
    }

    public static TrainingReportRuntimeProfile empty() {
        return new TrainingReportRuntimeProfile(
                Balance.empty(),
                WallClock.empty(),
                0,
                Optional.empty(),
                List.of(),
                0,
                Optional.empty(),
                List.of());
    }

    public static TrainingReportRuntimeProfile fromMetadata(Map<String, ?> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return empty();
        }
        List<Group> groups = groups(metadata.get("runtimeProfile.groups"));
        Optional<Group> primaryGroup = primaryGroup(metadata, groups);
        List<Hotspot> hotspots = hotspots(metadata.get("runtimeProfile.hotspots"));
        Optional<Hotspot> primary = primaryHotspot(metadata, hotspots);
        return new TrainingReportRuntimeProfile(
                Balance.fromMetadata(metadata),
                WallClock.fromMetadata(metadata),
                intValue(metadata.get("runtimeProfile.groupCount"), groups.size()),
                primaryGroup,
                groups,
                intValue(metadata.get("runtimeProfile.hotspotCount"), hotspots.size()),
                primary,
                hotspots);
    }

    public boolean available() {
        return groupCount > 0
                || !groups.isEmpty()
                || primaryGroup.isPresent()
                || hotspotCount > 0
                || !hotspots.isEmpty()
                || primaryHotspot.isPresent()
                || balance.available()
                || wallClock.available();
    }

    public Map<String, Object> toMap() {
        return Map.of(
                "available", available(),
                "balance", balance.toMap(),
                "wallClock", wallClock.toMap(),
                "groupCount", groupCount,
                "primaryGroup", primaryGroup.map(Group::toMap).orElse(Map.of()),
                "groups", groups.stream().map(Group::toMap).toList(),
                "hotspotCount", hotspotCount,
                "primaryHotspot", primaryHotspot.map(Hotspot::toMap).orElse(Map.of()),
                "hotspots", hotspots.stream().map(Hotspot::toMap).toList());
    }

    private static Optional<Group> primaryGroup(Map<String, ?> metadata, List<Group> groups) {
        String name = stringValue(metadata.get("runtimeProfile.primaryGroup.name"), "");
        if (!name.isBlank()) {
            return Optional.of(new Group(
                    name,
                    OptionalLong.empty(),
                    optionalDouble(metadata.get("runtimeProfile.primaryGroup.totalMillis")),
                    optionalDouble(metadata.get("runtimeProfile.primaryGroup.percentTotal")),
                    optionalDouble(metadata.get("runtimeProfile.primaryGroup.averageMillis")),
                    optionalDouble(metadata.get("runtimeProfile.primaryGroup.minMillis")),
                    OptionalDouble.empty(),
                    optionalDouble(metadata.get("runtimeProfile.primaryGroup.lastMillis")),
                    optionalDouble(metadata.get("runtimeProfile.primaryGroup.stddevMillis"))));
        }
        return groups.isEmpty() ? Optional.empty() : Optional.of(groups.get(0));
    }

    private static Optional<Hotspot> primaryHotspot(Map<String, ?> metadata, List<Hotspot> hotspots) {
        String phase = stringValue(metadata.get("runtimeProfile.primaryHotspot.phase"), "");
        if (!phase.isBlank()) {
            return Optional.of(new Hotspot(
                    phase,
                    OptionalLong.empty(),
                    optionalDouble(metadata.get("runtimeProfile.primaryHotspot.totalMillis")),
                    optionalDouble(metadata.get("runtimeProfile.primaryHotspot.percentTotal")),
                    optionalDouble(metadata.get("runtimeProfile.primaryHotspot.averageMillis")),
                    optionalDouble(metadata.get("runtimeProfile.primaryHotspot.minMillis")),
                    OptionalDouble.empty(),
                    optionalDouble(metadata.get("runtimeProfile.primaryHotspot.lastMillis")),
                    optionalDouble(metadata.get("runtimeProfile.primaryHotspot.stddevMillis"))));
        }
        return hotspots.isEmpty() ? Optional.empty() : Optional.of(hotspots.get(0));
    }

    private static List<Group> groups(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<Group> groups = new ArrayList<>();
        for (Object item : iterable) {
            if (item instanceof Map<?, ?> map) {
                groups.add(Group.fromMap(map));
            }
        }
        return List.copyOf(groups);
    }

    private static List<Hotspot> hotspots(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<Hotspot> hotspots = new ArrayList<>();
        for (Object item : iterable) {
            if (item instanceof Map<?, ?> map) {
                hotspots.add(Hotspot.fromMap(map));
            }
        }
        return List.copyOf(hotspots);
    }

    public record WallClock(
            boolean available,
            OptionalDouble totalMillis,
            int scopeCount,
            String primaryOverheadScope,
            WallScope primaryOverhead,
            WallScope trainBatch,
            WallScope validationBatch,
            WallScope optimizerStep) {
        public WallClock {
            totalMillis = totalMillis == null ? OptionalDouble.empty() : totalMillis;
            scopeCount = Math.max(0, scopeCount);
            primaryOverheadScope = primaryOverheadScope == null || primaryOverheadScope.isBlank()
                    ? "none"
                    : primaryOverheadScope.trim();
            primaryOverhead = primaryOverhead == null ? WallScope.empty() : primaryOverhead;
            trainBatch = trainBatch == null ? WallScope.empty() : trainBatch;
            validationBatch = validationBatch == null ? WallScope.empty() : validationBatch;
            optimizerStep = optimizerStep == null ? WallScope.empty() : optimizerStep;
            available = available
                    || totalMillis.isPresent()
                    || scopeCount > 0
                    || primaryOverhead.available()
                    || trainBatch.available()
                    || validationBatch.available()
                    || optimizerStep.available();
        }

        static WallClock empty() {
            return new WallClock(
                    false,
                    OptionalDouble.empty(),
                    0,
                    "none",
                    WallScope.empty(),
                    WallScope.empty(),
                    WallScope.empty(),
                    WallScope.empty());
        }

        static WallClock fromMetadata(Map<String, ?> metadata) {
            Objects.requireNonNull(metadata, "metadata must not be null");
            return new WallClock(
                    false,
                    optionalDouble(metadata.get("runtimeProfile.wall.totalMillis")),
                    intValue(metadata.get("runtimeProfile.wall.scopeCount"), 0),
                    stringValue(metadata.get("runtimeProfile.wall.primaryOverhead.scope"), "none"),
                    WallScope.fromMetadata(metadata, "runtimeProfile.wall.primaryOverhead"),
                    WallScope.fromMetadata(metadata, "runtimeProfile.wall.trainBatch"),
                    WallScope.fromMetadata(metadata, "runtimeProfile.wall.validationBatch"),
                    WallScope.fromMetadata(metadata, "runtimeProfile.wall.optimizerStep"));
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("available", available);
            totalMillis.ifPresent(value -> map.put("totalMillis", value));
            map.put("scopeCount", scopeCount);
            map.put("primaryOverheadScope", primaryOverheadScope);
            map.put("primaryOverhead", primaryOverhead.toMap());
            map.put("trainBatch", trainBatch.toMap());
            map.put("validationBatch", validationBatch.toMap());
            map.put("optimizerStep", optimizerStep.toMap());
            return Map.copyOf(map);
        }
    }

    public record WallScope(
            OptionalLong count,
            OptionalDouble totalMillis,
            OptionalDouble averageMillis,
            OptionalDouble minMillis,
            OptionalDouble maxMillis,
            OptionalDouble lastMillis,
            OptionalDouble stddevMillis,
            OptionalDouble profiledMillis,
            OptionalDouble overheadMillis,
            OptionalDouble overheadPercent) {
        public WallScope {
            count = count == null ? OptionalLong.empty() : count;
            totalMillis = totalMillis == null ? OptionalDouble.empty() : totalMillis;
            averageMillis = averageMillis == null ? OptionalDouble.empty() : averageMillis;
            minMillis = minMillis == null ? OptionalDouble.empty() : minMillis;
            maxMillis = maxMillis == null ? OptionalDouble.empty() : maxMillis;
            lastMillis = lastMillis == null ? OptionalDouble.empty() : lastMillis;
            stddevMillis = stddevMillis == null ? OptionalDouble.empty() : stddevMillis;
            profiledMillis = profiledMillis == null ? OptionalDouble.empty() : profiledMillis;
            overheadMillis = overheadMillis == null ? OptionalDouble.empty() : overheadMillis;
            overheadPercent = overheadPercent == null ? OptionalDouble.empty() : overheadPercent;
        }

        static WallScope empty() {
            return new WallScope(
                    OptionalLong.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty(),
                    OptionalDouble.empty());
        }

        static WallScope fromMetadata(Map<String, ?> metadata, String prefix) {
            Objects.requireNonNull(metadata, "metadata must not be null");
            return new WallScope(
                    optionalLong(metadata.get(prefix + ".count")),
                    optionalDouble(metadata.get(prefix + ".totalMillis")),
                    optionalDouble(metadata.get(prefix + ".averageMillis")),
                    optionalDouble(metadata.get(prefix + ".minMillis")),
                    optionalDouble(metadata.get(prefix + ".maxMillis")),
                    optionalDouble(metadata.get(prefix + ".lastMillis")),
                    optionalDouble(metadata.get(prefix + ".stddevMillis")),
                    optionalDouble(metadata.get(prefix + ".profiledMillis")),
                    optionalDouble(metadata.get(prefix + ".overheadMillis")),
                    optionalDouble(metadata.get(prefix + ".overheadPercent")));
        }

        boolean available() {
            return count.isPresent()
                    || totalMillis.isPresent()
                    || averageMillis.isPresent()
                    || minMillis.isPresent()
                    || maxMillis.isPresent()
                    || lastMillis.isPresent()
                    || stddevMillis.isPresent()
                    || profiledMillis.isPresent()
                    || overheadMillis.isPresent()
                    || overheadPercent.isPresent();
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("available", available());
            count.ifPresent(value -> map.put("count", value));
            totalMillis.ifPresent(value -> map.put("totalMillis", value));
            averageMillis.ifPresent(value -> map.put("averageMillis", value));
            minMillis.ifPresent(value -> map.put("minMillis", value));
            maxMillis.ifPresent(value -> map.put("maxMillis", value));
            lastMillis.ifPresent(value -> map.put("lastMillis", value));
            stddevMillis.ifPresent(value -> map.put("stddevMillis", value));
            profiledMillis.ifPresent(value -> map.put("profiledMillis", value));
            overheadMillis.ifPresent(value -> map.put("overheadMillis", value));
            overheadPercent.ifPresent(value -> map.put("overheadPercent", value));
            return Map.copyOf(map);
        }
    }

    public record Balance(
            boolean available,
            OptionalDouble totalMillis,
            String bottleneckGroup,
            Bucket input,
            Bucket compute,
            Bucket train,
            Bucket validation,
            Bucket optimizer,
            Bucket bottleneck) {
        public Balance {
            totalMillis = totalMillis == null ? OptionalDouble.empty() : totalMillis;
            bottleneckGroup = bottleneckGroup == null || bottleneckGroup.isBlank() ? "none" : bottleneckGroup.trim();
            input = input == null ? Bucket.empty() : input;
            compute = compute == null ? Bucket.empty() : compute;
            train = train == null ? Bucket.empty() : train;
            validation = validation == null ? Bucket.empty() : validation;
            optimizer = optimizer == null ? Bucket.empty() : optimizer;
            bottleneck = bottleneck == null ? Bucket.empty() : bottleneck;
            available = available
                    || totalMillis.isPresent()
                    || input.available()
                    || compute.available()
                    || train.available()
                    || validation.available()
                    || optimizer.available()
                    || bottleneck.available();
        }

        static Balance empty() {
            return new Balance(
                    false,
                    OptionalDouble.empty(),
                    "none",
                    Bucket.empty(),
                    Bucket.empty(),
                    Bucket.empty(),
                    Bucket.empty(),
                    Bucket.empty(),
                    Bucket.empty());
        }

        static Balance fromMetadata(Map<String, ?> metadata) {
            Objects.requireNonNull(metadata, "metadata must not be null");
            return new Balance(
                    false,
                    optionalDouble(metadata.get("runtimeProfile.totalMillis")),
                    stringValue(metadata.get("runtimeProfile.balance.bottleneckGroup"), "none"),
                    Bucket.fromMetadata(metadata, "runtimeProfile.balance.input"),
                    Bucket.fromMetadata(metadata, "runtimeProfile.balance.compute"),
                    Bucket.fromMetadata(metadata, "runtimeProfile.balance.train"),
                    Bucket.fromMetadata(metadata, "runtimeProfile.balance.validation"),
                    Bucket.fromMetadata(metadata, "runtimeProfile.balance.optimizer"),
                    Bucket.fromMetadata(metadata, "runtimeProfile.balance.bottleneck"));
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("available", available);
            totalMillis.ifPresent(value -> map.put("totalMillis", value));
            map.put("bottleneckGroup", bottleneckGroup);
            map.put("input", input.toMap());
            map.put("compute", compute.toMap());
            map.put("train", train.toMap());
            map.put("validation", validation.toMap());
            map.put("optimizer", optimizer.toMap());
            map.put("bottleneck", bottleneck.toMap());
            map.put("dominantBucket", dominantBucket().name());
            dominantPercent().ifPresent(value -> map.put("dominantPercent", value));
            return Map.copyOf(map);
        }

        public BalanceBucket dominantBucket() {
            BalanceBucket bucket = BalanceBucket.fromMetadataGroup(bottleneckGroup);
            if (bucket != BalanceBucket.NONE) {
                return bucket;
            }
            return java.util.List.of(
                            Map.entry(BalanceBucket.INPUT, input),
                            Map.entry(BalanceBucket.COMPUTE, compute),
                            Map.entry(BalanceBucket.TRAIN, train),
                            Map.entry(BalanceBucket.VALIDATION, validation),
                            Map.entry(BalanceBucket.OPTIMIZER, optimizer))
                    .stream()
                    .filter(entry -> entry.getValue().percentTotal().isPresent())
                    .max(java.util.Comparator.comparingDouble(entry -> entry.getValue()
                            .percentTotal()
                            .orElse(0.0)))
                    .map(Map.Entry::getKey)
                    .orElse(BalanceBucket.NONE);
        }

        public OptionalDouble dominantPercent() {
            return bucket(dominantBucket()).percentTotal();
        }

        public boolean inputBound(double thresholdPercent) {
            return exceeds(input, thresholdPercent);
        }

        public boolean computeBound(double thresholdPercent) {
            return exceeds(compute, thresholdPercent);
        }

        public boolean trainBound(double thresholdPercent) {
            return exceeds(train, thresholdPercent);
        }

        public boolean validationBound(double thresholdPercent) {
            return exceeds(validation, thresholdPercent);
        }

        public boolean optimizerBound(double thresholdPercent) {
            return exceeds(optimizer, thresholdPercent);
        }

        public Bucket bucket(BalanceBucket bucket) {
            return switch (bucket == null ? BalanceBucket.NONE : bucket) {
                case INPUT -> input;
                case COMPUTE -> compute;
                case TRAIN -> train;
                case VALIDATION -> validation;
                case OPTIMIZER -> optimizer;
                case NONE -> Bucket.empty();
            };
        }

        private static boolean exceeds(Bucket bucket, double thresholdPercent) {
            return Double.isFinite(thresholdPercent)
                    && bucket.percentTotal().isPresent()
                    && bucket.percentTotal().orElseThrow() > thresholdPercent;
        }
    }

    /**
     * Stable category for aggregate trainer runtime-balance buckets.
     */
    public enum BalanceBucket {
        INPUT,
        COMPUTE,
        TRAIN,
        VALIDATION,
        OPTIMIZER,
        NONE;

        static BalanceBucket fromMetadataGroup(String value) {
            if (value == null || value.isBlank()) {
                return NONE;
            }
            return switch (value.trim().toLowerCase(java.util.Locale.ROOT)) {
                case "input" -> INPUT;
                case "compute" -> COMPUTE;
                case "train" -> TRAIN;
                case "validation" -> VALIDATION;
                case "optimizer" -> OPTIMIZER;
                default -> NONE;
            };
        }
    }

    public record Bucket(
            OptionalDouble totalMillis,
            OptionalDouble percentTotal) {
        public Bucket {
            totalMillis = totalMillis == null ? OptionalDouble.empty() : totalMillis;
            percentTotal = percentTotal == null ? OptionalDouble.empty() : percentTotal;
        }

        static Bucket empty() {
            return new Bucket(OptionalDouble.empty(), OptionalDouble.empty());
        }

        static Bucket fromMetadata(Map<String, ?> metadata, String prefix) {
            Objects.requireNonNull(metadata, "metadata must not be null");
            return new Bucket(
                    optionalDouble(metadata.get(prefix + ".totalMillis")),
                    optionalDouble(metadata.get(prefix + ".percentTotal")));
        }

        boolean available() {
            return totalMillis.isPresent() || percentTotal.isPresent();
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("available", available());
            totalMillis.ifPresent(value -> map.put("totalMillis", value));
            percentTotal.ifPresent(value -> map.put("percentTotal", value));
            return Map.copyOf(map);
        }
    }

    public record Hotspot(
            String phase,
            OptionalLong count,
            OptionalDouble totalMillis,
            OptionalDouble percentTotal,
            OptionalDouble averageMillis,
            OptionalDouble minMillis,
            OptionalDouble maxMillis,
            OptionalDouble lastMillis,
            OptionalDouble stddevMillis) {
        public Hotspot {
            phase = phase == null ? "" : phase.trim();
            count = count == null ? OptionalLong.empty() : count;
            totalMillis = totalMillis == null ? OptionalDouble.empty() : totalMillis;
            percentTotal = percentTotal == null ? OptionalDouble.empty() : percentTotal;
            averageMillis = averageMillis == null ? OptionalDouble.empty() : averageMillis;
            minMillis = minMillis == null ? OptionalDouble.empty() : minMillis;
            maxMillis = maxMillis == null ? OptionalDouble.empty() : maxMillis;
            lastMillis = lastMillis == null ? OptionalDouble.empty() : lastMillis;
            stddevMillis = stddevMillis == null ? OptionalDouble.empty() : stddevMillis;
        }

        static Hotspot fromMap(Map<?, ?> map) {
            Objects.requireNonNull(map, "map must not be null");
            return new Hotspot(
                    stringValue(map.get("phase"), ""),
                    optionalLong(map.get("count")),
                    optionalDouble(map.get("totalMillis")),
                    optionalDouble(map.get("percentTotal")),
                    optionalDouble(map.get("averageMillis")),
                    optionalDouble(map.get("minMillis")),
                    optionalDouble(map.get("maxMillis")),
                    optionalDouble(map.get("lastMillis")),
                    optionalDouble(map.get("stddevMillis")));
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("phase", phase);
            count.ifPresent(value -> map.put("count", value));
            totalMillis.ifPresent(value -> map.put("totalMillis", value));
            percentTotal.ifPresent(value -> map.put("percentTotal", value));
            averageMillis.ifPresent(value -> map.put("averageMillis", value));
            minMillis.ifPresent(value -> map.put("minMillis", value));
            maxMillis.ifPresent(value -> map.put("maxMillis", value));
            lastMillis.ifPresent(value -> map.put("lastMillis", value));
            stddevMillis.ifPresent(value -> map.put("stddevMillis", value));
            return Map.copyOf(map);
        }
    }

    public record Group(
            String name,
            OptionalLong count,
            OptionalDouble totalMillis,
            OptionalDouble percentTotal,
            OptionalDouble averageMillis,
            OptionalDouble minMillis,
            OptionalDouble maxMillis,
            OptionalDouble lastMillis,
            OptionalDouble stddevMillis) {
        public Group {
            name = name == null ? "" : name.trim();
            count = count == null ? OptionalLong.empty() : count;
            totalMillis = totalMillis == null ? OptionalDouble.empty() : totalMillis;
            percentTotal = percentTotal == null ? OptionalDouble.empty() : percentTotal;
            averageMillis = averageMillis == null ? OptionalDouble.empty() : averageMillis;
            minMillis = minMillis == null ? OptionalDouble.empty() : minMillis;
            maxMillis = maxMillis == null ? OptionalDouble.empty() : maxMillis;
            lastMillis = lastMillis == null ? OptionalDouble.empty() : lastMillis;
            stddevMillis = stddevMillis == null ? OptionalDouble.empty() : stddevMillis;
        }

        static Group fromMap(Map<?, ?> map) {
            Objects.requireNonNull(map, "map must not be null");
            return new Group(
                    stringValue(map.get("name"), ""),
                    optionalLong(map.get("count")),
                    optionalDouble(map.get("totalMillis")),
                    optionalDouble(map.get("percentTotal")),
                    optionalDouble(map.get("averageMillis")),
                    optionalDouble(map.get("minMillis")),
                    optionalDouble(map.get("maxMillis")),
                    optionalDouble(map.get("lastMillis")),
                    optionalDouble(map.get("stddevMillis")));
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("name", name);
            count.ifPresent(value -> map.put("count", value));
            totalMillis.ifPresent(value -> map.put("totalMillis", value));
            percentTotal.ifPresent(value -> map.put("percentTotal", value));
            averageMillis.ifPresent(value -> map.put("averageMillis", value));
            minMillis.ifPresent(value -> map.put("minMillis", value));
            maxMillis.ifPresent(value -> map.put("maxMillis", value));
            lastMillis.ifPresent(value -> map.put("lastMillis", value));
            stddevMillis.ifPresent(value -> map.put("stddevMillis", value));
            return Map.copyOf(map);
        }
    }
}

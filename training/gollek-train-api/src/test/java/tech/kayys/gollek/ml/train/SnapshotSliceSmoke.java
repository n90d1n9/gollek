package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.SmokeAssertions.require;
import static tech.kayys.gollek.ml.train.SmokeAssertions.requireEquals;
import static tech.kayys.gollek.ml.train.SmokeAssertions.requireThrows;
import static tech.kayys.gollek.ml.train.SmokeFixtures.countingIterable;
import static tech.kayys.gollek.ml.train.SmokeFixtures.listWithNull;
import static tech.kayys.gollek.ml.train.SmokeFixtures.nestedLists;
import static tech.kayys.gollek.ml.train.SmokeFixtures.numberedList;
import static tech.kayys.gollek.ml.train.SmokeFixtures.numberedMap;
import static tech.kayys.gollek.ml.train.SmokeFixtures.stableLabel;

import java.net.URI;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;

public final class SnapshotSliceSmoke {
    public static void main(String[] args) {
        Map<String, Object> nested = TrainingReportValues.mapValue(
                Map.of("root", Map.of("items", List.of(Map.of("name", "ok")))),
                "root");
        requireEquals(List.of(Map.of("name", "ok")), nested.get("items"), "immutable nested snapshot");

        Map<String, Object> arraySnapshot = TrainingReportSnapshots.immutableMap(
                Map.of("values", new int[] {1, 2, 3}, "labels", new Object[] {Map.of("name", "a")}));
        requireEquals(List.of(1, 2, 3), arraySnapshot.get("values"), "primitive array snapshot");
        requireEquals(List.of(Map.of("name", "a")), arraySnapshot.get("labels"), "object array snapshot");

        Map<String, Object> scalarSnapshot = TrainingReportSnapshots.immutableMap(Map.of(
                "state", Thread.State.RUNNABLE,
                "path", Path.of("reports", "run.json"),
                "createdAt", Instant.parse("2026-06-05T00:00:00Z"),
                "uri", URI.create("https://example.test/report"),
                "custom", stableLabel("custom-scalar")));
        requireEquals("RUNNABLE", scalarSnapshot.get("state"), "enum scalar snapshot");
        requireEquals(Path.of("reports", "run.json").toString(), scalarSnapshot.get("path"), "path scalar snapshot");
        requireEquals("2026-06-05T00:00:00Z", scalarSnapshot.get("createdAt"), "temporal scalar snapshot");
        requireEquals("https://example.test/report", scalarSnapshot.get("uri"), "uri scalar snapshot");
        requireEquals("custom-scalar", scalarSnapshot.get("custom"), "custom scalar snapshot");

        Map<String, Object> numericSnapshot = TrainingReportSnapshots.immutableMap(Map.of(
                "finiteDouble", 0.25d,
                "finiteFloat", 0.5f,
                "nan", Double.NaN,
                "positiveInfinity", Float.POSITIVE_INFINITY,
                "negativeInfinity", Double.NEGATIVE_INFINITY));
        requireEquals(0.25d, numericSnapshot.get("finiteDouble"), "finite double snapshot");
        requireEquals(0.5f, numericSnapshot.get("finiteFloat"), "finite float snapshot");
        requireEquals("NaN", numericSnapshot.get("nan"), "nan snapshot");
        requireEquals("Infinity", numericSnapshot.get("positiveInfinity"), "positive infinity snapshot");
        requireEquals("-Infinity", numericSnapshot.get("negativeInfinity"), "negative infinity snapshot");

        Map<String, Object> optionals = new LinkedHashMap<>();
        optionals.put("label", Optional.of("train"));
        optionals.put("nested", Optional.of(Map.of("epoch", OptionalInt.of(3))));
        optionals.put("empty", Optional.empty());
        optionals.put("steps", OptionalLong.of(42L));
        optionals.put("finite", OptionalDouble.of(0.75d));
        optionals.put("nonFinite", OptionalDouble.of(Double.NaN));
        Map<String, Object> optionalSnapshot = TrainingReportSnapshots.immutableMap(optionals);
        requireEquals("train", optionalSnapshot.get("label"), "optional string snapshot");
        requireEquals(Map.of("epoch", 3), optionalSnapshot.get("nested"), "nested optional snapshot");
        require(optionalSnapshot.containsKey("empty"), "empty optional key should be preserved");
        requireEquals(null, optionalSnapshot.get("empty"), "empty optional snapshot");
        requireEquals(42L, optionalSnapshot.get("steps"), "optional long snapshot");
        requireEquals(0.75d, optionalSnapshot.get("finite"), "optional finite double snapshot");
        requireEquals("NaN", optionalSnapshot.get("nonFinite"), "optional non-finite double snapshot");

        Map<Object, Object> keyed = new LinkedHashMap<>();
        keyed.put(Thread.State.BLOCKED, "enum-key");
        keyed.put(Path.of("reports", "key"), "path-key");
        Map<String, Object> keyedSnapshot = TrainingReportSnapshots.immutableMap(keyed);
        requireEquals("enum-key", keyedSnapshot.get("BLOCKED"), "enum key snapshot");
        requireEquals("path-key", keyedSnapshot.get(Path.of("reports", "key").toString()), "path key snapshot");

        Map<Object, Object> duplicateKeys = new LinkedHashMap<>();
        duplicateKeys.put(1, "number");
        duplicateKeys.put("1", "string");
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot contains duplicate map key after string normalization: 1",
                () -> TrainingReportSnapshots.immutableMap(duplicateKeys));

        Map<String, Object> nullable = new LinkedHashMap<>();
        nullable.put("missingMetric", null);
        nullable.put("items", new Object[] {"ok", null});
        Map<String, Object> nullableSnapshot = TrainingReportSnapshots.immutableMap(nullable);
        require(nullableSnapshot.containsKey("missingMetric"), "null map key should be preserved");
        requireEquals(null, nullableSnapshot.get("missingMetric"), "null map value snapshot");
        requireEquals(listWithNull("ok", null), nullableSnapshot.get("items"), "null array item snapshot");
        requireThrows(UnsupportedOperationException.class, null, () -> nullableSnapshot.put("new", "value"));
        @SuppressWarnings("unchecked")
        List<Object> nullableItems = (List<Object>) nullableSnapshot.get("items");
        requireThrows(UnsupportedOperationException.class, null, () -> nullableItems.add("new"));

        requireThrows(NullPointerException.class, "map must not be null", () -> TrainingReportSnapshots.immutableMap(null));
        Map<String, Object> cyclicMap = new LinkedHashMap<>();
        cyclicMap.put("self", cyclicMap);
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot contains a cyclic object graph",
                () -> TrainingReportSnapshots.immutableMap(cyclicMap));
        List<Object> cyclicList = new java.util.ArrayList<>();
        cyclicList.add(cyclicList);
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot contains a cyclic object graph",
                () -> TrainingReportSnapshots.immutableSnapshot(cyclicList));
        Object[] cyclicArray = new Object[1];
        cyclicArray[0] = cyclicArray;
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot contains a cyclic object graph",
                () -> TrainingReportSnapshots.immutableSnapshot(cyclicArray));

        Object shallowNested = nestedLists("leaf", 8);
        requireEquals(shallowNested, TrainingReportSnapshots.immutableSnapshot(shallowNested), "bounded nested snapshot");
        Object tooDeepNested = nestedLists("leaf", 260);
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot exceeds maximum nesting depth: 256",
                () -> TrainingReportSnapshots.immutableSnapshot(tooDeepNested));
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot map exceeds maximum size: 100000",
                () -> TrainingReportSnapshots.immutableMap(numberedMap(100_001)));
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot array exceeds maximum size: 100000",
                () -> TrainingReportSnapshots.immutableSnapshot(new int[100_001]));
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot iterable exceeds maximum size: 100000",
                () -> TrainingReportSnapshots.immutableSnapshot(numberedList(100_001)));
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot iterable exceeds maximum size: 100000",
                () -> TrainingReportSnapshots.immutableSnapshot(countingIterable(100_001)));

        TrainingReportSnapshotPolicy tinyPolicy = new TrainingReportSnapshotPolicy(2, 3);
        requireEquals(
                List.of(1, 2, 3),
                TrainingReportSnapshots.immutableSnapshot(List.of(1, 2, 3), tinyPolicy),
                "custom policy size boundary");
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot exceeds maximum nesting depth: 2",
                () -> TrainingReportSnapshots.immutableSnapshot(nestedLists("leaf", 3), tinyPolicy));
        requireThrows(
                IllegalArgumentException.class,
                "report snapshot iterable exceeds maximum size: 3",
                () -> TrainingReportSnapshots.immutableSnapshot(List.of(1, 2, 3, 4), tinyPolicy));
        requireThrows(
                IllegalArgumentException.class,
                "maxDepth must be positive",
                () -> new TrainingReportSnapshotPolicy(0, 3));
        requireThrows(
                IllegalArgumentException.class,
                "maxContainerSize must be positive",
                () -> new TrainingReportSnapshotPolicy(2, 0));
        requireThrows(
                NullPointerException.class,
                "policy must not be null",
                () -> TrainingReportSnapshots.immutableSnapshot(List.of(1), null));
    }

}

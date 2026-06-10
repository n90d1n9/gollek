package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerRuntimeProfilerTest {
    @Test
    void recordsPhaseSnapshotsAndClampsNegativeDurations() {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();

        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, 1_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, 3_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, -1L);

        TrainerRuntimeProfiler.PhaseSnapshot snapshot =
                profiler.snapshot(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD);
        assertEquals(3L, snapshot.count());
        assertEquals(4_000_000L, snapshot.totalNanos());
        assertEquals(0L, snapshot.minNanos());
        assertEquals(3_000_000L, snapshot.maxNanos());
        assertEquals(0L, snapshot.lastNanos());
        assertEquals(4.0, snapshot.totalMillis(), 1e-6);
        assertEquals(4.0 / 3.0, snapshot.averageMillis(), 1e-6);
        assertEquals(0.0, snapshot.minMillis(), 1e-6);
        assertEquals(3.0, snapshot.maxMillis(), 1e-6);
        assertEquals(0.0, snapshot.lastMillis(), 1e-6);
        assertEquals(Math.sqrt(14.0 / 9.0), snapshot.stddevMillis(), 1e-6);
    }

    @Test
    void timesSuppliersAndExportsMetadata() {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();

        int value = profiler.time(TrainerRuntimeProfiler.Phase.TRAIN_LOSS, () -> 7);
        profiler.time(TrainerRuntimeProfiler.Phase.TRAIN_METRICS, () -> {
        });

        assertEquals(7, value);
        assertEquals(1L, profiler.snapshot(TrainerRuntimeProfiler.Phase.TRAIN_LOSS).count());
        assertEquals(1L, profiler.snapshot(TrainerRuntimeProfiler.Phase.TRAIN_METRICS).count());

        Map<String, Object> metadata = profiler.toMetadata("profile");
        assertEquals(1L, metadata.get("profile.train.loss.count"));
        assertTrue(metadata.containsKey("profile.train.loss.totalMillis"));
        assertEquals(2, metadata.get("profile.hotspotCount"));
        assertEquals(1, metadata.get("profile.groupCount"));
        assertEquals("train", metadata.get("profile.primaryGroup.name"));
    }

    @Test
    void profiledIterableRecordsInputPipelineIterationPhases() {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();
        Iterable<?> profiled = TrainerProfiledIterable.train(List.of("a", "b"), profiler);

        List<Object> values = new java.util.ArrayList<>();
        for (Object value : profiled) {
            values.add(value);
        }

        assertEquals(List.of("a", "b"), values);
        assertEquals(1L, profiler.snapshot(TrainerRuntimeProfiler.Phase.INPUT_TRAIN_ITERATOR).count());
        assertEquals(3L, profiler.snapshot(TrainerRuntimeProfiler.Phase.INPUT_TRAIN_HAS_NEXT).count());
        assertEquals(2L, profiler.snapshot(TrainerRuntimeProfiler.Phase.INPUT_TRAIN_NEXT).count());

        Map<String, Object> metadata = profiler.toMetadata("runtimeProfile");
        assertEquals(1L, metadata.get("runtimeProfile.input.train.iterator.count"));
        assertEquals(3L, metadata.get("runtimeProfile.input.train.hasNext.count"));
        assertEquals(2L, metadata.get("runtimeProfile.input.train.next.count"));
    }

    @Test
    void profiledIterablePreservesExplicitEpochLoaderViews() {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();
        Iterable<?> profiled = TrainerProfiledIterable.validation(new EpochAwareLoader(), profiler);

        Iterable<?> epochOne = ((TrainerProfiledIterable) profiled).epoch(1L);
        List<Object> values = new java.util.ArrayList<>();
        for (Object value : epochOne) {
            values.add(value);
        }

        assertEquals(List.of("epoch-1-a", "epoch-1-b"), values);
        assertEquals(1L, profiler.snapshot(TrainerRuntimeProfiler.Phase.INPUT_VALIDATION_ITERATOR).count());
        assertEquals(3L, profiler.snapshot(TrainerRuntimeProfiler.Phase.INPUT_VALIDATION_HAS_NEXT).count());
        assertEquals(2L, profiler.snapshot(TrainerRuntimeProfiler.Phase.INPUT_VALIDATION_NEXT).count());
    }

    @Test
    @SuppressWarnings("unchecked")
    void exportsRankedHotspotsForSlowestRuntimePhases() {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();

        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, 4_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_LOSS, 10_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.OPTIMIZER_STEP, 7_000_000L);

        Map<String, Object> metadata = profiler.toMetadata("runtimeProfile");
        List<Map<String, Object>> hotspots =
                (List<Map<String, Object>>) metadata.get("runtimeProfile.hotspots");
        List<Map<String, Object>> groups =
                (List<Map<String, Object>>) metadata.get("runtimeProfile.groups");

        assertEquals(3, metadata.get("runtimeProfile.hotspotCount"));
        assertEquals("train.loss", metadata.get("runtimeProfile.primaryHotspot.phase"));
        assertEquals(10.0, (double) metadata.get("runtimeProfile.primaryHotspot.totalMillis"), 1e-6);
        assertEquals(1000.0 / 21.0, (double) metadata.get("runtimeProfile.primaryHotspot.percentTotal"), 1e-6);
        assertEquals("train.loss", hotspots.get(0).get("phase"));
        assertEquals(10.0, (double) hotspots.get(0).get("minMillis"), 1e-6);
        assertEquals(10.0, (double) hotspots.get(0).get("lastMillis"), 1e-6);
        assertEquals(0.0, (double) hotspots.get(0).get("stddevMillis"), 1e-6);
        assertEquals("optimizer.step", hotspots.get(1).get("phase"));
        assertEquals("train.forward", hotspots.get(2).get("phase"));
        assertEquals(2, metadata.get("runtimeProfile.groupCount"));
        assertEquals("train", metadata.get("runtimeProfile.primaryGroup.name"));
        assertEquals(14.0, (double) metadata.get("runtimeProfile.primaryGroup.totalMillis"), 1e-6);
        assertEquals(200.0 / 3.0, (double) metadata.get("runtimeProfile.primaryGroup.percentTotal"), 1e-6);
        assertEquals("train", groups.get(0).get("name"));
        assertEquals(2L, groups.get(0).get("count"));
        assertEquals(14.0, (double) groups.get(0).get("totalMillis"), 1e-6);
        assertEquals(200.0 / 3.0, (double) groups.get(0).get("percentTotal"), 1e-6);
        assertEquals(4.0, (double) groups.get(0).get("minMillis"), 1e-6);
        assertEquals(10.0, (double) groups.get(0).get("lastMillis"), 1e-6);
        assertEquals(3.0, (double) groups.get(0).get("stddevMillis"), 1e-6);
        assertEquals("optimizer", groups.get(1).get("name"));
        assertEquals(7.0, (double) groups.get(1).get("totalMillis"), 1e-6);
    }

    @Test
    void exportsRuntimeBalanceMetadataForInputComputeAndBottleneck() {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();

        profiler.record(TrainerRuntimeProfiler.Phase.INPUT_TRAIN_NEXT, 6_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, 3_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_BACKWARD, 5_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.OPTIMIZER_STEP, 2_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.VALIDATION_FORWARD, 4_000_000L);

        Map<String, Object> metadata = profiler.toMetadata("runtimeProfile");

        assertEquals(20.0, (double) metadata.get("runtimeProfile.totalMillis"), 1e-6);
        assertEquals(6.0, (double) metadata.get("runtimeProfile.balance.input.totalMillis"), 1e-6);
        assertEquals(30.0, (double) metadata.get("runtimeProfile.balance.input.percentTotal"), 1e-6);
        assertEquals(14.0, (double) metadata.get("runtimeProfile.balance.compute.totalMillis"), 1e-6);
        assertEquals(70.0, (double) metadata.get("runtimeProfile.balance.compute.percentTotal"), 1e-6);
        assertEquals(8.0, (double) metadata.get("runtimeProfile.balance.train.totalMillis"), 1e-6);
        assertEquals(4.0, (double) metadata.get("runtimeProfile.balance.validation.totalMillis"), 1e-6);
        assertEquals(2.0, (double) metadata.get("runtimeProfile.balance.optimizer.totalMillis"), 1e-6);
        assertEquals("train", metadata.get("runtimeProfile.balance.bottleneckGroup"));
        assertEquals(8.0, (double) metadata.get("runtimeProfile.balance.bottleneck.totalMillis"), 1e-6);
        assertEquals(40.0, (double) metadata.get("runtimeProfile.balance.bottleneck.percentTotal"), 1e-6);
    }

    @Test
    void exportsWallClockScopeOverheadWithoutChangingPhaseBalance() {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();

        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, 3_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_BACKWARD, 4_000_000L);
        profiler.record(TrainerRuntimeProfiler.Scope.TRAIN_BATCH, 10_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.OPTIMIZER_STEP, 2_000_000L);
        profiler.record(TrainerRuntimeProfiler.Scope.OPTIMIZER_STEP, 5_000_000L);

        Map<String, Object> metadata = profiler.toMetadata("runtimeProfile");

        assertEquals(7.0, (double) metadata.get("runtimeProfile.balance.train.totalMillis"), 1e-6);
        assertEquals(2.0, (double) metadata.get("runtimeProfile.balance.optimizer.totalMillis"), 1e-6);
        assertEquals(15.0, (double) metadata.get("runtimeProfile.wall.totalMillis"), 1e-6);
        assertEquals(2, metadata.get("runtimeProfile.wall.scopeCount"));
        assertEquals(10.0, (double) metadata.get("runtimeProfile.wall.trainBatch.totalMillis"), 1e-6);
        assertEquals(7.0, (double) metadata.get("runtimeProfile.wall.trainBatch.profiledMillis"), 1e-6);
        assertEquals(3.0, (double) metadata.get("runtimeProfile.wall.trainBatch.overheadMillis"), 1e-6);
        assertEquals(30.0, (double) metadata.get("runtimeProfile.wall.trainBatch.overheadPercent"), 1e-6);
        assertEquals(5.0, (double) metadata.get("runtimeProfile.wall.optimizerStep.totalMillis"), 1e-6);
        assertEquals(2.0, (double) metadata.get("runtimeProfile.wall.optimizerStep.profiledMillis"), 1e-6);
        assertEquals("trainBatch", metadata.get("runtimeProfile.wall.primaryOverhead.scope"));

        TrainingReportRuntimeProfile profile = TrainingReportRuntimeProfile.fromMetadata(metadata);
        assertTrue(profile.wallClock().available());
        assertEquals(15.0, profile.wallClock().totalMillis().orElseThrow(), 1e-6);
        assertEquals(3.0, profile.wallClock().trainBatch().overheadMillis().orElseThrow(), 1e-6);
        assertTrue(TrainingReportRuntimeProfileMarkdown.render(profile).contains("### Wall-Clock Overhead"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void groupedLastDurationUsesObservationOrder() {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();

        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_LOSS, 10_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, 4_000_000L);

        Map<String, Object> metadata = profiler.toMetadata("runtimeProfile");
        List<Map<String, Object>> groups =
                (List<Map<String, Object>>) metadata.get("runtimeProfile.groups");

        assertEquals("train", metadata.get("runtimeProfile.primaryGroup.name"));
        assertEquals(14.0, (double) metadata.get("runtimeProfile.primaryGroup.totalMillis"), 1e-6);
        assertEquals(4.0, (double) metadata.get("runtimeProfile.primaryGroup.lastMillis"), 1e-6);
        assertEquals(4.0, (double) groups.get(0).get("lastMillis"), 1e-6);
    }

    @Test
    void resetClearsSnapshots() {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();

        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, 1L);
        profiler.reset();

        assertEquals(0L, profiler.snapshot(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD).count());
        assertTrue(profiler.snapshot().isEmpty());
        assertEquals(0, profiler.toMetadata("runtimeProfile").get("runtimeProfile.hotspotCount"));
        assertEquals(0, profiler.toMetadata("runtimeProfile").get("runtimeProfile.groupCount"));
        assertEquals(0.0, (double) profiler.toMetadata("runtimeProfile").get("runtimeProfile.totalMillis"), 1e-6);
        assertEquals(0.0, (double) profiler.toMetadata("runtimeProfile").get("runtimeProfile.wall.totalMillis"), 1e-6);
        assertEquals(0, profiler.toMetadata("runtimeProfile").get("runtimeProfile.wall.scopeCount"));
        assertEquals("none", profiler.toMetadata("runtimeProfile").get("runtimeProfile.balance.bottleneckGroup"));
    }

    public static final class EpochAwareLoader implements Iterable<String> {
        @Override
        public java.util.Iterator<String> iterator() {
            return List.of("base").iterator();
        }

        public Iterable<String> epoch(long epoch) {
            return List.of("epoch-" + epoch + "-a", "epoch-" + epoch + "-b");
        }
    }
}

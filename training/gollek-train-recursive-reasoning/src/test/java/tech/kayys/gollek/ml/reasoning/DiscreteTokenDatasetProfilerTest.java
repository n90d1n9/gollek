package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetProfilerTest {
    @Test
    void profilesMixedTokenDataset() {
        DiscreteTokenDatasetProfile profile = DiscreteTokenDatasetProfiler.profile(List.of(
                new DiscreteTokenDatasetExample(
                        "graph-coloring",
                        0,
                        new int[] {1, 2, 3},
                        new int[] {4},
                        5,
                        Map.of()),
                new DiscreteTokenDatasetExample(
                        "nqueens",
                        1,
                        new int[] {-1, 2},
                        new int[] {2, 2},
                        -1,
                        Map.of()),
                new DiscreteTokenDatasetExample(
                        "graph-coloring",
                        2,
                        new int[] {7},
                        new int[] {8, 9, 10},
                        2,
                        Map.of())));

        assertEquals(3, profile.exampleCount());
        assertEquals(2, profile.taskCount());
        assertEquals(2, profile.taskExampleCounts().get("graph-coloring"));
        assertEquals(1, profile.taskExampleCounts().get("nqueens"));
        assertEquals(3, profile.inputLengths().sequenceCount());
        assertEquals(1, profile.inputLengths().minLength());
        assertEquals(3, profile.inputLengths().maxLength());
        assertEquals(6L, profile.inputLengths().totalLength());
        assertEquals(2.0d, profile.inputLengths().meanLength(), 1e-9);
        assertEquals(1, profile.targetLengths().minLength());
        assertEquals(3, profile.targetLengths().maxLength());
        assertEquals(6L, profile.targetLengths().totalLength());
        assertEquals(12L, profile.observedTokenCount());
        assertEquals(9, profile.distinctTokenCount());
        assertEquals(-1, profile.minToken());
        assertEquals(10, profile.maxToken());
        assertEquals(2, profile.knownSolutionExampleCount());
        assertEquals(1, profile.unknownSolutionExampleCount());
        assertEquals(2, profile.minKnownSolutionCount());
        assertEquals(5, profile.maxKnownSolutionCount());
        assertTrue(profile.hasExamples());
        assertTrue(profile.hasKnownSolutionCounts());
        assertEquals(2.0d / 3.0d, profile.knownSolutionCoverageRate(), 1e-9);
        assertThrows(
                UnsupportedOperationException.class,
                () -> profile.taskExampleCounts().put("bad", 1));
    }

    @Test
    void profilesEmptyDatasetWithZeroStats() {
        DiscreteTokenDatasetProfile profile = DiscreteTokenDatasetProfiler.profile(List.of());

        assertEquals(0, profile.exampleCount());
        assertEquals(0, profile.taskCount());
        assertEquals(DiscreteTokenLengthStats.empty(), profile.inputLengths());
        assertEquals(DiscreteTokenLengthStats.empty(), profile.targetLengths());
        assertEquals(0L, profile.observedTokenCount());
        assertEquals(0, profile.distinctTokenCount());
        assertEquals(0, profile.minToken());
        assertEquals(0, profile.maxToken());
        assertEquals(0, profile.knownSolutionExampleCount());
        assertEquals(0, profile.unknownSolutionExampleCount());
        assertEquals(-1, profile.minKnownSolutionCount());
        assertEquals(-1, profile.maxKnownSolutionCount());
        assertFalse(profile.hasExamples());
        assertFalse(profile.hasKnownSolutionCounts());
        assertEquals(0.0d, profile.knownSolutionCoverageRate(), 1e-9);
    }

    @Test
    void splitProfilesEachPartition() {
        DiscreteTokenDatasetSplit split = DiscreteTokenDatasetSplitter.sequentialByCounts(
                List.of(
                        example("train", 0),
                        example("train", 1),
                        example("validation", 2),
                        example("test", 3)),
                1,
                1);

        assertEquals(4, split.profile().exampleCount());
        assertEquals(2, split.trainProfile().exampleCount());
        assertEquals(1, split.validationProfile().exampleCount());
        assertEquals(1, split.testProfile().exampleCount());
        assertEquals(2, split.trainProfile().taskExampleCounts().get("train"));
        assertEquals(1, split.validationProfile().taskExampleCounts().get("validation"));
        assertEquals(1, split.testProfile().taskExampleCounts().get("test"));
    }

    @Test
    void rejectsMalformedProfilesAndStats() {
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetProfiler.profile(java.util.Arrays.asList(example("task", 0), null)));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenLengthStats(1, 0, 1, 1L, 1.0d));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenLengthStats(2, 1, 3, 4L, 1.0d));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetProfile(
                        1,
                        Map.of("task", 1),
                        DiscreteTokenLengthStats.of(1, 1, 1, 1L),
                        DiscreteTokenLengthStats.of(1, 1, 1, 1L),
                        2L,
                        1,
                        0,
                        1,
                        0,
                        1,
                        1,
                        1));
    }

    private static DiscreteTokenDatasetExample example(String taskId, int index) {
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                new int[] {index + 1},
                new int[] {index + 2},
                1,
                Map.of());
    }
}

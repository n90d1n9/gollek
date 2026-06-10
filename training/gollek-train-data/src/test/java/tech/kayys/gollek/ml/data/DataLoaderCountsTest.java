package tech.kayys.gollek.ml.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class DataLoaderCountsTest {
    @Test
    void acceptsNonNegativeCounts() {
        assertEquals(0, DataLoaderCounts.requireDatasetSize(0));
        assertEquals(7, DataLoaderCounts.requireSampleCount(7));
        assertEquals(3, DataLoaderCounts.requireBatchCount(3));
    }

    @Test
    void rejectsNegativeCounts() {
        assertThrows(IllegalArgumentException.class, () -> DataLoaderCounts.requireDatasetSize(-1));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderCounts.requireSampleCount(-1));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderCounts.requireBatchCount(-1));
    }
}

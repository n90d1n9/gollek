package tech.kayys.gollek.train.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class DataLoaderBatchSizesTest {
    @Test
    void acceptsPositiveBatchSize() {
        assertEquals(32, DataLoaderBatchSizes.requirePositive(32));
    }

    @Test
    void rejectsZeroOrNegativeBatchSize() {
        assertThrows(IllegalArgumentException.class, () -> DataLoaderBatchSizes.requirePositive(0));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderBatchSizes.requirePositive(-1));
    }
}

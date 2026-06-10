package tech.kayys.gollek.train.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class DataLoaderEpochsTest {
    @Test
    void nextReturnsInitialEpochWhenReshuffleIsDisabled() {
        AtomicLong counter = DataLoaderEpochs.counter(4L);

        assertEquals(4L, DataLoaderEpochs.next(counter, false, 4L));
        assertEquals(4L, DataLoaderEpochs.next(counter, false, 4L));
    }

    @Test
    void nextAdvancesCounterWhenReshuffleIsEnabled() {
        AtomicLong counter = DataLoaderEpochs.counter(2L);

        assertEquals(2L, DataLoaderEpochs.next(counter, true, 2L));
        assertEquals(3L, DataLoaderEpochs.next(counter, true, 2L));
    }

    @Test
    void rejectsInvalidEpochState() {
        assertThrows(IllegalArgumentException.class, () -> DataLoaderEpochs.counter(-1L));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderEpochs.requireEpoch(-1L));
        assertThrows(IllegalArgumentException.class, () -> DataLoaderEpochs.next(new AtomicLong(0L), false, -1L));
        assertThrows(NullPointerException.class, () -> DataLoaderEpochs.next(null, true, 0L));
    }
}

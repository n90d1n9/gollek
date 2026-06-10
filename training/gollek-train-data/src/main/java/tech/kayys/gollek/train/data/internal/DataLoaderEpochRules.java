package tech.kayys.gollek.train.data.internal;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

public final class DataLoaderEpochRules {
    private DataLoaderEpochRules() {
    }

    public static AtomicLong counter(long initialEpoch) {
        requireInitialEpoch(initialEpoch);
        return new AtomicLong(initialEpoch);
    }

    public static long next(AtomicLong counter, boolean reshuffleEachEpoch, long initialEpoch) {
        Objects.requireNonNull(counter, "counter must not be null");
        requireInitialEpoch(initialEpoch);
        return reshuffleEachEpoch ? counter.getAndIncrement() : initialEpoch;
    }

    public static void requireEpoch(long epoch) {
        if (epoch < 0L) {
            throw new IllegalArgumentException("epoch must be non-negative, got: " + epoch);
        }
    }

    public static void requireInitialEpoch(long initialEpoch) {
        if (initialEpoch < 0L) {
            throw new IllegalArgumentException("initialEpoch must be non-negative, got: " + initialEpoch);
        }
    }
}

package tech.kayys.gollek.train.data;

import java.util.concurrent.atomic.AtomicLong;
import tech.kayys.gollek.train.data.internal.DataLoaderEpochRules;

final class DataLoaderEpochs {
    private DataLoaderEpochs() {
    }

    static AtomicLong counter(long initialEpoch) {
        return DataLoaderEpochRules.counter(initialEpoch);
    }

    static long next(AtomicLong counter, boolean reshuffleEachEpoch, long initialEpoch) {
        return DataLoaderEpochRules.next(counter, reshuffleEachEpoch, initialEpoch);
    }

    static void requireEpoch(long epoch) {
        DataLoaderEpochRules.requireEpoch(epoch);
    }

    static void requireInitialEpoch(long initialEpoch) {
        DataLoaderEpochRules.requireInitialEpoch(initialEpoch);
    }
}

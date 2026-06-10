package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.train.data.internal.DataLoaderShuffleSeedRules;

final class DataLoaderShuffleSeeds {
    private DataLoaderShuffleSeeds() {
    }

    static Long forEpoch(Long baseSeed, boolean reshuffleEachEpoch, long epoch) {
        return DataLoaderShuffleSeedRules.forEpoch(baseSeed, reshuffleEachEpoch, epoch);
    }
}

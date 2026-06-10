package tech.kayys.gollek.train.data.internal;

public final class DataLoaderShuffleSeedRules {
    private DataLoaderShuffleSeedRules() {
    }

    public static Long forEpoch(Long baseSeed, boolean reshuffleEachEpoch, long epoch) {
        if (baseSeed == null || !reshuffleEachEpoch || epoch <= 0L) {
            return baseSeed;
        }
        return mix(baseSeed, epoch);
    }

    private static long mix(long seed, long epoch) {
        long value = seed + 0x9E3779B97F4A7C15L * epoch;
        value = (value ^ (value >>> 30)) * 0xBF58476D1CE4E5B9L;
        value = (value ^ (value >>> 27)) * 0x94D049BB133111EBL;
        return value ^ (value >>> 31);
    }
}

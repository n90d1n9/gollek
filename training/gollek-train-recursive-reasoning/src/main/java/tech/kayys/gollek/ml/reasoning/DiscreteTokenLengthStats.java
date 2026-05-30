package tech.kayys.gollek.ml.reasoning;

/**
 * Compact length statistics for a token sequence collection.
 */
public record DiscreteTokenLengthStats(
        int sequenceCount,
        int minLength,
        int maxLength,
        long totalLength,
        double meanLength) {

    public DiscreteTokenLengthStats {
        if (sequenceCount < 0) {
            throw new IllegalArgumentException("sequenceCount must be >= 0 but was " + sequenceCount);
        }
        if (minLength < 0) {
            throw new IllegalArgumentException("minLength must be >= 0 but was " + minLength);
        }
        if (maxLength < 0) {
            throw new IllegalArgumentException("maxLength must be >= 0 but was " + maxLength);
        }
        if (totalLength < 0) {
            throw new IllegalArgumentException("totalLength must be >= 0 but was " + totalLength);
        }
        if (!Double.isFinite(meanLength) || meanLength < 0.0d) {
            throw new IllegalArgumentException("meanLength must be finite and >= 0 but was " + meanLength);
        }
        if (sequenceCount == 0) {
            if (minLength != 0 || maxLength != 0 || totalLength != 0 || meanLength != 0.0d) {
                throw new IllegalArgumentException("empty length stats must use zero min/max/total/mean");
            }
        } else {
            if (minLength == 0) {
                throw new IllegalArgumentException("minLength must be > 0 when sequenceCount > 0");
            }
            if (maxLength < minLength) {
                throw new IllegalArgumentException("maxLength must be >= minLength");
            }
            if (totalLength < (long) minLength * sequenceCount || totalLength > (long) maxLength * sequenceCount) {
                throw new IllegalArgumentException("totalLength is inconsistent with min/max and sequenceCount");
            }
            double expectedMean = (double) totalLength / sequenceCount;
            if (Math.abs(meanLength - expectedMean) > 1e-9d) {
                throw new IllegalArgumentException("meanLength must equal totalLength / sequenceCount");
            }
        }
    }

    public static DiscreteTokenLengthStats empty() {
        return new DiscreteTokenLengthStats(0, 0, 0, 0L, 0.0d);
    }

    public static DiscreteTokenLengthStats of(int sequenceCount, int minLength, int maxLength, long totalLength) {
        if (sequenceCount == 0) {
            return empty();
        }
        return new DiscreteTokenLengthStats(
                sequenceCount,
                minLength,
                maxLength,
                totalLength,
                (double) totalLength / sequenceCount);
    }
}

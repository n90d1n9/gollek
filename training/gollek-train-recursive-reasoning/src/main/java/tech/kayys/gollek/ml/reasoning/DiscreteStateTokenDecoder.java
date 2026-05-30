package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Objects;

/**
 * Extracts integer token predictions from recursive-reasoning state metadata.
 */
@FunctionalInterface
public interface DiscreteStateTokenDecoder {
    int[] decodeTokens(RecursiveReasoningState state);

    static DiscreteStateTokenDecoder fromMetadata(String metadataKey) {
        Objects.requireNonNull(metadataKey, "metadataKey must not be null");
        return state -> {
            Objects.requireNonNull(state, "state must not be null");
            Object value = state.metadata().get(metadataKey);
            if (value == null) {
                throw new IllegalArgumentException(
                        "state metadata does not contain token key: " + metadataKey);
            }
            return toIntTokens(value, metadataKey);
        };
    }

    static DiscreteStateTokenDecoder fromMetadata(String metadataKey, int expectedLength) {
        if (expectedLength < 0) {
            throw new IllegalArgumentException("expectedLength must be >= 0 but was " + expectedLength);
        }
        DiscreteStateTokenDecoder decoder = fromMetadata(metadataKey);
        return state -> {
            int[] tokens = decoder.decodeTokens(state);
            if (tokens.length != expectedLength) {
                throw new IllegalArgumentException(
                        "state metadata token key " + metadataKey + " must contain " + expectedLength
                                + " tokens but contained " + tokens.length);
            }
            return tokens;
        };
    }

    static int[] toIntTokens(Object value, String metadataKey) {
        Objects.requireNonNull(metadataKey, "metadataKey must not be null");
        if (value instanceof int[] ints) {
            return ints.clone();
        }
        if (value instanceof long[] longs) {
            int[] tokens = new int[longs.length];
            for (int i = 0; i < longs.length; i++) {
                tokens[i] = Math.toIntExact(longs[i]);
            }
            return tokens;
        }
        if (value instanceof Number[] numbers) {
            int[] tokens = new int[numbers.length];
            for (int i = 0; i < numbers.length; i++) {
                tokens[i] = numbers[i].intValue();
            }
            return tokens;
        }
        if (value instanceof List<?> values) {
            int[] tokens = new int[values.size()];
            for (int i = 0; i < values.size(); i++) {
                Object token = values.get(i);
                if (!(token instanceof Number number)) {
                    throw new IllegalArgumentException(
                            "metadata key " + metadataKey + " contains a non-numeric token at index " + i);
                }
                tokens[i] = number.intValue();
            }
            return tokens;
        }
        throw new IllegalArgumentException(
                "metadata key " + metadataKey + " must be int[], long[], Number[], or List<Number>");
    }
}

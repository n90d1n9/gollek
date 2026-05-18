package tech.kayys.gollek.ml.bytelatent;

import java.util.List;
import java.util.Objects;

/**
 * Batch container for byte-native language-model inputs.
 */
public record ByteSequenceBatch(List<byte[]> sequences) {

    public ByteSequenceBatch {
        Objects.requireNonNull(sequences, "sequences must not be null");
        sequences = sequences.stream()
                .map(sequence -> {
                    Objects.requireNonNull(sequence, "sequence must not be null");
                    return sequence.clone();
                })
                .toList();
    }

    @Override
    public List<byte[]> sequences() {
        return sequences.stream()
                .map(byte[]::clone)
                .toList();
    }

    public int batchSize() {
        return sequences.size();
    }
}

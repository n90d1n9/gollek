package tech.kayys.gollek.ml.bytelatent;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Lightweight text dataset bridge for byte-latent training flows.
 */
public final class TextByteSequenceDataset {
    private final List<byte[]> sequences;
    private final Charset charset;

    public TextByteSequenceDataset(List<String> texts) {
        this(texts, StandardCharsets.UTF_8);
    }

    public TextByteSequenceDataset(List<String> texts, Charset charset) {
        Objects.requireNonNull(texts, "texts must not be null");
        this.charset = Objects.requireNonNull(charset, "charset must not be null");
        this.sequences = texts.stream()
                .map(text -> {
                    Objects.requireNonNull(text, "text entry must not be null");
                    return text.getBytes(this.charset);
                })
                .toList();
    }

    public static TextByteSequenceDataset fromLines(Path file) throws IOException {
        return fromLines(file, StandardCharsets.UTF_8);
    }

    public static TextByteSequenceDataset fromLines(Path file, Charset charset) throws IOException {
        Objects.requireNonNull(file, "file must not be null");
        Objects.requireNonNull(charset, "charset must not be null");
        return new TextByteSequenceDataset(Files.readAllLines(file, charset), charset);
    }

    public int size() {
        return sequences.size();
    }

    public Charset charset() {
        return charset;
    }

    public byte[] get(int index) {
        return sequences.get(index).clone();
    }

    public ByteSequenceBatch batch(int startInclusive, int batchSize) {
        if (startInclusive < 0 || startInclusive > sequences.size()) {
            throw new IllegalArgumentException(
                    "startInclusive must be within [0, " + sequences.size() + "] but was " + startInclusive);
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0 but was " + batchSize);
        }
        int endExclusive = Math.min(startInclusive + batchSize, sequences.size());
        List<byte[]> rows = new ArrayList<>(endExclusive - startInclusive);
        for (int i = startInclusive; i < endExclusive; i++) {
            rows.add(sequences.get(i).clone());
        }
        return new ByteSequenceBatch(rows);
    }

    public ByteSequenceBatch all() {
        return batch(0, sequences.size() == 0 ? 1 : sequences.size());
    }
}

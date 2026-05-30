package tech.kayys.gollek.train.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

class TokenizedDatasetTest {

    @Test
    void fromFilesPacksDocumentsWithTokenizerEosInCallerOrder(@TempDir Path tempDir) throws IOException {
        Path first = tempDir.resolve("first.txt");
        Path second = tempDir.resolve("second.txt");
        Files.writeString(first, "one two");
        Files.writeString(second, "three four five");

        TokenizedDataset dataset = TokenizedDataset.fromFiles(List.of(first, second), tokenizer(99), 3, 2);
        Files.writeString(first, "changed after loading");

        assertEquals(2, dataset.size());
        assertArrayEquals(new float[] {1f, 2f, 99f}, dataset.get(0).input().data(), 1e-6f);
        assertArrayEquals(new float[] {2f, 99f, 3f}, dataset.get(0).label().data(), 1e-6f);
        assertArrayEquals(new float[] {99f, 3f, 4f}, dataset.get(1).input().data(), 1e-6f);
        assertArrayEquals(new float[] {3f, 4f, 5f}, dataset.get(1).label().data(), 1e-6f);
    }

    @Test
    void fromDirectoryCorpusSortsTxtFilesAndIgnoresOtherFiles(@TempDir Path tempDir) throws IOException {
        Files.writeString(tempDir.resolve("z.txt"), "three");
        Files.writeString(tempDir.resolve("a.txt"), "one two");
        Files.writeString(tempDir.resolve("notes.md"), "five");

        TokenizedDataset dataset = TokenizedDataset.fromDirectoryCorpus(tempDir, tokenizer(99), 2, 1);

        assertEquals(2, dataset.size());
        assertArrayEquals(new float[] {1f, 2f}, dataset.get(0).input().data(), 1e-6f);
        assertArrayEquals(new float[] {2f, 99f}, dataset.get(0).label().data(), 1e-6f);
        assertArrayEquals(new float[] {2f, 99f}, dataset.get(1).input().data(), 1e-6f);
        assertArrayEquals(new float[] {99f, 3f}, dataset.get(1).label().data(), 1e-6f);
    }

    @Test
    void fromFilesRequiresTokenizerEosForDocumentBoundaries(@TempDir Path tempDir) throws IOException {
        Path file = tempDir.resolve("one.txt");
        Files.writeString(file, "one two");

        assertThrows(IllegalArgumentException.class,
                () -> TokenizedDataset.fromFiles(List.of(file), tokenizer(-1), 2));
    }

    private static Tokenizer tokenizer(int eosTokenId) {
        return new Tokenizer() {
            private final Map<String, Long> ids = Map.of(
                    "one", 1L,
                    "two", 2L,
                    "three", 3L,
                    "four", 4L,
                    "five", 5L);

            @Override
            public long[] encode(String text, EncodeOptions options) {
                String trimmed = text == null ? "" : text.trim();
                List<Long> tokens = new ArrayList<>();
                if (!trimmed.isEmpty()) {
                    for (String word : trimmed.split("\\s+")) {
                        tokens.add(ids.getOrDefault(word, 0L));
                    }
                }
                if (options != null && options.addEos && eosTokenId >= 0) {
                    tokens.add((long) eosTokenId);
                }
                long[] encoded = new long[tokens.size()];
                for (int i = 0; i < tokens.size(); i++) {
                    encoded[i] = tokens.get(i);
                }
                return encoded;
            }

            @Override
            public String decode(long[] tokens, DecodeOptions options) {
                return "";
            }

            @Override
            public int vocabSize() {
                return 128;
            }

            @Override
            public int bosTokenId() {
                return -1;
            }

            @Override
            public int eosTokenId() {
                return eosTokenId;
            }

            @Override
            public int padTokenId() {
                return 0;
            }

            @Override
            public int[] allStopTokenIds() {
                return eosTokenId >= 0 ? new int[] {eosTokenId} : new int[0];
            }
        };
    }
}

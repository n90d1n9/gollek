package tech.kayys.gollek.train.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.function.UnaryOperator;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;

/**
 * Text dataset — loads text samples from a directory or file for NLP training.
 *
 * <p>
 * Supports two formats:
 * <ul>
 * <li>Classification: directory with one subdirectory per class</li>
 * <li>Language modeling: single text file or packed text corpus, split into fixed-length chunks</li>
 * </ul>
 *
 * <h3>Example — classification</h3>
 *
 * <pre>{@code
 * // Directory structure: data/pos/*.txt, data/neg/*.txt
 * var dataset = TokenizedDataset.fromDirectory(Path.of("data"), tokenizer, maxLength = 128);
 * }</pre>
 *
 * <h3>Example — language modeling</h3>
 *
 * <pre>{@code
 * var dataset = TokenizedDataset.fromDirectoryCorpus(Path.of("corpus"), tokenizer, chunkSize = 512);
 * }</pre>
 */
public final class TokenizedDataset implements Dataset<Dataset.Sample> {

    private final List<Sample> samples;

    private TokenizedDataset(List<Sample> samples) {
        this.samples = samples;
    }

    @Override
    public int size() {
        return samples.size();
    }

    @Override
    public Sample get(int index) {
        return samples.get(index);
    }

    /**
     * Loads a text classification dataset from a directory.
     *
     * <p>
     * Each subdirectory is treated as a class label (sorted alphabetically).
     * Each {@code .txt} file in a subdirectory is one sample.
     *
     * @param dir       root directory containing class subdirectories
     * @param tokenizer tokenizer for encoding text
     * @param maxLength maximum token sequence length
     * @return loaded dataset
     * @throws IOException if the directory cannot be read
     */
    public static TokenizedDataset fromDirectory(
            Path dir,
            Tokenizer tokenizer,
            int maxLength) throws IOException {

        List<Path> classDirs = new ArrayList<>();
        try (var stream = Files.list(dir)) {
            stream.filter(Files::isDirectory).sorted().forEach(classDirs::add);
        }

        List<Sample> samples = new ArrayList<>();
        for (int classIdx = 0; classIdx < classDirs.size(); classIdx++) {
            final int label = classIdx;
            try (var files = Files.walk(classDirs.get(classIdx), 1)) {
                files.filter(p -> p.toString().endsWith(".txt")).forEach(file -> {
                    try {
                        String text = Files.readString(file);
                        long[] tokens = tokenizer.encode(text,
                                EncodeOptions.defaultOptions());
                        float[] ids = new float[maxLength];
                        int padId = tokenizer.padTokenId();
                        for (int i = 0; i < maxLength; i++) {
                            ids[i] = i < tokens.length ? (float) tokens[i] : (float) padId;
                        }
                        samples.add(new Sample(
                                GradTensor.of(ids, maxLength),
                                GradTensor.scalar(label)));
                    } catch (IOException e) {
                        /* skip unreadable files */ }
                });
            }
        }
        return new TokenizedDataset(samples);
    }

    /**
     * Loads a language modeling dataset from a single text file.
     *
     * <p>
     * Tokenizes the entire file and splits into fixed-length chunks.
     * Input = chunk[0..n-1], label = chunk[1..n] (next-token prediction).
     *
     * @param file      path to text file
     * @param tokenizer tokenizer for encoding
     * @param chunkSize number of tokens per chunk
     * @return loaded dataset
     * @throws IOException if the file cannot be read
     */
    public static TokenizedDataset fromFile(
            Path file,
            Tokenizer tokenizer,
            int chunkSize) throws IOException {
        return fromFile(file, tokenizer, chunkSize, chunkSize);
    }

    /**
     * Loads a language modeling dataset from a single text file with overlapping chunks.
     *
     * @param file      path to text file
     * @param tokenizer tokenizer for encoding
     * @param chunkSize number of tokens per input/target sample
     * @param stride    step between chunk starts
     * @return loaded dataset
     * @throws IOException if the file cannot be read
     */
    public static TokenizedDataset fromFile(
            Path file,
            Tokenizer tokenizer,
            int chunkSize,
            int stride) throws IOException {

        String text = Files.readString(file);
        long[] allIds = tokenizer.encode(text, EncodeOptions.defaultOptions());
        return new TokenizedDataset(DataLoader.causalLanguageModelingDataset(allIds, chunkSize, stride).toList());
    }

    /**
     * Loads a packed language modeling dataset from multiple text files in caller-provided order.
     *
     * <p>
     * Each file is tokenized independently, then non-empty documents are packed with the tokenizer
     * EOS token between documents before next-token windows are produced.
     */
    public static TokenizedDataset fromFiles(
            List<Path> files,
            Tokenizer tokenizer,
            int chunkSize) throws IOException {
        return fromFiles(files, tokenizer, chunkSize, chunkSize);
    }

    /**
     * Loads a packed language modeling dataset from multiple text files with overlapping chunks.
     *
     * @param files     ordered text files to tokenize
     * @param tokenizer tokenizer for encoding and EOS separation
     * @param chunkSize number of tokens per input/target sample
     * @param stride    step between chunk starts
     * @return loaded packed corpus dataset
     * @throws IOException if any file cannot be read
     */
    public static TokenizedDataset fromFiles(
            List<Path> files,
            Tokenizer tokenizer,
            int chunkSize,
            int stride) throws IOException {
        Objects.requireNonNull(files, "files must not be null");
        Objects.requireNonNull(tokenizer, "tokenizer must not be null");
        long[][] documents = new long[files.size()][];
        for (int i = 0; i < files.size(); i++) {
            Path file = Objects.requireNonNull(files.get(i), "files[" + i + "] must not be null");
            documents[i] = tokenizer.encode(Files.readString(file), EncodeOptions.defaultOptions());
        }
        return packedCorpus(tokenizer, documents, chunkSize, stride);
    }

    /**
     * Loads a packed language modeling dataset from all {@code .txt} files under a directory.
     *
     * <p>
     * Files are discovered recursively and processed in sorted path order for reproducible training.
     */
    public static TokenizedDataset fromDirectoryCorpus(
            Path dir,
            Tokenizer tokenizer,
            int chunkSize) throws IOException {
        return fromDirectoryCorpus(dir, tokenizer, chunkSize, chunkSize);
    }

    /**
     * Loads a packed language modeling dataset from all {@code .txt} files under a directory with overlapping chunks.
     */
    public static TokenizedDataset fromDirectoryCorpus(
            Path dir,
            Tokenizer tokenizer,
            int chunkSize,
            int stride) throws IOException {
        Objects.requireNonNull(dir, "dir must not be null");
        List<Path> files;
        try (var stream = Files.walk(dir)) {
            files = stream
                    .filter(Files::isRegularFile)
                    .filter(TokenizedDataset::isTextFile)
                    .sorted()
                    .toList();
        }
        return fromFiles(files, tokenizer, chunkSize, stride);
    }

    private static TokenizedDataset packedCorpus(
            Tokenizer tokenizer,
            long[][] documents,
            int chunkSize,
            int stride) {
        int eosTokenId = requireEosTokenId(tokenizer);
        return new TokenizedDataset(
                DataLoader.packedCausalLanguageModelingDataset(documents, eosTokenId, chunkSize, stride).toList());
    }

    private static int requireEosTokenId(Tokenizer tokenizer) {
        Objects.requireNonNull(tokenizer, "tokenizer must not be null");
        int eosTokenId = tokenizer.eosTokenId();
        if (eosTokenId < 0) {
            throw new IllegalArgumentException("tokenizer must provide a non-negative EOS token id for packed corpus loading");
        }
        return eosTokenId;
    }

    private static boolean isTextFile(Path path) {
        Path fileName = path.getFileName();
        return fileName != null && fileName.toString().endsWith(".txt");
    }
}

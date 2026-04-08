package tech.kayys.gollek.ml.nlp;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Core tokenizer contract for the Gollek NLP SDK.
 *
 * <p>A {@code Tokenizer} converts raw text into token IDs and back.
 * Implementations include {@link BpeTokenizer} (Byte-Pair Encoding) and
 * can be loaded from a HuggingFace {@code tokenizer.json} via
 * {@link #fromFile(Path)}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Tokenizer tok = Tokenizer.fromFile(Path.of("tokenizer.json"));
 * List<Integer> ids = tok.encode("Hello world");
 * String text = tok.decode(ids);
 *
 * BatchEncoding batch = tok.batchEncode(
 *     List.of("Hello", "World"), 128, true, true);
 * }</pre>
 *
 * @see BpeTokenizer
 */
public sealed interface Tokenizer permits BpeTokenizer {

    /**
     * Encodes a single string into a list of token IDs.
     *
     * @param text input text (must not be null)
     * @return ordered list of token IDs
     */
    List<Integer> encode(String text);

    /**
     * Decodes a list of token IDs back to a string.
     *
     * @param ids token IDs (must not be null)
     * @return decoded text
     */
    String decode(List<Integer> ids);

    /** @return total vocabulary size */
    int vocabSize();

    /** @return ID of the padding token {@code [PAD]} */
    int padId();

    /** @return ID of the end-of-sequence token {@code [EOS]} */
    int eosId();

    /** @return ID of the beginning-of-sequence token {@code [BOS]} */
    int bosId();

    /** @return ID of the unknown token {@code [UNK]} */
    int unkId();

    /**
     * Batch-encodes a list of strings with optional padding and truncation.
     *
     * @param texts      list of input strings
     * @param maxLength  maximum sequence length
     * @param padding    if {@code true}, pad shorter sequences to {@code maxLength}
     * @param truncation if {@code true}, truncate longer sequences to {@code maxLength}
     * @return {@link BatchEncoding} containing input IDs and attention masks
     */
    BatchEncoding batchEncode(List<String> texts, int maxLength,
                               boolean padding, boolean truncation);

    /**
     * Loads a tokenizer from a HuggingFace {@code tokenizer.json} file.
     *
     * <p>Supports BPE tokenizers (GPT-2, RoBERTa, LLaMA style).
     *
     * @param tokenizerJson path to {@code tokenizer.json}
     * @return a ready-to-use {@link BpeTokenizer}
     * @throws IOException if the file cannot be read or parsed
     */
    static Tokenizer fromFile(Path tokenizerJson) throws IOException {
        return BpeTokenizer.fromFile(tokenizerJson);
    }

    /**
     * Batch encoding result containing padded/truncated token IDs and attention masks.
     *
     * @param inputIds      {@code [N][maxLength]} token ID arrays
     * @param attentionMask {@code [N][maxLength]} mask: 1 = real token, 0 = padding
     */
    record BatchEncoding(int[][] inputIds, int[][] attentionMask) {}
}

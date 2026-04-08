package tech.kayys.gollek.ml.nlp.tokenization;

import java.util.List;
import java.util.Map;

/**
 * Tokenizer interface for converting text to token IDs and vice versa.
 *
 * <p>Implements standard tokenization operations used in NLP models:
 * <ul>
 *   <li>Text → Token IDs: {@link #encode(String)}</li>
 *   <li>Token IDs → Text: {@link #decode(List)}</li>
 *   <li>Batch processing: {@link #encodeBatch(List)}, {@link #decodeBatch(List)}</li>
 *   <li>Token info: {@link #vocabSize()}, {@link #getTokenId(String)}</li>
 * </ul>
 *
 * <h3>Supported Tokenization Methods</h3>
 * <ul>
 *   <li><b>BPE (Byte Pair Encoding)</b> - Used by GPT2, GPT3, BERT variants</li>
 *   <li><b>WordPiece</b> - Used by BERT, RoBERTa</li>
 *   <li><b>SentencePiece</b> - Used by T5, ALBERT, XLNet</li>
 *   <li><b>Character-level</b> - Simple character tokenization</li>
 * </ul>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * // Load from HuggingFace
 * Tokenizer tokenizer = HFTokenizerLoader.load("bert-base-uncased");
 *
 * // Encode text to token IDs
 * List<Integer> tokenIds = tokenizer.encode("Hello world!");
 * // Output: [101, 7592, 2088, 999, 102]  (with special tokens)
 *
 * // Decode back to text
 * String text = tokenizer.decode(tokenIds);
 * // Output: "[CLS] hello world! [SEP]"
 *
 * // Batch processing
 * List<String> texts = List.of("Hello", "World", "Test");
 * List<List<Integer>> tokenIdsBatch = tokenizer.encodeBatch(texts);
 *
 * // Padding and truncation
 * EncodedTokens encoded = tokenizer.encode("Long text...", maxLength=128, padding=true);
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 * @see HFTokenizerLoader
 * @see BPETokenizer
 * @see WordPieceTokenizer
 * @see SentencePieceTokenizer
 */
public interface Tokenizer {

    // ─────────────────────────────────────────────────────────────────────────
    // Basic Encoding/Decoding
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Encode text to token IDs.
     *
     * @param text input text
     * @return list of token IDs
     * <p>Includes special tokens (e.g., [CLS], [SEP] for BERT)</p>
     */
    List<Integer> encode(String text);

    /**
     * Encode with detailed output including attention masks and token types.
     *
     * @param text input text
     * @param maxLength maximum sequence length (pads/truncates if needed)
     * @param padding if true, pads to maxLength
     * @return encoded tokens with metadata
     */
    EncodedTokens encodeWithMetadata(String text, int maxLength, boolean padding);

    /**
     * Decode token IDs back to text.
     *
     * @param tokenIds list of token IDs
     * @return reconstructed text (with special tokens replaced by readable text)
     */
    String decode(List<Integer> tokenIds);

    /**
     * Decode token IDs, optionally skipping special tokens.
     *
     * @param tokenIds token IDs to decode
     * @param skipSpecialTokens if true, removes [CLS], [SEP], [PAD], etc.
     * @return decoded text
     */
    String decode(List<Integer> tokenIds, boolean skipSpecialTokens);

    // ─────────────────────────────────────────────────────────────────────────
    // Batch Operations
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Encode multiple texts to token IDs.
     *
     * @param texts list of input texts
     * @return list of token ID lists (each with same length if padded)
     * <p>Automatically pads all sequences to same length</p>
     */
    List<List<Integer>> encodeBatch(List<String> texts);

    /**
     * Encode batch with padding and truncation to specific length.
     *
     * @param texts input texts
     * @param maxLength maximum sequence length
     * @param padding if true, pads to maxLength
     * @return batch of encoded tokens
     */
    List<EncodedTokens> encodeBatchWithMetadata(List<String> texts, int maxLength, boolean padding);

    /**
     * Decode multiple token ID sequences.
     *
     * @param tokenIdsList list of token ID lists
     * @return list of decoded texts
     */
    List<String> decodeBatch(List<List<Integer>> tokenIdsList);

    // ─────────────────────────────────────────────────────────────────────────
    // Tokenization Info
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Get vocabulary size.
     *
     * @return total number of tokens in vocab
     */
    int vocabSize();

    /**
     * Get token ID for a string token.
     *
     * @param token string representation of token
     * @return token ID (or -1 if not in vocab)
     */
    int getTokenId(String token);

    /**
     * Get string token for ID.
     *
     * @param tokenId token ID
     * @return string representation
     */
    String getToken(int tokenId);

    /**
     * Check if token is special (e.g., [CLS], [SEP], [PAD]).
     *
     * @param tokenId token ID
     * @return true if special token
     */
    boolean isSpecialToken(int tokenId);

    /**
     * Get all special tokens and their IDs.
     *
     * @return map of special token names to IDs
     * <p>Includes: "[PAD]" → 0, "[UNK]" → 100, "[CLS]" → 101, etc.</p>
     */
    Map<String, Integer> getSpecialTokens();

    /**
     * Get pad token ID.
     *
     * @return pad token ID (usually 0)
     */
    int getPadTokenId();

    /**
     * Get unknown token ID.
     *
     * @return unknown token ID
     */
    int getUnknownTokenId();

    /**
     * Get start token ID (e.g., [CLS]).
     *
     * @return start token ID (-1 if not applicable)
     */
    int getStartTokenId();

    /**
     * Get end token ID (e.g., [SEP]).
     *
     * @return end token ID (-1 if not applicable)
     */
    int getEndTokenId();

    // ─────────────────────────────────────────────────────────────────────────
    // Tokenizer Metadata
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Get tokenizer type/method.
     *
     * @return tokenizer type: "bpe", "wordpiece", "sentencepiece", etc.
     */
    String getTokenizerType();

    /**
     * Get model name/identifier.
     *
     * @return model identifier (e.g., "bert-base-uncased", "gpt2")
     */
    String getModelName();

    /**
     * Get model configuration.
     *
     * @return configuration map with tokenizer parameters
     */
    Map<String, Object> getConfig();

    /**
     * Get maximum sequence length supported by model.
     *
     * @return max length (-1 if unlimited)
     */
    int getMaxLength();

    // ─────────────────────────────────────────────────────────────────────────
    // Advanced Operations
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Get token probabilities for language model head.
     *
     * @param tokenId token ID to analyze
     * @return float array of logits (or null if not available)
     */
    float[] getTokenLogits(int tokenId);

    /**
     * Add custom token to tokenizer.
     *
     * @param token new token to add
     * @return assigned token ID
     * @throws IllegalStateException if tokenizer is read-only
     */
    int addToken(String token);

    /**
     * Add multiple custom tokens.
     *
     * @param tokens tokens to add
     * @return list of assigned token IDs
     */
    List<Integer> addTokens(List<String> tokens);

    // ─────────────────────────────────────────────────────────────────────────
    // Nested Types
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Encoded tokens with metadata.
     *
     * @param tokenIds token IDs
     * @param tokenTypeIds token type IDs (0 for first segment, 1 for second, etc.)
     * @param attentionMask attention mask (1 for real tokens, 0 for padding)
     * @param specialTokensMask special token mask
     */
    record EncodedTokens(
        List<Integer> tokenIds,
        List<Integer> tokenTypeIds,
        List<Integer> attentionMask,
        List<Integer> specialTokensMask
    ) {
        /**
         * Convert to primitive arrays for neural network input.
         *
         * @return array wrapper with primitive arrays
         */
        public TokenArrays toArrays() {
            return new TokenArrays(
                tokenIds.stream().mapToInt(Integer::intValue).toArray(),
                tokenTypeIds.stream().mapToInt(Integer::intValue).toArray(),
                attentionMask.stream().mapToInt(Integer::intValue).toArray()
            );
        }
    }

    /**
     * Primitive arrays for neural network input.
     */
    record TokenArrays(
        int[] tokenIds,
        int[] tokenTypeIds,
        int[] attentionMask
    ) {}

    /**
     * Builder for tokenizer configuration.
     */
    interface Builder {
        Builder maxLength(int length);
        Builder padding(boolean enable);
        Builder truncation(boolean enable);
        Builder returnAttentionMask(boolean enable);
        Builder returnTokenTypeIds(boolean enable);
        Builder addSpecialTokens(boolean enable);
        Tokenizer build();
    }
}

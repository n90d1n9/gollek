package tech.kayys.gollek.tokenizer.spi;

import java.util.Arrays;

/**
 * Stateful helper for decoding generated tokens into text pieces.
 *
 * <p>
 * Handles the complexity of multi-token BPE merges, UTF-8 byte boundaries,
 * and tokenizer-specific whitespace symbols (e.g., GPT-2's 'Ġ' or SPM's ' ').
 */
public class StreamingDecoder {

    private final Tokenizer tokenizer;
    private final DecodeOptions options;
    private long[] tokens = new long[32];
    private int tokenCount;
    private String lastFullText = "";

    public StreamingDecoder(Tokenizer tokenizer, DecodeOptions options) {
        this.tokenizer = tokenizer;
        this.options = options;
    }

    /**
     * Append a new token and return ONLY the newly decoded text delta.
     *
     * @param tokenId the next token from the model
     * @return the decoded string fragment corresponding to this token
     */
    public String decodeNext(long tokenId) {
        ensureCapacity(tokenCount + 1);
        tokens[tokenCount++] = tokenId;

        // We re-decode the full sequence to handle BPE merges and multi-byte UTF-8
        // correctly across token boundaries.
        long[] currentIds = Arrays.copyOf(tokens, tokenCount);
        String currentFullText = tokenizer.decode(currentIds, options);



        if (currentFullText.length() < lastFullText.length()) {
            // This can happen with certain special tokens or normalization rules.
            // Reset and return the full text for this step.
            lastFullText = currentFullText;
            return currentFullText;
        }

        String delta = currentFullText.substring(lastFullText.length());
        lastFullText = currentFullText;
        return delta;
    }



    /**
     * Clear the internal state for a new generation session.
     */
    public void reset() {
        tokenCount = 0;
        lastFullText = "";
    }

    public String currentText() {
        return lastFullText;
    }

    private void ensureCapacity(int required) {
        if (required <= tokens.length) {
            return;
        }
        int next = tokens.length;
        while (next < required) {
            next *= 2;
        }
        tokens = Arrays.copyOf(tokens, next);
    }
}

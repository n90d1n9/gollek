package tech.kayys.gollek.tokenizer.impl;

import java.util.List;
import tech.kayys.gollek.tokenizer.spi.PreTokenizer;

/**
 * SentencePiece-style BPE pre-tokenizer.
 *
 * <p>Many Gemma-family tokenizer.json files are converted SentencePiece BPE
 * tokenizers whose fast-tokenizer graph contains a plain
 * {@code Replace(" ", "▁")} normalizer, not SentencePiece's native
 * add-dummy-prefix behavior. The dummy-prefix behavior is therefore explicit
 * here instead of being baked in, because adding {@code ▁} after chat-control
 * tokens changes {@code user/model} into {@code ▁user/▁model} and breaks
 * instruction-tuned checkpoints.</p>
 */
public final class SentencePiecePreTokenizer implements PreTokenizer {
    private final boolean addDummyPrefix;

    public SentencePiecePreTokenizer() {
        this(false);
    }

    public SentencePiecePreTokenizer(boolean addDummyPrefix) {
        this.addDummyPrefix = addDummyPrefix;
    }

    @Override
    public List<String> split(String text) {
        if (text == null || text.isEmpty()) {
            return List.of("");
        }
        String normalized = text.replace(' ', '▁');
        if (addDummyPrefix && !normalized.isEmpty() && normalized.charAt(0) != '▁') {
            normalized = "▁" + normalized;
        }
        return List.of(normalized);
    }
}

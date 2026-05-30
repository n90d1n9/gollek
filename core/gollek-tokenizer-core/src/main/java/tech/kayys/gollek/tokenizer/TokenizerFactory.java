package tech.kayys.gollek.tokenizer;

import tech.kayys.gollek.tokenizer.loader.HuggingFaceBpeLoader;
import tech.kayys.gollek.tokenizer.impl.WordPieceTokenizer;
import tech.kayys.gollek.tokenizer.spi.*;

import java.io.IOException;

public class TokenizerFactory {

    public static Tokenizer create(ModelConfig config) {

        return switch (config.getTokenizerType()) {

            case BPE -> new HuggingFaceBpeLoader()
                    .load(config.getTokenizerPath());

            case SENTENCE_PIECE -> {
                throw new UnsupportedOperationException("SentencePiece not yet implemented");
            }

            case WORD_PIECE -> loadWordPiece(config);

            case TIKTOKEN -> {
                throw new UnsupportedOperationException("Tiktoken not yet implemented");
            }
        };
    }

    private static Tokenizer loadWordPiece(ModelConfig config) {
        try {
            String filename = config.getTokenizerPath().getFileName().toString();
            if (filename.endsWith(".json")) {
                return WordPieceTokenizer.fromTokenizerJsonFile(config.getTokenizerPath());
            }
            return WordPieceTokenizer.fromVocabFile(config.getTokenizerPath());
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to load WordPiece tokenizer from "
                    + config.getTokenizerPath(), e);
        }
    }
}

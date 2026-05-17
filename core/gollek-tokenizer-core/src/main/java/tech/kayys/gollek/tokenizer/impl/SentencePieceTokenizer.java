package tech.kayys.gollek.tokenizer.impl;

import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.nio.file.Path;

/**
 * Compatibility placeholder for the removed native SentencePiece bridge.
 */
public final class SentencePieceTokenizer implements Tokenizer {

    public SentencePieceTokenizer(Path libPath, Path modelPath) {
        throw unsupported(modelPath);
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        throw unsupported(null);
    }

    @Override
    public String decode(long[] tokens, DecodeOptions options) {
        throw unsupported(null);
    }

    @Override
    public int vocabSize() {
        return -1;
    }

    @Override
    public int bosTokenId() {
        return -1;
    }

    @Override
    public int eosTokenId() {
        return -1;
    }

    @Override
    public int padTokenId() {
        return -1;
    }

    @Override
    public int[] allStopTokenIds() {
        return new int[0];
    }

    private static UnsupportedOperationException unsupported(Path modelPath) {
        String suffix = modelPath == null ? "" : ": " + modelPath;
        return new UnsupportedOperationException(
                "Native SentencePiece is not part of the pure-Java tokenizer runtime. "
                        + "Use tokenizer.json for SentencePiece-style models" + suffix);
    }
}

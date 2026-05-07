package tech.kayys.gollek.tokenizer.spi;

public interface Tokenizer {

    long[] encode(String text, EncodeOptions options);

    String decode(long[] tokens, DecodeOptions options);

    int vocabSize();

    int bosTokenId();

    int eosTokenId();

    int padTokenId();
    
    int[] allStopTokenIds();

    default java.util.Map<String, Integer> specialTokens() {
        return java.util.Collections.emptyMap();
    }
}
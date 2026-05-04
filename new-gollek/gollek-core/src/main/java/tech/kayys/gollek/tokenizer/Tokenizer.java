package tech.kayys.gollek.tokenizer;

public interface Tokenizer {
    int[] encode(String text);

    String decode(int[] tokens);
}
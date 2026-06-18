package tech.kayys.aljabr.tokenizer.spi;

import java.util.List;

public interface PreTokenizer {
    List<String> split(String text);
}
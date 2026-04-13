package tech.kayys.gollek.ml.nlp;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * A text-processing pipeline.
 */
public class Language {
    private final String lang;
    private final Tokenizer tokenizer;
    private final List<Processor> pipeline;

    public Language(String lang, Tokenizer tokenizer) {
        this.lang = lang;
        this.tokenizer = tokenizer;
        this.pipeline = new ArrayList<>();
    }

    public void addPipe(Processor component) {
        pipeline.add(component);
    }

    /**
     * Process a single text.
     */
    public Doc process(String text) {
        Doc doc = tokenizer.tokenize(text);
        for (Processor component : pipeline) {
            doc = component.apply(doc);
        }
        return doc;
    }

    /**
     * Batch processing pipeline using Java Streams.
     * Can be parallelized by calling .parallel() on the input stream.
     */
    public Stream<Doc> pipe(Stream<String> texts) {
        return texts.map(this::process);
    }

    public interface Processor extends Function<Doc, Doc> {
        String name();
    }

    public interface Tokenizer {
        Doc tokenize(String text);
    }
}

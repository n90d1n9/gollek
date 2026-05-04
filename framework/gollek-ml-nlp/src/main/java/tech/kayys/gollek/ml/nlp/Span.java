package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.ml.tensor.Tensor;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A slice from a Doc object.
 */
public class Span {
    private final Doc doc;
    private final int start;
    private final int end;
    private final String label;
    private Tensor vector;

    public Span(Doc doc, int start, int end, String label) {
        this.doc = doc;
        this.start = start;
        this.end = end;
        this.label = label;
    }

    public Doc getDoc() { return doc; }
    public int getStart() { return start; }
    public int getEnd() { return end; }
    public String getLabel() { return label; }
    public Tensor getVector() { return vector; }
    public void setVector(Tensor vector) { this.vector = vector; }

    public List<Token> tokens() {
        return doc.getTokens().subList(start, end);
    }

    public String text() {
        return tokens().stream()
                .map(Token::getText)
                .collect(Collectors.joining(" "));
    }

    public double similarity(Span other) {
        if (this.vector == null || other.vector == null) return 0.0;
        return VectorUtils.cosineSimilarity(this.vector, other.vector);
    }

    @Override
    public String toString() { return text(); }
}

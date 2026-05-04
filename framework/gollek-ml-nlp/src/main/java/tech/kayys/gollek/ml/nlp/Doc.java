package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.ml.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/**
 * A container for accessing linguistic annotations of a processed document.
 */
public class Doc {
    private final String text;
    private final List<Token> tokens;
    private final List<Span> ents;
    private final List<Span> sents;
    private Tensor vector;

    public Doc(String text) {
        this.text = text;
        this.tokens = new ArrayList<>();
        this.ents = new ArrayList<>();
        this.sents = new ArrayList<>();
    }

    public String getText() { return text; }
    public List<Token> getTokens() { return tokens; }
    public List<Span> getEnts() { return ents; }
    public List<Span> getSents() { return sents; }
    public Tensor getVector() { return vector; }

    public void addToken(Token token) { this.tokens.add(token); }
    public void addEnt(Span ent) { this.ents.add(ent); }
    public void addSent(Span sent) { this.sents.add(sent); }
    public void setVector(Tensor vector) { this.vector = vector; }

    public Token get(int i) { return tokens.get(i); }
    public int length() { return tokens.size(); }

    public Span slice(int start, int end) {
        return new Span(this, start, end, null);
    }

    public double similarity(Doc other) {
        if (this.vector == null || other.vector == null) return 0.0;
        return VectorUtils.cosineSimilarity(this.vector, other.vector);
    }

    @Override
    public String toString() { return text; }
}

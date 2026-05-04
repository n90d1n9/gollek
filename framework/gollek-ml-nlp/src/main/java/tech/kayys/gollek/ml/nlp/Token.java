package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.ml.tensor.Tensor;
import java.util.HashMap;
import java.util.Map;

/**
 * Represents an individual token in a Doc.
 */
public class Token {
    private final Doc doc;
    private final int i;
    private final String text;
    private final int startChar;
    private final int endChar;
    
    private String pos;
    private String tag;
    private String lemma;
    private String dep;
    private int head;
    
    private boolean isAlpha;
    private boolean isStop;
    private boolean isPunct;
    
    private Tensor vector;
    private Map<String, Object> userDict;

    private Token(Builder builder) {
        this.doc = builder.doc;
        this.i = builder.i;
        this.text = builder.text;
        this.startChar = builder.startChar;
        this.endChar = builder.endChar;
        this.pos = builder.pos;
        this.tag = builder.tag;
        this.lemma = builder.lemma;
        this.dep = builder.dep;
        this.head = builder.head;
        this.isAlpha = builder.isAlpha;
        this.isStop = builder.isStop;
        this.isPunct = builder.isPunct;
        this.vector = builder.vector;
        this.userDict = builder.userDict;
    }

    public static Builder builder() {
        return new Builder();
    }

    // Getters
    public Doc getDoc() { return doc; }
    public int getI() { return i; }
    public String getText() { return text; }
    public int getStartChar() { return startChar; }
    public int getEndChar() { return endChar; }
    public String getPos() { return pos; }
    public String getTag() { return tag; }
    public String getLemma() { return lemma; }
    public String getDep() { return dep; }
    public int getHead() { return head; }
    public boolean isAlpha() { return isAlpha; }
    public boolean isStop() { return isStop; }
    public boolean isPunct() { return isPunct; }
    public Tensor getVector() { return vector; }

    // Setters for pipeline components
    public void setPos(String pos) { this.pos = pos; }
    public void setTag(String tag) { this.tag = tag; }
    public void setLemma(String lemma) { this.lemma = lemma; }
    public void setDep(String dep) { this.dep = dep; }
    public void setHead(int head) { this.head = head; }
    public void setAlpha(boolean alpha) { isAlpha = alpha; }
    public void setStop(boolean stop) { isStop = stop; }
    public void setPunct(boolean punct) { isPunct = punct; }
    public void setVector(Tensor vector) { this.vector = vector; }

    public double similarity(Token other) {
        if (this.vector == null || other.vector == null) return 0.0;
        return VectorUtils.cosineSimilarity(this.vector, other.vector);
    }

    @Override
    public String toString() { return text; }

    public static class Builder {
        private Doc doc;
        private int i;
        private String text;
        private int startChar;
        private int endChar;
        private String pos;
        private String tag;
        private String lemma;
        private String dep;
        private int head;
        private boolean isAlpha;
        private boolean isStop;
        private boolean isPunct;
        private Tensor vector;
        private Map<String, Object> userDict = new HashMap<>();

        public Builder doc(Doc doc) { this.doc = doc; return this; }
        public Builder i(int i) { this.i = i; return this; }
        public Builder text(String text) { this.text = text; return this; }
        public Builder startChar(int startChar) { this.startChar = startChar; return this; }
        public Builder endChar(int endChar) { this.endChar = endChar; return this; }
        public Builder pos(String pos) { this.pos = pos; return this; }
        public Builder tag(String tag) { this.tag = tag; return this; }
        public Builder lemma(String lemma) { this.lemma = lemma; return this; }
        public Builder dep(String dep) { this.dep = dep; return this; }
        public Builder head(int head) { this.head = head; return this; }
        public Builder isAlpha(boolean isAlpha) { this.isAlpha = isAlpha; return this; }
        public Builder isStop(boolean isStop) { this.isStop = isStop; return this; }
        public Builder isPunct(boolean isPunct) { this.isPunct = isPunct; return this; }
        public Builder vector(Tensor vector) { this.vector = vector; return this; }
        
        public Token build() { return new Token(this); }
    }
}

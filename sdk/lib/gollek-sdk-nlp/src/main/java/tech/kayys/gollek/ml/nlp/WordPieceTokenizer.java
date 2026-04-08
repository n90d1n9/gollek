package tech.kayys.gollek.ml.nlp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * WordPiece tokenizer — BERT-style subword tokenization.
 *
 * <p>Based on the WordPiece algorithm used in BERT, DistilBERT, and RoBERTa.
 * Splits unknown words into known subword pieces prefixed with {@code ##}.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Lowercase and whitespace-tokenize input</li>
 *   <li>For each word, greedily find the longest matching prefix in vocab</li>
 *   <li>Remaining suffix is prefixed with {@code ##} and matched again</li>
 * </ol>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Tokenizer tok = WordPieceTokenizer.fromVocabFile(Path.of("vocab.txt"));
 * List<Integer> ids = tok.encode("unaffable");
 * // → [un, ##aff, ##able] → [ids...]
 * }</pre>
 */
public final class WordPieceTokenizer implements Tokenizer {

    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private final int padId, bosId, eosId, unkId;
    private final int maxWordLen;

    /**
     * Creates a WordPiece tokenizer from a vocabulary map.
     *
     * @param vocab      token → ID mapping
     * @param padId      {@code [PAD]} token ID
     * @param bosId      {@code [CLS]} token ID
     * @param eosId      {@code [SEP]} token ID
     * @param unkId      {@code [UNK]} token ID
     */
    public WordPieceTokenizer(Map<String, Integer> vocab,
                               int padId, int bosId, int eosId, int unkId) {
        this.vocab       = Collections.unmodifiableMap(vocab);
        this.padId       = padId;
        this.bosId       = bosId;
        this.eosId       = eosId;
        this.unkId       = unkId;
        this.maxWordLen  = 200;
        this.reverseVocab = new HashMap<>(vocab.size());
        vocab.forEach((k, v) -> reverseVocab.put(v, k));
    }

    /**
     * Loads a WordPiece tokenizer from a BERT-style {@code vocab.txt} file.
     *
     * <p>Each line in the file is one token; line number = token ID.
     *
     * @param vocabFile path to {@code vocab.txt}
     * @return configured {@link WordPieceTokenizer}
     * @throws IOException if the file cannot be read
     */
    public static WordPieceTokenizer fromVocabFile(Path vocabFile) throws IOException {
        List<String> lines = Files.readAllLines(vocabFile);
        Map<String, Integer> vocab = new LinkedHashMap<>(lines.size());
        for (int i = 0; i < lines.size(); i++) vocab.put(lines.get(i).strip(), i);

        int padId = vocab.getOrDefault("[PAD]", 0);
        int bosId = vocab.getOrDefault("[CLS]", 101);
        int eosId = vocab.getOrDefault("[SEP]", 102);
        int unkId = vocab.getOrDefault("[UNK]", 100);
        return new WordPieceTokenizer(vocab, padId, bosId, eosId, unkId);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Lowercases input, whitespace-tokenizes, then applies WordPiece per word.
     * Prepends {@code [CLS]} and appends {@code [SEP]}.
     */
    @Override
    public List<Integer> encode(String text) {
        List<Integer> ids = new ArrayList<>();
        ids.add(bosId); // [CLS]
        for (String word : text.toLowerCase().split("\\s+")) {
            if (word.isEmpty()) continue;
            ids.addAll(wordPiece(word));
        }
        ids.add(eosId); // [SEP]
        return ids;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Joins tokens, removes {@code ##} prefixes, and strips special tokens.
     */
    @Override
    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String tok = reverseVocab.getOrDefault(id, "[UNK]");
            if (tok.equals("[CLS]") || tok.equals("[SEP]") || tok.equals("[PAD]")) continue;
            if (tok.startsWith("##")) sb.append(tok.substring(2));
            else { if (sb.length() > 0) sb.append(' '); sb.append(tok); }
        }
        return sb.toString().trim();
    }

    @Override public int vocabSize() { return vocab.size(); }
    @Override public int padId()     { return padId; }
    @Override public int eosId()     { return eosId; }
    @Override public int bosId()     { return bosId; }
    @Override public int unkId()     { return unkId; }

    @Override
    public BatchEncoding batchEncode(List<String> texts, int maxLength,
                                      boolean padding, boolean truncation) {
        int N = texts.size();
        int[][] inputIds = new int[N][];
        int[][] mask     = new int[N][];
        for (int i = 0; i < N; i++) {
            List<Integer> ids = encode(texts.get(i));
            if (truncation && ids.size() > maxLength) ids = ids.subList(0, maxLength);
            int len = ids.size(), padLen = padding ? maxLength : len;
            inputIds[i] = new int[padLen]; mask[i] = new int[padLen];
            for (int j = 0; j < len && j < padLen; j++) { inputIds[i][j] = ids.get(j); mask[i][j] = 1; }
        }
        return new BatchEncoding(inputIds, mask);
    }

    // ── WordPiece algorithm ───────────────────────────────────────────────

    private List<Integer> wordPiece(String word) {
        if (word.length() > maxWordLen) return List.of(unkId);
        List<Integer> tokens = new ArrayList<>();
        int start = 0;
        while (start < word.length()) {
            int end = word.length();
            String cur = null;
            while (start < end) {
                String substr = (start == 0 ? "" : "##") + word.substring(start, end);
                if (vocab.containsKey(substr)) { cur = substr; break; }
                end--;
            }
            if (cur == null) return List.of(unkId); // word not decomposable
            tokens.add(vocab.get(cur));
            start = end;
        }
        return tokens;
    }
}

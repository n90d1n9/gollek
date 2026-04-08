package tech.kayys.gollek.ml.nlp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Byte-Pair Encoding (BPE) tokenizer — compatible with HuggingFace GPT-2/LLaMA tokenizers.
 *
 * <p>Loads vocabulary and merge rules from a HuggingFace {@code tokenizer.json} file.
 * The BPE algorithm iteratively merges the most frequent adjacent symbol pair
 * according to the ranked merge table.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Pre-tokenize text on whitespace; prefix non-first words with {@code Ġ} (U+0120)</li>
 *   <li>Split each word into individual characters</li>
 *   <li>Repeatedly find the adjacent pair with the lowest merge rank and merge it</li>
 *   <li>Map final symbols to IDs via the vocabulary</li>
 * </ol>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * Tokenizer tok = BpeTokenizer.fromFile(Path.of("tokenizer.json"));
 * List<Integer> ids = tok.encode("Hello world");
 * // → [15496, 995]  (GPT-2 style)
 * }</pre>
 *
 * @see Tokenizer
 */
public final class BpeTokenizer implements Tokenizer {

    /** Vocabulary: symbol → token ID. */
    private final Map<String, Integer> vocab;

    /** Reverse vocabulary: token ID → symbol. */
    private final Map<Integer, String> reverseVocab;

    /**
     * Merge rules: "A B" → rank (lower rank = higher priority).
     * Key format: {@code "symbolA symbolB"}.
     */
    private final Map<String, Integer> mergeRanks;

    private final int padId, bosId, eosId, unkId;

    /**
     * Constructs a BPE tokenizer from pre-parsed vocabulary and merge rules.
     *
     * @param vocab      symbol-to-ID mapping
     * @param mergeRanks merge pair to rank mapping (key = "A B")
     * @param padId      padding token ID
     * @param bosId      beginning-of-sequence token ID
     * @param eosId      end-of-sequence token ID
     * @param unkId      unknown token ID
     */
    public BpeTokenizer(Map<String, Integer> vocab, Map<String, Integer> mergeRanks,
                        int padId, int bosId, int eosId, int unkId) {
        this.vocab       = Collections.unmodifiableMap(vocab);
        this.mergeRanks  = Collections.unmodifiableMap(mergeRanks);
        this.padId       = padId;
        this.bosId       = bosId;
        this.eosId       = eosId;
        this.unkId       = unkId;
        this.reverseVocab = new HashMap<>(vocab.size());
        vocab.forEach((k, v) -> reverseVocab.put(v, k));
    }

    // ── Tokenizer interface ───────────────────────────────────────────────

    /**
     * {@inheritDoc}
     *
     * <p>Pre-tokenizes on whitespace, prefixes non-first words with {@code Ġ},
     * then applies BPE merges per word.
     */
    @Override
    public List<Integer> encode(String text) {
        if (text == null || text.isEmpty()) return List.of();
        List<Integer> ids = new ArrayList<>();
        String[] words = text.split(" ");
        for (int w = 0; w < words.length; w++) {
            String word = (w == 0 ? "" : "\u0120") + words[w];
            List<String> symbols = bpe(word);
            for (String sym : symbols) ids.add(vocab.getOrDefault(sym, unkId));
        }
        return ids;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Joins symbols and replaces the {@code Ġ} prefix with a space.
     */
    @Override
    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) sb.append(reverseVocab.getOrDefault(id, ""));
        return sb.toString().replace("\u0120", " ").trim();
    }

    /** {@inheritDoc} */
    @Override public int vocabSize() { return vocab.size(); }

    /** {@inheritDoc} */
    @Override public int padId() { return padId; }

    /** {@inheritDoc} */
    @Override public int eosId() { return eosId; }

    /** {@inheritDoc} */
    @Override public int bosId() { return bosId; }

    /** {@inheritDoc} */
    @Override public int unkId() { return unkId; }

    /**
     * {@inheritDoc}
     *
     * <p>Each sequence is encoded independently, then padded (right-pad with {@link #padId()})
     * or truncated to {@code maxLength}.
     */
    @Override
    public BatchEncoding batchEncode(List<String> texts, int maxLength,
                                      boolean padding, boolean truncation) {
        int N = texts.size();
        int[][] inputIds = new int[N][];
        int[][] mask     = new int[N][];

        for (int i = 0; i < N; i++) {
            List<Integer> ids = encode(texts.get(i));
            if (truncation && ids.size() > maxLength) ids = ids.subList(0, maxLength);
            int len = ids.size();
            int padLen = padding ? maxLength : len;
            inputIds[i] = new int[padLen];
            mask[i]     = new int[padLen];
            for (int j = 0; j < len && j < padLen; j++) {
                inputIds[i][j] = ids.get(j);
                mask[i][j]     = 1;
            }
            // remaining slots stay 0 (padId=0) and mask=0
        }
        return new BatchEncoding(inputIds, mask);
    }

    // ── BPE core ─────────────────────────────────────────────────────────

    /**
     * Applies BPE merges to a single word (already prefixed with Ġ if needed).
     *
     * @param word the word to tokenize
     * @return list of BPE symbols
     */
    private List<String> bpe(String word) {
        // Split into individual characters (code points for Unicode safety)
        List<String> symbols = new ArrayList<>();
        word.codePoints().forEach(cp -> symbols.add(new String(Character.toChars(cp))));

        while (symbols.size() > 1) {
            // Find the pair with the lowest merge rank
            int bestRank = Integer.MAX_VALUE;
            int bestIdx  = -1;
            for (int i = 0; i < symbols.size() - 1; i++) {
                String pair = symbols.get(i) + " " + symbols.get(i + 1);
                int rank = mergeRanks.getOrDefault(pair, Integer.MAX_VALUE);
                if (rank < bestRank) { bestRank = rank; bestIdx = i; }
            }
            if (bestIdx == -1) break; // no more merges possible
            // Merge the best pair
            String merged = symbols.get(bestIdx) + symbols.get(bestIdx + 1);
            symbols.set(bestIdx, merged);
            symbols.remove(bestIdx + 1);
        }
        return symbols;
    }

    // ── Factory ───────────────────────────────────────────────────────────

    /**
     * Loads a {@link BpeTokenizer} from a HuggingFace {@code tokenizer.json} file.
     *
     * <p>Parses {@code model.vocab} and {@code model.merges} using only standard
     * Java I/O — no external JSON library required.
     *
     * @param path path to {@code tokenizer.json}
     * @return configured {@link BpeTokenizer}
     * @throws IOException if the file cannot be read
     */
    public static BpeTokenizer fromFile(Path path) throws IOException {
        String json = Files.readString(path);

        Map<String, Integer> vocab = parseVocab(json);
        Map<String, Integer> mergeRanks = parseMerges(json);

        // Special token defaults — override from added_tokens if present
        int padId = vocab.getOrDefault("<pad>",  vocab.getOrDefault("[PAD]", 0));
        int bosId = vocab.getOrDefault("<s>",    vocab.getOrDefault("[BOS]", 1));
        int eosId = vocab.getOrDefault("</s>",   vocab.getOrDefault("[EOS]", 2));
        int unkId = vocab.getOrDefault("<unk>",  vocab.getOrDefault("[UNK]", 3));

        return new BpeTokenizer(vocab, mergeRanks, padId, bosId, eosId, unkId);
    }

    // ── Minimal JSON parsers ──────────────────────────────────────────────

    /**
     * Extracts {@code model.vocab} section as a {@code Map<String, Integer>}.
     * Handles escaped unicode sequences (e.g. {@code \u0120}).
     */
    private static Map<String, Integer> parseVocab(String json) {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        int vocabStart = json.indexOf("\"vocab\"");
        if (vocabStart < 0) return vocab;
        int braceOpen = json.indexOf('{', vocabStart);
        int braceClose = findMatchingBrace(json, braceOpen);
        String section = json.substring(braceOpen + 1, braceClose);

        // Parse "token": id pairs
        int i = 0;
        while (i < section.length()) {
            int ks = section.indexOf('"', i); if (ks < 0) break;
            int ke = nextUnescapedQuote(section, ks + 1);
            String key = unescape(section.substring(ks + 1, ke));
            int colon = section.indexOf(':', ke);
            int vs = colon + 1;
            while (vs < section.length() && Character.isWhitespace(section.charAt(vs))) vs++;
            int ve = vs;
            while (ve < section.length() && (Character.isDigit(section.charAt(ve)) || section.charAt(ve) == '-')) ve++;
            vocab.put(key, Integer.parseInt(section.substring(vs, ve)));
            i = ve + 1;
        }
        return vocab;
    }

    /**
     * Extracts {@code model.merges} as a rank-ordered {@code Map<String, Integer>}.
     * Each entry in the JSON array becomes a key {@code "A B"} with its array index as rank.
     */
    private static Map<String, Integer> parseMerges(String json) {
        Map<String, Integer> ranks = new LinkedHashMap<>();
        int mergesStart = json.indexOf("\"merges\"");
        if (mergesStart < 0) return ranks;
        int arrOpen = json.indexOf('[', mergesStart);
        int arrClose = findMatchingBracket(json, arrOpen);
        String section = json.substring(arrOpen + 1, arrClose);

        int rank = 0, i = 0;
        while (i < section.length()) {
            int qs = section.indexOf('"', i); if (qs < 0) break;
            int qe = nextUnescapedQuote(section, qs + 1);
            ranks.put(section.substring(qs + 1, qe), rank++);
            i = qe + 1;
        }
        return ranks;
    }

    private static int findMatchingBrace(String s, int open) {
        int depth = 0;
        for (int i = open; i < s.length(); i++) {
            if (s.charAt(i) == '{') depth++;
            else if (s.charAt(i) == '}' && --depth == 0) return i;
        }
        return s.length() - 1;
    }

    private static int findMatchingBracket(String s, int open) {
        int depth = 0;
        for (int i = open; i < s.length(); i++) {
            if (s.charAt(i) == '[') depth++;
            else if (s.charAt(i) == ']' && --depth == 0) return i;
        }
        return s.length() - 1;
    }

    private static int nextUnescapedQuote(String s, int from) {
        for (int i = from; i < s.length(); i++) {
            if (s.charAt(i) == '"' && (i == 0 || s.charAt(i - 1) != '\\')) return i;
        }
        return s.length();
    }

    private static String unescape(String s) {
        return s.replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\t", "\t");
    }
}

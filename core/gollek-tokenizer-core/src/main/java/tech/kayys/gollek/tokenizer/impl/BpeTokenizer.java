package tech.kayys.gollek.tokenizer.impl;

import tech.kayys.gollek.tokenizer.spi.*;

import java.nio.charset.StandardCharsets;
import java.util.*;

public class BpeTokenizer implements Tokenizer {

    private final Map<String, Integer> tokenToId;
    private final Map<Integer, String> idToToken;
    private final Map<String, Integer> mergeRanks;
    private final Map<Character, String> byteEncoder;

    private final int bosTokenId;
    private final int eosTokenId;
    private final int padTokenId;
    private final int unkTokenId;

    private final PreTokenizer preTokenizer;

    public BpeTokenizer(
            Map<String, Integer> tokenToId,
            Map<Integer, String> idToToken,
            Map<String, Integer> mergeRanks,
            Map<Character, String> byteEncoder,
            int bosTokenId,
            int eosTokenId,
            int padTokenId,
            int unkTokenId,
            PreTokenizer preTokenizer) {
        this.tokenToId = tokenToId;
        this.idToToken = idToToken;
        this.mergeRanks = mergeRanks;
        this.byteEncoder = byteEncoder;
        this.bosTokenId = bosTokenId;
        this.eosTokenId = eosTokenId;
        this.padTokenId = padTokenId;
        this.unkTokenId = unkTokenId;
        this.preTokenizer = preTokenizer;
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        List<Long> tokens = new ArrayList<>();

        if (options.addBos && bosTokenId >= 0) {
            tokens.add((long) bosTokenId);
        }

        for (String word : preTokenizer.split(text)) {
            tokens.addAll(bpeEncode(word));
        }

        if (options.addEos && eosTokenId >= 0) {
            tokens.add((long) eosTokenId);
        }

        return tokens.stream().mapToLong(Long::longValue).toArray();
    }

    @Override
    public String decode(long[] tokenIds, DecodeOptions options) {
        StringBuilder sb = new StringBuilder();

        for (long tokenId : tokenIds) {
            int id = (int) tokenId;

            if (options.skipSpecialTokens &&
                    (id == bosTokenId || id == eosTokenId || id == padTokenId)) {
                continue;
            }

            sb.append(idToToken.getOrDefault(id, ""));
        }

        return decodeBytes(sb.toString());
    }

    private List<Long> bpeEncode(String word) {
        List<String> symbols = new ArrayList<>();
        for (char c : word.toCharArray()) {
            symbols.add(byteEncoder.getOrDefault(c, String.valueOf(c)));
        }

        while (symbols.size() > 1) {
            int bestIdx = -1;
            int bestRank = Integer.MAX_VALUE;

            for (int i = 0; i < symbols.size() - 1; i++) {
                String pair = symbols.get(i) + symbols.get(i + 1);
                Integer rank = mergeRanks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0)
                break;

            String merged = symbols.get(bestIdx) + symbols.get(bestIdx + 1);
            symbols.set(bestIdx, merged);
            symbols.remove(bestIdx + 1);
        }

        List<Long> ids = new ArrayList<>();
        for (String sym : symbols) {
            Integer id = tokenToId.get(sym);
            if (id != null) {
                ids.add((long) id);
            } else if (unkTokenId >= 0) {
                ids.add((long) unkTokenId);
            }
        }
        return ids;
    }

    private String decodeBytes(String text) {
        return new String(text.getBytes(StandardCharsets.UTF_8), StandardCharsets.UTF_8);
    }

    @Override
    public int vocabSize() {
        return tokenToId.size();
    }

    @Override
    public int bosTokenId() {
        return bosTokenId;
    }

    @Override
    public int eosTokenId() {
        return eosTokenId;
    }

    @Override
    public int padTokenId() {
        return padTokenId;
    }
}
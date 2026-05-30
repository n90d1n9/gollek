package tech.kayys.gollek.train.data;

import java.util.Arrays;
import java.util.Objects;
import tech.kayys.gollek.ml.autograd.GradTensor;

final class LanguageModelingDatasets {
    private LanguageModelingDatasets() {
    }

    static Dataset<Dataset.Sample> causalNextToken(int[] tokenIds, int sequenceLength) {
        return causalNextToken(tokenIds, sequenceLength, sequenceLength);
    }

    static Dataset<Dataset.Sample> causalNextToken(int[] tokenIds, int sequenceLength, int stride) {
        Objects.requireNonNull(tokenIds, "tokenIds must not be null");
        int[] copy = Arrays.copyOf(tokenIds, tokenIds.length);
        return new CausalNextTokenDataset(copy, sequenceLength, stride);
    }

    static Dataset<Dataset.Sample> causalNextToken(long[] tokenIds, int sequenceLength) {
        return causalNextToken(tokenIds, sequenceLength, sequenceLength);
    }

    static Dataset<Dataset.Sample> causalNextToken(long[] tokenIds, int sequenceLength, int stride) {
        Objects.requireNonNull(tokenIds, "tokenIds must not be null");
        int[] copy = new int[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            copy[i] = Math.toIntExact(tokenIds[i]);
        }
        return new CausalNextTokenDataset(copy, sequenceLength, stride);
    }

    static Dataset<Dataset.Sample> packedCausalNextToken(
            int[][] tokenDocuments,
            int eosTokenId,
            int sequenceLength) {
        return packedCausalNextToken(tokenDocuments, eosTokenId, sequenceLength, sequenceLength);
    }

    static Dataset<Dataset.Sample> packedCausalNextToken(
            int[][] tokenDocuments,
            int eosTokenId,
            int sequenceLength,
            int stride) {
        return new CausalNextTokenDataset(packDocuments(tokenDocuments, eosTokenId), sequenceLength, stride);
    }

    static Dataset<Dataset.Sample> packedCausalNextToken(
            long[][] tokenDocuments,
            long eosTokenId,
            int sequenceLength) {
        return packedCausalNextToken(tokenDocuments, eosTokenId, sequenceLength, sequenceLength);
    }

    static Dataset<Dataset.Sample> packedCausalNextToken(
            long[][] tokenDocuments,
            long eosTokenId,
            int sequenceLength,
            int stride) {
        return new CausalNextTokenDataset(packDocuments(tokenDocuments, eosTokenId), sequenceLength, stride);
    }

    private static int[] packDocuments(int[][] tokenDocuments, int eosTokenId) {
        Objects.requireNonNull(tokenDocuments, "tokenDocuments must not be null");
        int totalTokens = packedLength(tokenDocuments);
        int[] packed = new int[totalTokens];
        int offset = 0;
        boolean appendSeparator = false;
        for (int[] document : tokenDocuments) {
            if (document.length == 0) {
                continue;
            }
            if (appendSeparator) {
                packed[offset++] = eosTokenId;
            }
            System.arraycopy(document, 0, packed, offset, document.length);
            offset += document.length;
            appendSeparator = true;
        }
        return packed;
    }

    private static int[] packDocuments(long[][] tokenDocuments, long eosTokenId) {
        Objects.requireNonNull(tokenDocuments, "tokenDocuments must not be null");
        int eos = Math.toIntExact(eosTokenId);
        int totalTokens = packedLength(tokenDocuments);
        int[] packed = new int[totalTokens];
        int offset = 0;
        boolean appendSeparator = false;
        for (long[] document : tokenDocuments) {
            if (document.length == 0) {
                continue;
            }
            if (appendSeparator) {
                packed[offset++] = eos;
            }
            for (long tokenId : document) {
                packed[offset++] = Math.toIntExact(tokenId);
            }
            appendSeparator = true;
        }
        return packed;
    }

    private static int packedLength(int[][] tokenDocuments) {
        int totalTokens = 0;
        int nonEmptyDocuments = 0;
        for (int i = 0; i < tokenDocuments.length; i++) {
            int[] document = Objects.requireNonNull(tokenDocuments[i], "tokenDocuments[" + i + "] must not be null");
            if (document.length == 0) {
                continue;
            }
            totalTokens = Math.addExact(totalTokens, document.length);
            nonEmptyDocuments++;
        }
        return nonEmptyDocuments < 2 ? totalTokens : Math.addExact(totalTokens, nonEmptyDocuments - 1);
    }

    private static int packedLength(long[][] tokenDocuments) {
        int totalTokens = 0;
        int nonEmptyDocuments = 0;
        for (int i = 0; i < tokenDocuments.length; i++) {
            long[] document = Objects.requireNonNull(tokenDocuments[i], "tokenDocuments[" + i + "] must not be null");
            if (document.length == 0) {
                continue;
            }
            totalTokens = Math.addExact(totalTokens, document.length);
            nonEmptyDocuments++;
        }
        return nonEmptyDocuments < 2 ? totalTokens : Math.addExact(totalTokens, nonEmptyDocuments - 1);
    }

    private static final class CausalNextTokenDataset implements Dataset<Dataset.Sample> {
        private final int[] tokenIds;
        private final int sequenceLength;
        private final int stride;
        private final int size;

        private CausalNextTokenDataset(int[] tokenIds, int sequenceLength, int stride) {
            if (sequenceLength <= 0) {
                throw new IllegalArgumentException("sequenceLength must be positive, got: " + sequenceLength);
            }
            if (stride <= 0) {
                throw new IllegalArgumentException("stride must be positive, got: " + stride);
            }
            this.tokenIds = tokenIds;
            this.sequenceLength = sequenceLength;
            this.stride = stride;
            this.size = tokenIds.length <= sequenceLength
                    ? 0
                    : ((tokenIds.length - sequenceLength - 1) / stride) + 1;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public Dataset.Sample get(int index) {
            if (index < 0 || index >= size) {
                throw new IndexOutOfBoundsException("index out of range: " + index);
            }
            int offset = Math.multiplyExact(index, stride);
            float[] input = new float[sequenceLength];
            float[] label = new float[sequenceLength];
            for (int i = 0; i < sequenceLength; i++) {
                input[i] = tokenIds[offset + i];
                label[i] = tokenIds[offset + i + 1];
            }
            return new Dataset.Sample(
                    GradTensor.of(input, sequenceLength),
                    GradTensor.of(label, sequenceLength));
        }
    }
}

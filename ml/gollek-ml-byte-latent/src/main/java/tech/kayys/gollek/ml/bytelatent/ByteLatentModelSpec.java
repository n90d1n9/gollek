package tech.kayys.gollek.ml.bytelatent;

/**
 * Minimal architecture spec for byte-latent language models.
 */
public record ByteLatentModelSpec(
        int byteVocabularySize,
        int latentDimension,
        int transformerLayerCount,
        int attentionHeadCount,
        int maxByteSequenceLength) {

    public ByteLatentModelSpec {
        requirePositive(byteVocabularySize, "byteVocabularySize");
        requirePositive(latentDimension, "latentDimension");
        requirePositive(transformerLayerCount, "transformerLayerCount");
        requirePositive(attentionHeadCount, "attentionHeadCount");
        requirePositive(maxByteSequenceLength, "maxByteSequenceLength");
    }

    private static void requirePositive(int value, String label) {
        if (value <= 0) {
            throw new IllegalArgumentException(label + " must be > 0 but was " + value);
        }
    }
}

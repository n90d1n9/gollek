package tech.kayys.gollek.ml.bytelatent;

/**
 * Decodes latent codes back to byte-native sequences.
 */
public interface ByteLatentDecoder {
    ByteSequenceBatch decode(ByteLatentState state);
}

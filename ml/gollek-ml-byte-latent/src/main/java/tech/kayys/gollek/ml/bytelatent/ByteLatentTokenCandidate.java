package tech.kayys.gollek.ml.bytelatent;

/**
 * Minimal ranked next-token candidate for byte-latent inference.
 */
public record ByteLatentTokenCandidate(
        int tokenId,
        double score,
        int rank) {
}

package tech.kayys.gollek.ml.bytelatent;

import java.util.List;

/**
 * Canonical metadata for Gollek's byte-latent model family.
 */
public final class ByteLatentModelFamily {
    public static final String FAMILY_ID = "fast-byte-latent-transformer";
    public static final String DISPLAY_NAME = "Fast Byte Latent Transformer";
    public static final String PAPER_CITATION =
            "Julie Kallini et al., Fast Byte Latent Transformer, arXiv:2605.08044v1 (2026)";
    public static final String DOI = "10.48550/arXiv.2605.08044";

    private ByteLatentModelFamily() {
    }

    public static List<String> recommendedModuleIds() {
        return List.of(
                "ml:gollek-ml-language-core",
                "ml:gollek-ml-byte-latent",
                "ml:gollek-ml-byte-io",
                "runner:gollek-runner-byte-latent",
                "trainer:gollek-trainer-byte-latent");
    }
}

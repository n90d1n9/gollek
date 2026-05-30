package tech.kayys.gollek.ml.reasoning;

import java.util.List;

/**
 * Canonical metadata for Gollek's recursive reasoning model family.
 */
public final class RecursiveReasoningModelFamily {
    public static final String FAMILY_ID = "generative-recursive-reasoning";
    public static final String DISPLAY_NAME = "Generative Recursive Reasoning Models";
    public static final String SHORT_NAME = "GRAM";
    public static final String PAPER_CITATION =
            "Junyeob Baek et al., Generative Recursive Reasoning, arXiv:2605.19376 (2026)";
    public static final String DOI = "10.48550/arXiv.2605.19376";

    private RecursiveReasoningModelFamily() {
    }

    public static List<String> recommendedModuleIds() {
        return List.of(
                "ml:gollek-ml-reasoning-core",
                "ml:gollek-ml-recursive-reasoning",
                "trainer:gollek-trainer-recursive-reasoning");
    }
}

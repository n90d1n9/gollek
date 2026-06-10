package tech.kayys.gollek.spi.model;

/**
 * Stable model-family diagnostic codes used by registries and runners.
 */
public final class ModelFamilyProblemCodes {

    public static final String QUANTIZED_WEIGHT_LOADER_PENDING =
            "model_family_quantized_weight_loader_pending";
    public static final String QAT_Q4_0_LOADER_PENDING =
            "model_family_qat_q4_0_loader_pending";
    public static final String QAT_MOBILE_LOADER_PENDING =
            "model_family_qat_mobile_loader_pending";

    private ModelFamilyProblemCodes() {
    }
}

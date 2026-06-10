package tech.kayys.gollek.spi.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * Lightweight detector for quantized artifacts that need a non-default loader.
 */
public record ModelFamilyQuantizedLoaderProfile(
        String format,
        String container,
        String loaderScope,
        boolean inferredFromConfig,
        boolean gemma4MobileQat) {

    private static final ObjectMapper JSON = new ObjectMapper();

    public ModelFamilyQuantizedLoaderProfile {
        format = normalizeLabel(format);
        container = normalizeLabel(container);
        loaderScope = normalizeLabel(loaderScope);
    }

    public static ModelFamilyQuantizedLoaderProfile fromModelDir(Path modelDir) {
        if (modelDir == null) {
            return null;
        }
        Path configPath = modelDir.resolve("config.json");
        if (!Files.isRegularFile(configPath)) {
            return null;
        }
        try {
            return fromConfig(JSON.readTree(configPath.toFile()));
        } catch (IOException | RuntimeException error) {
            return null;
        }
    }

    public static ModelFamilyQuantizedLoaderProfile fromConfig(JsonNode config) {
        if (config == null || config.isMissingNode() || config.isNull()) {
            return null;
        }
        JsonNode quantization = config.path("quantization_config");
        if (quantization.isMissingNode() || quantization.isNull()) {
            return null;
        }

        String loaderScope = normalizeLabel(text(quantization.path("loader_scope")));
        if (loaderScope.startsWith("metadata_only_pending")) {
            return new ModelFamilyQuantizedLoaderProfile(
                    text(quantization.path("format")),
                    text(quantization.path("container")),
                    loaderScope,
                    false,
                    false);
        }
        if (isGemma4MobileQat(config, quantization)) {
            return new ModelFamilyQuantizedLoaderProfile(
                    "mobile",
                    "transformers",
                    "metadata_only_pending_mobile_quant_loader",
                    true,
                    true);
        }
        return null;
    }

    public List<String> problemCodes() {
        if ("q4_0".equals(format)) {
            return List.of(ModelFamilyProblemCodes.QUANTIZED_WEIGHT_LOADER_PENDING,
                    ModelFamilyProblemCodes.QAT_Q4_0_LOADER_PENDING);
        }
        if ("mobile".equals(format)) {
            return List.of(ModelFamilyProblemCodes.QUANTIZED_WEIGHT_LOADER_PENDING,
                    ModelFamilyProblemCodes.QAT_MOBILE_LOADER_PENDING);
        }
        return List.of(ModelFamilyProblemCodes.QUANTIZED_WEIGHT_LOADER_PENDING);
    }

    public String remediationHint() {
        return "Route " + description()
                + " through a supported quantized runner, or implement the pending direct loader before selecting direct SafeTensor.";
    }

    private String description() {
        String label = format.isBlank() ? "quantized weights" : format + " quantized weights";
        return container.isBlank() ? label : label + " in " + container;
    }

    private static boolean isGemma4MobileQat(JsonNode config, JsonNode quantization) {
        return "gemma4".equals(normalizeLabel(text(config.path("model_type"))))
                && "gemma".equals(normalizeLabel(text(quantization.path("quant_method"))))
                && hasFourBitHint(quantization)
                && modelTypeStartsWith(config.path("text_config"), "gemma4_text")
                && modelTypeStartsWith(config.path("vision_config"), "gemma4_vision")
                && modelTypeStartsWith(config.path("audio_config"), "gemma4_audio");
    }

    private static boolean hasFourBitHint(JsonNode quantization) {
        JsonNode numBits = quantization.path("num_bits");
        if (numBits.isNumber() && numBits.asInt() == 4) {
            return true;
        }
        String normalized = normalizeLabel(text(quantization.path("format")));
        return normalized.contains("q4") || normalized.contains("4bit") || normalized.contains("mobile");
    }

    private static boolean modelTypeStartsWith(JsonNode node, String expectedPrefix) {
        return node != null
                && !node.isMissingNode()
                && !node.isNull()
                && normalizeLabel(text(node.path("model_type"))).startsWith(expectedPrefix);
    }

    private static String text(JsonNode node) {
        return node == null || node.isMissingNode() || node.isNull() ? "" : node.asText("").trim();
    }

    private static String normalizeLabel(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }
}

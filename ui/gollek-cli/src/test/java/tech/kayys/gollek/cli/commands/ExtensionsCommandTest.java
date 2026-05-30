package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExtensionsCommandTest {
    @Test
    void jsonOutputIsParseableForAutomation() throws Exception {
        ExtensionsCommand command = new ExtensionsCommand();
        command.jsonOutput = true;

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            command.run();
        } finally {
            System.setOut(originalOut);
        }

        JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
        assertEquals(1, root.path("schemaVersion").asInt());
        assertTrue(root.path("kernel").path("displayName").isTextual());
        assertTrue(root.path("runtimeModules").isArray());
        assertTrue(root.path("runnerPlugins").path("plugins").isArray());
        assertTrue(root.path("modelFamilyBundle").path("selection").isArray());
        assertTrue(root.path("modelFamilyBundle").has("bundlePreset"));
        assertTrue(root.path("modelFamilyBundle").path("selectorSource").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("explicitSelectors").isArray());
        assertTrue(root.path("modelFamilyBundle").path("presetSelectors").isArray());
        assertTrue(root.path("modelFamilyBundle").path("defaultSelectors").isArray());
        assertTrue(root.path("modelFamilyBundle").path("policySource").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("presetRequiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("presetForbiddenFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("presetRequiredAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("presetForbiddenAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("explicitRequiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("explicitForbiddenFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("explicitRequiredAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("explicitForbiddenAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("requiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("forbiddenFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("requiredAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("forbiddenAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("policyStatus").path("status").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("policyViolations").path("missingRequiredAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("fixtureStatus").path("status").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("fixtureStatus").path("requiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("fixtureStatus").path("missingRequiredFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("status").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("healthy").isBoolean());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("summary").isTextual());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("fixtureStatus").isTextual());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("fixtureMissingRequiredFamilies")
                .isArray());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("problems").isArray());
        assertTrue(root.path("modelFamilyBundle").path("availabilityStatus").path("remediationHints").isArray());
        assertTrue(root.path("modelFamilyBundle")
                .path("availabilityStatus")
                .path("missingSelectedFamilies")
                .isArray());
        assertTrue(root.path("modelFamilyBundle").path("availableBundlePresets").isArray());
        assertTrue(root.path("modelFamilyBundle").has("activeBundlePreset"));
        assertTrue(root.path("modelFamilyBundle")
                .path("activeBundlePresetConformance")
                .path("status")
                .isTextual());
        assertTrue(root.path("modelFamilyBundle")
                .path("activeBundlePresetConformance")
                .path("selectorsMatch")
                .isBoolean());
        assertTrue(root.path("modelFamilyBundle")
                .path("activeBundlePresetConformance")
                .path("policyInputsMatch")
                .isBoolean());
        if (!root.path("modelFamilyBundle").path("activeBundlePreset").isNull()) {
            assertTrue(root.path("modelFamilyBundle")
                    .path("activeBundlePreset")
                    .path("policyStatus")
                    .path("status")
                    .isTextual());
        }
        assertTrue(root.path("modelFamilyBundle").path("bundlePresets").isArray());
        if (!root.path("modelFamilyBundle").path("bundlePresets").isEmpty()) {
            JsonNode preset = root.path("modelFamilyBundle").path("bundlePresets").get(0);
            assertTrue(preset.path("selectedFamilies").isArray());
            assertTrue(preset.path("selectedCount").canConvertToInt());
            assertTrue(preset.path("policyStatus").path("status").isTextual());
            assertTrue(preset.path("policyViolations").path("missingRequiredAliases").isArray());
        }
        assertTrue(root.path("modelFamilyBundle").path("requestedFamilies").isArray());
        assertTrue(root.path("modelFamilyBundle").path("requestedProfiles").isArray());
        assertTrue(root.path("modelFamilyBundle").path("requestedAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("bundleAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("completeAliases").isArray());
        assertTrue(root.path("modelFamilyBundle").path("partialAliases").isArray());
        assertTrue(root.path("modelFamilyPlugins").path("plugins").isArray());
        assertTrue(root.path("modelFamilyPlugins").path("capabilityMatrix").isArray());
        assertTrue(root.path("modelFamilyPlugins").path("capabilityTotals").path("families").canConvertToInt());
        assertTrue(root.path("modelFamilyPlugins").path("contractViolations").isArray());
        assertTrue(root.path("dynamicPlugins").path("plugins").isArray());
    }

    @Test
    void jsonCapabilityMatrixIncludesArchitectureAdapterHealth() throws Exception {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "json_adapter_family",
                        "JSON Adapter Family",
                        List.of("json_adapter"),
                        List.of("JsonAdapterForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/json_adapter"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture(
                        "json-adapter",
                        "json_adapter",
                        "JsonAdapterForCausalLM"));
            }
        };
        ModelFamilyPluginRegistry.global().register(plugin);

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            ExtensionsCommand command = new ExtensionsCommand();
            command.jsonOutput = true;
            command.run();
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }

        JsonNode root = new ObjectMapper().readTree(output.toString(StandardCharsets.UTF_8));
        JsonNode row = matrixEntry(root, "json_adapter_family");
        assertEquals("json-adapter", row.path("architectureAdapterIds").get(0).asText());
        assertEquals(1, row.path("architectureAdapterCount").asInt());
        assertTrue(row.path("architectureAdapterPresent").asBoolean());
        assertTrue(root.path("modelFamilyPlugins")
                .path("capabilityTotals")
                .path("architectureAdapterCount")
                .asInt() >= 1);
    }

    @Test
    void textOutputShowsPartialDirectSafetensorCaveats() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "text_caveat_family",
                        "Text Caveat Family",
                        List.of("text_caveat"),
                        List.of("TextCaveatForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/text_caveat",
                                "direct_safetensor", "experimental_text_path",
                                "moe_direct_safetensor", "pending_packed_expert_runtime"));
            }
        };
        ModelFamilyPluginRegistry.global().register(plugin);

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            new ExtensionsCommand().run();
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }

        String text = output.toString(StandardCharsets.UTF_8);
        assertTrue(text.contains("Capability matrix summary:"));
        assertTrue(text.contains("Partial direct SafeTensor caveats:"));
        assertTrue(text.contains("text_caveat_family: moe:pending_packed_expert_runtime"));
    }

    private static JsonNode matrixEntry(JsonNode root, String familyId) {
        for (JsonNode row : root.path("modelFamilyPlugins").path("capabilityMatrix")) {
            if (familyId.equals(row.path("id").asText())) {
                return row;
            }
        }
        throw new AssertionError("matrix row not found: " + familyId);
    }

    private record StubArchitecture(
            String id,
            String modelType,
            String architectureClassName) implements ModelArchitecture {

        @Override
        public List<String> supportedArchClassNames() {
            return List.of(architectureClassName);
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of(modelType);
        }

        @Override
        public String embedTokensWeight() {
            return "embed.weight";
        }

        @Override
        public String finalNormWeight() {
            return "norm.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "q.weight";
        }

        @Override
        public String layerKeyWeight(int i) {
            return "k.weight";
        }

        @Override
        public String layerValueWeight(int i) {
            return "v.weight";
        }

        @Override
        public String layerOutputWeight(int i) {
            return "o.weight";
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "attn_norm.weight";
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "gate.weight";
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "up.weight";
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "down.weight";
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "ffn_norm.weight";
        }
    }
}

package tech.kayys.gollek.cli.util;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;
import tech.kayys.gollek.plugin.runner.gguf.GgufRunnerPlugin;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;

import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PluginAvailabilityCheckerTest {
    @Test
    void cliClasspathIncludesServiceLoadedGgufRunnerPlugin() {
        assertTrue(RunnerPluginManager.getInstance().getPlugin(GgufRunnerPlugin.ID).isPresent());
    }

    @Test
    void availabilityCheckerIncludesServiceLoadedRunnerPlugins() {
        PluginAvailabilityChecker checker = new PluginAvailabilityChecker();

        assertTrue(checker.hasRunnerPlugins());
        assertTrue(checker.getRunnerPluginIds().contains(GgufRunnerPlugin.ID));
    }

    @Test
    void availabilityCheckerSummariesIncludePartialDirectCaveats() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "cli-partial-direct",
                        "CLI Partial Direct",
                        List.of("cli_partial_direct"),
                        List.of("CliPartialDirectForCausalLM"),
                        List.of(ModelFamilyCapability.CAUSAL_LM,
                                ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of(
                                "bundle_profile", "optional",
                                "origin", "3rdparty/transformers/src/transformers/models/cli_partial_direct",
                                "direct_safetensor", "experimental_text_path",
                                "moe_direct_safetensor", "pending_packed_expert_runtime"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new StubArchitecture());
            }
        };

        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            PluginAvailabilityChecker checker = new PluginAvailabilityChecker();

            assertTrue(checker.getModelFamilySupportSummaries().stream()
                    .anyMatch(summary -> summary.contains(
                            "cli-partial-direct[optional](experimental:experimental_text_path;"
                                    + "caveats=moe:pending_packed_expert_runtime)")));
            assertTrue(checker.getModelFamilyCapabilityMatrixSummaries().stream()
                    .anyMatch(summary -> summary.contains(
                            "cli-partial-direct[optional](tok=no,gguf=no,safetensor=experimental")));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void availabilityCheckerReportsModelFamilyContractViolations() {
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "cli bad contract",
                        "CLI Bad Contract",
                        List.of("cli bad type"),
                        List.of(),
                        List.of(),
                        Map.of("bundle_profile", "research"));
            }
        };

        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            PluginAvailabilityChecker checker = new PluginAvailabilityChecker();

            assertTrue(checker.getModelFamilyContractViolations().stream()
                    .anyMatch(summary -> summary.contains("invalid_family_id")));
            assertTrue(checker.getModelFamilyContractViolations().stream()
                    .anyMatch(summary -> summary.contains("unknown_bundle_profile")));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void providerNotFoundErrorIncludesActiveBundlePresetPolicy() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "direct");
        properties.setProperty("selectorSource", "preset");
        properties.setProperty("presetSelectors", "direct");
        properties.setProperty("policySource", "mixed");
        properties.setProperty("presetRequiredAliases", "direct");
        properties.setProperty("explicitRequiredFamilies", "phi");
        properties.setProperty("families", "gemma");
        properties.setProperty("availableFamilies", "bert,gemma,qwen");
        properties.setProperty("requiredFamilies", "phi");
        properties.setProperty("policyPassed", "false");
        properties.setProperty("policyViolationCount", "1");
        properties.setProperty("missingRequiredFamilies", "phi");
        properties.setProperty("fixtureRequiredSelectors", "direct");
        properties.setProperty("fixtureRequiredFamilies", "gemma,phi");
        properties.setProperty("fixturePassed", "false");
        properties.setProperty("fixtureRequiredFamilyCount", "2");
        properties.setProperty("fixtureRequiredPassedCount", "1");
        properties.setProperty("fixtureMissingRequiredCount", "1");
        properties.setProperty("fixtureProblemFamilyCount", "1");
        properties.setProperty("fixtureMissingRequiredFamilies", "phi");
        properties.setProperty("fixtureProblemFamilies", "phi");
        properties.setProperty("bundlePreset", "prod_llm");
        properties.setProperty("bundlePresets", "prod_llm");
        properties.setProperty("bundlePreset.prod_llm.description", "Lean production LLM");
        properties.setProperty("bundlePreset.prod_llm.selectors", "direct");
        properties.setProperty("bundlePreset.prod_llm.requiredAliases", "direct");
        properties.setProperty("bundlePreset.prod_llm.forbiddenAliases", "embedding");
        properties.setProperty("bundlePreset.prod_llm.selectedFamilies", "gemma");
        properties.setProperty("bundlePreset.prod_llm.policyPassed", "false");
        properties.setProperty("bundlePreset.prod_llm.policyViolationCount", "2");
        properties.setProperty("bundlePreset.prod_llm.missingRequiredFamilies", "qwen");
        properties.setProperty("bundlePreset.prod_llm.selectedForbiddenAlias.embedding.families", "bert");

        PluginAvailabilityChecker checker = new PluginAvailabilityChecker() {
            @Override
            public ModelFamilyBundleManifest getModelFamilyBundleManifest() {
                return ModelFamilyBundleManifest.fromProperties(properties);
            }
        };

        assertTrue(checker.getActiveModelFamilyBundlePresetSummary()
                .contains("prod_llm(selectors=direct, selected=1, policy=failed"));

        String message = checker.getProviderNotFoundError("missing-provider");
        assertTrue(message.contains("selectorSource=preset prod_llm"));
        assertTrue(message.contains("policySource=mixed preset+explicit"));
        assertTrue(message.contains("Packaged model-family bundle policy: failed (1 violation(s))"));
        assertTrue(message.contains("Packaged model-family fixture status: failed (1/2 required, 1 missing, 1 problem)"));
        assertTrue(message.contains("Packaged model-family missing required fixtures: phi"));
        assertTrue(message.contains("Packaged model-family problem fixtures: phi"));
        assertTrue(message.contains("Packaged bundle policy missing required families: phi"));
        assertTrue(message.contains("Active model-family bundle preset: prod_llm("));
        assertTrue(message.contains("Active model-family bundle preset policy: failed (2 violation(s), 1 selected families)"));
        assertTrue(message.contains("Active model-family bundle preset conformance: drifted"));
        assertTrue(message.contains("explicit policy override"));
        assertTrue(message.contains("Active preset missing required families: qwen"));
        assertTrue(message.contains("Active preset selected forbidden alias embedding: bert"));
    }

    @Test
    void availabilityCheckerClassifiesMissingBundledModelFamilies() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "direct");
        properties.setProperty("families", "gemma,phi");
        properties.setProperty("availableFamilies", "bert,gemma,phi");
        properties.setProperty("policyPassed", "true");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("fixturePassed", "true");
        properties.setProperty("fixtureRequiredFamilyCount", "2");
        properties.setProperty("fixtureRequiredPassedCount", "2");
        properties.setProperty("bundlePreset", "prod_llm");
        properties.setProperty("bundlePresets", "prod_llm");
        properties.setProperty("bundlePreset.prod_llm.description", "Lean production LLM");
        properties.setProperty("bundlePreset.prod_llm.selectors", "direct");
        properties.setProperty("bundlePreset.prod_llm.selectedFamilies", "gemma,phi");
        properties.setProperty("bundlePreset.prod_llm.policyPassed", "true");
        properties.setProperty("bundlePreset.prod_llm.policyViolationCount", "0");

        PluginAvailabilityChecker checker = new PluginAvailabilityChecker() {
            @Override
            public ModelFamilyBundleManifest getModelFamilyBundleManifest() {
                return ModelFamilyBundleManifest.fromProperties(properties);
            }

            @Override
            public List<String> getModelFamilyPluginIds() {
                return List.of("gemma");
            }
        };

        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                checker.getModelFamilyBundleAvailability();

        assertFalse(availability.healthy());
        assertEquals("missing_plugins", availability.status());
        assertEquals(2, availability.selectedFamilyCount());
        assertEquals(1, availability.discoveredSelectedFamilyCount());
        assertEquals(1, availability.missingSelectedFamilyCount());
        assertEquals(List.of("phi"), availability.missingSelectedFamilies());
        assertTrue(availability.problems().stream()
                .anyMatch(problem -> problem.contains("selected model-family plugins were not discovered: phi")));
        assertTrue(availability.remediationHints().stream()
                .anyMatch(hint -> hint.contains("CDI or ServiceLoader: phi")));
    }

    @Test
    void availabilityCheckerClassifiesFailedFixtureGate() {
        Properties properties = new Properties();
        properties.setProperty("schemaVersion", "1");
        properties.setProperty("bundleFingerprint", "sha256:test");
        properties.setProperty("selectors", "core");
        properties.setProperty("families", "gemma");
        properties.setProperty("availableFamilies", "gemma");
        properties.setProperty("policyPassed", "true");
        properties.setProperty("policyViolationCount", "0");
        properties.setProperty("fixtureRequiredSelectors", "core");
        properties.setProperty("fixtureRequiredFamilies", "gemma");
        properties.setProperty("fixturePassed", "false");
        properties.setProperty("fixtureRequiredFamilyCount", "1");
        properties.setProperty("fixtureRequiredPassedCount", "0");
        properties.setProperty("fixtureMissingRequiredCount", "1");
        properties.setProperty("fixtureProblemFamilyCount", "1");
        properties.setProperty("fixtureMissingRequiredFamilies", "gemma");
        properties.setProperty("fixtureProblemFamilies", "gemma");

        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                PluginAvailabilityChecker.modelFamilyBundleAvailability(
                        ModelFamilyBundleManifest.fromProperties(properties),
                        Set.of("gemma"));

        assertFalse(availability.healthy());
        assertEquals("fixture_failed", availability.status());
        assertEquals("failed", availability.fixtureStatus());
        assertEquals(Boolean.FALSE, availability.fixturePassed());
        assertEquals(1, availability.fixtureMissingRequiredCount());
        assertEquals(1, availability.fixtureProblemFamilyCount());
        assertEquals(List.of("gemma"), availability.fixtureMissingRequiredFamilies());
        assertEquals(List.of("gemma"), availability.fixtureProblemFamilies());
        assertTrue(availability.problems().stream()
                .anyMatch(problem -> problem.contains("model-family fixture gate failed")));
        assertTrue(availability.remediationHints().stream()
                .anyMatch(hint -> hint.contains("validateModelFamilyFixtures")));
    }

    private static final class StubArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "cli-stub";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("CliPartialDirectForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("cli_partial_direct");
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

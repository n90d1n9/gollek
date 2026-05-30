package tech.kayys.gollek.cli.commands;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;
import jakarta.inject.Inject;

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.context.RequestContext;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@QuarkusTest
public class ShowCommandTest {

    private static final ObjectMapper JSON = new ObjectMapper();

    @Inject
    ShowCommand showCommand;

    @InjectMock
    GollekSdk sdk;

    @TempDir
    Path tempDir;

    @Test
    public void testShowCommandModelFound() throws Exception {
        ModelInfo model = ModelInfo.builder()
                .modelId("test-model")
                .name("Test Model")
                .version("1.0")
                .requestContext(RequestContext.of("community", "community"))
                .format("GGUF")
                .metadata(Collections.emptyMap())
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();

        Mockito.when(sdk.getModelInfo(eq("test-model")))
                .thenReturn(Optional.of(model));

        showCommand.modelId = "test-model";
        showCommand.json = false;

        showCommand.run();

        Mockito.verify(sdk).getModelInfo(eq("test-model"));
    }

    @Test
    public void testShowCommandModelNotFound() throws Exception {
        Mockito.when(sdk.getModelInfo(any(String.class)))
                .thenReturn(Optional.empty());

        showCommand.modelId = "nonexistent";
        showCommand.json = false;

        showCommand.run();

        Mockito.verify(sdk).getModelInfo(eq("nonexistent"));
    }

    @Test
    public void testShowCommandJsonIncludesModelFamilyResolution() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_resolution",
                  "architectures": ["ShowResolutionForCausalLM"]
                }
                """);
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "show-resolution-family",
                        "Show Resolution Family",
                        List.of("show_resolution"),
                        List.of("ShowResolutionForCausalLM"),
                        List.of(ModelFamilyCapability.TOKENIZER),
                        Map.of("bundle_profile", "metadata_only"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.huggingFaceBpe("show-resolution-bpe"));
            }
        };
        ModelInfo model = ModelInfo.builder()
                .modelId("show-model")
                .name("Show Model")
                .requestContext(RequestContext.of("community", "community"))
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-model")))
                .thenReturn(Optional.of(model));
        ModelFamilyPluginRegistry.global().register(plugin);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode root = JSON.readTree(stdout.toString(StandardCharsets.UTF_8));
            JsonNode modelFamily = root.path("modelFamily");
            assertEquals("RESOLVED", modelFamily.path("status").asText());
            assertTrue(!modelFamily.path("requiresAttention").asBoolean());
            assertTrue(modelFamily.path("problemCodes").isEmpty());
            assertEquals("show_resolution", modelFamily.path("modelType").asText());
            assertEquals("show-resolution-family", modelFamily.path("familyIds").get(0).asText());
            assertEquals("show-resolution-bpe", modelFamily.path("tokenizers").get(0).path("id").asText());
            assertTrue(modelFamily.path("tokenizers").get(0).path("fileStatusAvailable").asBoolean());
            assertTrue(modelFamily.path("tokenizers").get(0).path("usable").asBoolean());
            assertEquals("tokenizer.json",
                    modelFamily.path("tokenizers").get(0).path("existingFileGroup").get(0).asText());
            assertTrue(modelFamily.path("summary").asText().contains("show-resolution-family"));
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
            showCommand.json = false;
        }
    }

    @Test
    public void testShowCommandJsonIncludesModelFamilyRemediationWhenMissing() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_missing_resolution",
                  "architectures": ["ShowMissingResolutionForCausalLM"]
                }
                """);
        ModelInfo model = ModelInfo.builder()
                .modelId("show-missing-model")
                .name("Show Missing Model")
                .requestContext(RequestContext.of("community", "community"))
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-missing-model")))
                .thenReturn(Optional.of(model));
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-missing-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode modelFamily = JSON.readTree(stdout.toString(StandardCharsets.UTF_8)).path("modelFamily");
            assertEquals("NOT_FOUND", modelFamily.path("status").asText());
            assertTrue(modelFamily.path("requiresAttention").asBoolean());
            assertEquals("model_family_not_found", modelFamily.path("problemCodes").get(0).asText());
            assertTrue(modelFamily.path("remediationHints").get(0).asText().contains("show_missing_resolution"));
        } finally {
            System.setOut(originalOut);
            showCommand.json = false;
        }
    }

    @Test
    public void testShowCommandJsonIncludesTokenizerFileDiagnosticsWhenMissing() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_tokenizer_missing",
                  "architectures": ["ShowTokenizerMissingForCausalLM"]
                }
                """);
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "show-tokenizer-missing-family",
                        "Show Tokenizer Missing Family",
                        List.of("show_tokenizer_missing"),
                        List.of("ShowTokenizerMissingForCausalLM"),
                        List.of(ModelFamilyCapability.TOKENIZER),
                        Map.of("bundle_profile", "metadata_only"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.wordPiece("show-tokenizer-wordpiece"));
            }
        };
        ModelInfo model = ModelInfo.builder()
                .modelId("show-tokenizer-missing-model")
                .name("Show Tokenizer Missing Model")
                .requestContext(RequestContext.of("community", "community"))
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-tokenizer-missing-model")))
                .thenReturn(Optional.of(model));
        ModelFamilyPluginRegistry.global().register(plugin);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-tokenizer-missing-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode modelFamily = JSON.readTree(stdout.toString(StandardCharsets.UTF_8)).path("modelFamily");
            JsonNode tokenizer = modelFamily.path("tokenizers").get(0);
            assertTrue(modelFamily.path("requiresAttention").asBoolean());
            assertEquals("model_family_tokenizer_files_missing",
                    modelFamily.path("problemCodes").get(0).asText());
            assertEquals("show-tokenizer-wordpiece", tokenizer.path("id").asText());
            assertTrue(tokenizer.path("fileStatusAvailable").asBoolean());
            assertTrue(!tokenizer.path("usable").asBoolean());
            assertTrue(tokenizer.path("missingFileGroups").toString().contains("vocab.txt"));
            assertTrue(modelFamily.path("remediationHints").get(0).asText()
                    .contains("show-tokenizer-wordpiece"));
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
            showCommand.json = false;
        }
    }

    @Test
    public void testShowCommandJsonIncludesDirectArchitectureRouting() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_direct",
                  "architectures": ["ShowDirectForCausalLM"]
                }
                """);
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "show-direct-family",
                        "Show Direct Family",
                        List.of("show_direct"),
                        List.of("ShowDirectForCausalLM"),
                        List.of(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("bundle_profile", "core"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(new TestArchitecture(
                        "show-direct-adapter",
                        "show_direct",
                        "ShowDirectForCausalLM"));
            }
        };
        ModelInfo model = ModelInfo.builder()
                .modelId("show-direct-model")
                .name("Show Direct Model")
                .requestContext(RequestContext.of("community", "community"))
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-direct-model")))
                .thenReturn(Optional.of(model));
        ModelFamilyPluginRegistry.global().register(plugin);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-direct-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode directArchitecture = JSON.readTree(stdout.toString(StandardCharsets.UTF_8))
                    .path("modelFamily")
                    .path("directArchitecture");
            assertTrue(directArchitecture.path("directSupportExpected").asBoolean());
            assertEquals("show-direct-adapter", directArchitecture.path("adapterIds").get(0).asText());
            assertEquals("show-direct-adapter", directArchitecture.path("selectedAdapterId").asText());
            assertEquals("model_type", directArchitecture.path("selectedBy").asText());
            assertTrue(directArchitecture.path("problemCodes").isEmpty());
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
            showCommand.json = false;
        }
    }

    @Test
    public void testShowCommandJsonFlagsDirectFamilyWithoutArchitectureAdapter() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), """
                {
                  "model_type": "show_direct_missing_adapter",
                  "architectures": ["ShowDirectMissingAdapterForCausalLM"]
                }
                """);
        ModelFamilyPlugin plugin = new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        "show-direct-missing-adapter-family",
                        "Show Direct Missing Adapter Family",
                        List.of("show_direct_missing_adapter"),
                        List.of("ShowDirectMissingAdapterForCausalLM"),
                        List.of(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("bundle_profile", "core"));
            }
        };
        ModelInfo model = ModelInfo.builder()
                .modelId("show-direct-missing-adapter-model")
                .name("Show Direct Missing Adapter Model")
                .requestContext(RequestContext.of("community", "community"))
                .format("SAFETENSORS")
                .metadata(Map.of("path", tempDir.toString()))
                .build();

        Mockito.when(sdk.getModelInfo(eq("show-direct-missing-adapter-model")))
                .thenReturn(Optional.of(model));
        ModelFamilyPluginRegistry.global().register(plugin);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        try {
            System.setOut(new PrintStream(stdout, true, StandardCharsets.UTF_8));
            showCommand.modelId = "show-direct-missing-adapter-model";
            showCommand.json = true;

            showCommand.run();

            JsonNode modelFamily = JSON.readTree(stdout.toString(StandardCharsets.UTF_8)).path("modelFamily");
            JsonNode directArchitecture = modelFamily.path("directArchitecture");
            assertTrue(modelFamily.path("requiresAttention").asBoolean());
            assertEquals("model_family_architecture_adapters_missing",
                    modelFamily.path("problemCodes").get(0).asText());
            assertEquals("model_family_architecture_adapters_missing",
                    directArchitecture.path("problemCodes").get(0).asText());
            assertTrue(modelFamily.path("remediationHints").get(0).asText().contains("architecture adapter"));
        } finally {
            System.setOut(originalOut);
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
            showCommand.json = false;
        }
    }

    private record TestArchitecture(
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

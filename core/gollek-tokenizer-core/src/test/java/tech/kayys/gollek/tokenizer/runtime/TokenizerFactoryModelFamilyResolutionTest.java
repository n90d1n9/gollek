package tech.kayys.gollek.tokenizer.runtime;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.gollek.spi.model.ModelFamilyResolution;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TokenizerFactoryModelFamilyResolutionTest {

    @TempDir
    Path tempDir;

    @Test
    void resolvesModelFamilyMetadataForTokenizerRuntime() throws Exception {
        ModelFamilyPlugin plugin = wordPiecePlugin("tokenizer-resolution-family", "tokenizer_resolution");
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            writeConfig(tempDir, "tokenizer_resolution", "TokenizerResolutionForMaskedLM");
            writeWordPieceVocab(tempDir);

            ModelFamilyResolution resolution = TokenizerFactory.resolveModelFamily(tempDir).orElseThrow();

            assertEquals(ModelFamilyResolution.Status.RESOLVED, resolution.status());
            assertEquals("tokenizer-resolution-family", resolution.primaryFamilyId().orElseThrow());
            assertEquals(List.of("tokenizer_resolution-wordpiece"), resolution.tokenizerDescriptors().stream()
                    .map(ModelTokenizerDescriptor::id)
                    .toList());
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void loadsTokenizerThroughResolvedModelFamilyDescriptor() throws Exception {
        ModelFamilyPlugin plugin = wordPiecePlugin("tokenizer-load-family", "tokenizer_load");
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            writeConfig(tempDir, "tokenizer_load", "TokenizerLoadForMaskedLM");
            writeWordPieceVocab(tempDir);

            Tokenizer tokenizer = TokenizerFactory.load(tempDir, null);

            assertEquals(6, tokenizer.vocabSize());
            assertTrue(tokenizer.encode("hello", EncodeOptions.defaultOptions()).length > 0);
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void inspectsMissingTokenizerFilesForResolvedFamily() throws Exception {
        ModelFamilyPlugin plugin = wordPiecePlugin("tokenizer-missing-family", "tokenizer_missing");
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            writeConfig(tempDir, "tokenizer_missing", "TokenizerMissingForMaskedLM");

            ModelFamilyTokenizerInspection inspection =
                    TokenizerFactory.inspectModelFamilyTokenizers(tempDir).orElseThrow();

            assertTrue(inspection.requiresAttention());
            assertEquals(List.of("model_family_tokenizer_files_missing"), inspection.problemCodes());
            assertEquals(List.of(), inspection.usableDescriptorIds());
            assertEquals("tokenizer_missing-wordpiece", inspection.descriptors().get(0).id());
            assertTrue(inspection.descriptors().get(0).missingFileGroups().stream()
                    .anyMatch(group -> group.contains("vocab.txt")));
            assertTrue(inspection.remediationHints().stream()
                    .anyMatch(hint -> hint.contains("tokenizer_missing-wordpiece")));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    @Test
    void tokenizerLoadFailureIncludesResolvedFamilyFileHints() throws Exception {
        ModelFamilyPlugin plugin = wordPiecePlugin("tokenizer-failure-family", "tokenizer_failure");
        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            writeConfig(tempDir, "tokenizer_failure", "TokenizerFailureForMaskedLM");

            IOException error = assertThrows(IOException.class, () -> TokenizerFactory.load(tempDir, null));

            assertTrue(error.getMessage().contains("model_family_tokenizer_files_missing"));
            assertTrue(error.getMessage().contains("tokenizer_failure-wordpiece"));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    private static ModelFamilyPlugin wordPiecePlugin(String familyId, String modelType) {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        familyId,
                        familyId,
                        List.of(modelType),
                        List.of("TokenizerResolutionForMaskedLM", "TokenizerLoadForMaskedLM"),
                        List.of(ModelFamilyCapability.TOKENIZER),
                        Map.of("bundle_profile", "metadata_only"));
            }

            @Override
            public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
                return List.of(ModelTokenizerDescriptor.wordPiece(modelType + "-wordpiece"));
            }
        };
    }

    private static void writeConfig(Path dir, String modelType, String architecture) throws Exception {
        Files.writeString(dir.resolve("config.json"), """
                {
                  "model_type": "%s",
                  "architectures": ["%s"]
                }
                """.formatted(modelType, architecture));
    }

    private static void writeWordPieceVocab(Path dir) throws Exception {
        Files.writeString(dir.resolve("vocab.txt"), """
                [PAD]
                [UNK]
                [CLS]
                [SEP]
                hello
                world
                """);
    }
}

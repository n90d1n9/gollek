package tech.kayys.gollek.tokenizer.runtime;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.tokenizer.impl.BpeTokenizer;
import tech.kayys.gollek.tokenizer.impl.Gpt2PreTokenizer;
import tech.kayys.gollek.tokenizer.impl.HuggingFaceBpeTokenizer;
import tech.kayys.gollek.tokenizer.impl.SentencePiecePreTokenizer;
import tech.kayys.gollek.tokenizer.impl.WordPieceTokenizer;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.gollek.spi.model.ModelFamilyResolution;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.TokenizerType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Factory for creating Tokenizer instances based on model directory content.
 */
public class TokenizerFactory {
    private static final String JAVA_ONLY_ENV = "GOLLEK_TOKENIZER_JAVA_ONLY";
    private static final String JAVA_ONLY_PROP = "gollek.tokenizer.java-only";

    /**
     * Automatically detect and load a tokenizer from the given model directory.
     *
     * @param modelDir the directory containing tokenizer files
     * @param nativeLibPath optional path to the native SPM bridge library (required for SentencePiece)
     * @return a configured Tokenizer instance
     * @throws IOException if files are missing or unreadable
     */
    public static Tokenizer load(Path modelDir, Path nativeLibPath) throws IOException {
        modelDir = normalizeModelDir(modelDir);
        Optional<ModelFamilyTokenizerInspection> inspection = inspectModelFamilyTokenizers(modelDir);
        Optional<Tokenizer> pluginTokenizer = tryLoadFromModelFamilyPlugin(modelDir, nativeLibPath, inspection);
        if (pluginTokenizer.isPresent()) {
            return pluginTokenizer.get();
        }
        try {
            return loadDetected(modelDir, nativeLibPath);
        } catch (IOException error) {
            if (inspection.isPresent() && inspection.get().requiresAttention()) {
                throw new IOException(error.getMessage() + ". " + tokenizerInspectionHint(inspection.get()), error);
            }
            throw error;
        }
    }

    private static Path normalizeModelDir(Path modelDir) throws IOException {
        if (modelDir == null) {
            throw new IOException("Model path is null");
        }
        if (Files.isRegularFile(modelDir)) {
            Path parent = modelDir.getParent();
            if (parent == null) {
                throw new IOException("Model path has no parent directory: " + modelDir);
            }
            modelDir = parent;
        }
        return modelDir;
    }

    /**
     * Resolve a model directory's Hugging Face config to attached model-family metadata.
     */
    public static Optional<ModelFamilyResolution> resolveModelFamily(Path modelDir) throws IOException {
        Path normalizedModelDir = normalizeModelDir(modelDir);
        Optional<ModelConfig> modelConfig = loadModelConfig(normalizedModelDir);
        if (modelConfig.isEmpty()) {
            return Optional.empty();
        }
        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.discoverServiceLoaderPlugins();
        return Optional.of(registry.resolve(modelConfig.get()));
    }

    /**
     * Inspect advertised model-family tokenizer descriptors against files on disk.
     */
    public static Optional<ModelFamilyTokenizerInspection> inspectModelFamilyTokenizers(Path modelDir)
            throws IOException {
        Path normalizedModelDir = normalizeModelDir(modelDir);
        Optional<ModelFamilyResolution> resolution = resolveModelFamily(normalizedModelDir);
        return resolution.map(modelFamily -> ModelFamilyTokenizerInspection.inspect(normalizedModelDir, modelFamily));
    }

    private static Tokenizer loadDetected(Path modelDir, Path nativeLibPath) throws IOException {
        Path hfConfig = modelDir.resolve("tokenizer.json");
        Path spmModel = modelDir.resolve("tokenizer.model");

        // Diffusers / Stable Diffusion compatibility: check 'tokenizer' subdirectory
        if (!Files.exists(hfConfig) && !Files.exists(spmModel)) {
            Path subDir = modelDir.resolve("tokenizer");
            if (Files.isDirectory(subDir)) {
                hfConfig = subDir.resolve("tokenizer.json");
                spmModel = subDir.resolve("tokenizer.model");
            }
        }

        if (Files.exists(hfConfig)) {
            return loadTokenizerJson(modelDir, hfConfig, false);
        }

        if (Files.exists(spmModel)) {
            throw new IOException("SentencePiece tokenizer.model requires a pure-Java tokenizer.json in this runtime: "
                    + spmModel + ". Native SentencePiece fallback has been removed.");
        }

        // Try vocab.json + merges.txt fallback (Legacy CLIP/BPE format)
        Path vocabJson = modelDir.resolve("vocab.json");
        Path mergesTxt = modelDir.resolve("merges.txt");

        // Diffusers / Stable Diffusion compatibility: check 'tokenizer' subdirectory
        if (!Files.exists(vocabJson) || !Files.exists(mergesTxt)) {
            Path subDir = modelDir.resolve("tokenizer");
            if (Files.isDirectory(subDir)) {
                vocabJson = subDir.resolve("vocab.json");
                mergesTxt = subDir.resolve("merges.txt");
            }
        }

        if (Files.exists(vocabJson) && Files.exists(mergesTxt)) {
            // Assume GPT-2/CLIP style BPE
            return HuggingFaceBpeTokenizer.load(vocabJson, mergesTxt, new Gpt2PreTokenizer(), true, false);
        }

        throw new IOException("No supported tokenizer files found in " + modelDir + " (expected tokenizer.json, tokenizer.model, or vocab.json+merges.txt)");
    }

    private static Optional<Tokenizer> tryLoadFromModelFamilyPlugin(
            Path modelDir,
            Path nativeLibPath,
            Optional<ModelFamilyTokenizerInspection> inspection) throws IOException {
        List<ModelTokenizerDescriptor> descriptors = inspection
                .map(ModelFamilyTokenizerInspection::usableDescriptors)
                .orElse(List.of());
        IOException lastFailure = null;
        for (ModelTokenizerDescriptor descriptor : descriptors) {
            try {
                return Optional.of(loadWithDescriptor(modelDir, descriptor, nativeLibPath));
            } catch (IOException e) {
                if (descriptor.isStrict()) {
                    throw e;
                }
                lastFailure = e;
            }
        }
        if (lastFailure != null) {
            throw lastFailure;
        }
        return Optional.empty();
    }

    private static String tokenizerInspectionHint(ModelFamilyTokenizerInspection inspection) {
        String problems = inspection.problemCodes().isEmpty()
                ? ""
                : " Problems: " + String.join(", ", inspection.problemCodes()) + ".";
        String remediation = inspection.remediationHints().isEmpty()
                ? ""
                : " Remediation: " + String.join(" ", inspection.remediationHints());
        return inspection.summary() + "." + problems + remediation;
    }

    private static Optional<ModelConfig> loadModelConfig(Path modelDir) {
        try {
            Path configPath = modelDir.resolve("config.json");
            if (!Files.exists(configPath)) {
                return Optional.empty();
            }
            return Optional.of(ModelConfig.load(configPath, new ObjectMapper()));
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static Tokenizer loadWithDescriptor(
            Path modelDir,
            ModelTokenizerDescriptor descriptor,
            Path nativeLibPath) throws IOException {
        return switch (descriptor.kind()) {
            case HUGGING_FACE_BPE -> loadDetected(modelDir, nativeLibPath);
            case SENTENCE_PIECE_BPE -> {
                Path tokenizerJson = firstExisting(modelDir, "tokenizer.json", "tokenizer/tokenizer.json")
                        .orElseThrow(() -> new IOException("No tokenizer.json found for " + descriptor.id()
                                + " in " + modelDir));
                yield loadTokenizerJson(modelDir, tokenizerJson, true);
            }
            case WORD_PIECE -> {
                Optional<Path> vocabTxt = firstExisting(modelDir, "vocab.txt", "tokenizer/vocab.txt");
                if (vocabTxt.isPresent()) {
                    yield WordPieceTokenizer.fromVocabFile(vocabTxt.get());
                }
                Path tokenizerJson = firstExisting(modelDir, "tokenizer.json", "tokenizer/tokenizer.json")
                        .orElseThrow(() -> new IOException("No WordPiece tokenizer files found for "
                                + descriptor.id() + " in " + modelDir));
                yield WordPieceTokenizer.fromTokenizerJsonFile(tokenizerJson);
            }
            case TIKTOKEN, CUSTOM -> throw new IOException("Tokenizer profile is advertised but not implemented in tokenizer-core: "
                    + descriptor.kind() + " (" + descriptor.id() + ")");
        };
    }

    private static Optional<Path> firstExisting(Path modelDir, String... relativePaths) {
        for (String relativePath : relativePaths) {
            Path candidate = modelDir.resolve(relativePath);
            if (Files.exists(candidate)) {
                return Optional.of(candidate);
            }
        }
        return Optional.empty();
    }

    private static Tokenizer loadTokenizerJson(Path modelDir, Path hfConfig, boolean forceSentencePieceBpe)
            throws IOException {
        boolean javaOnly = isJavaOnlyTokenizerMode();
        String modelType = detectTokenizerModelType(hfConfig);
        if (modelType != null && !modelType.equalsIgnoreCase("BPE")) {
            String suffix = forceSentencePieceBpe
                    ? " for SentencePiece-style BPE profile"
                    : " for Java-only mode";
            if (javaOnly || forceSentencePieceBpe) {
                throw new IOException("Unsupported tokenizer.json model type" + suffix + ": " + modelType
                        + " (expected BPE). Use a pure-Java tokenizer.json export with BPE model data.");
            }
            throw new IOException("Native SentencePiece fallback has been removed; use tokenizer.json with BPE model data.");
        }
        if (forceSentencePieceBpe || isSentencePieceStyle(hfConfig) || isMetaspaceNormalizedBpe(hfConfig)) {
            // Prefer full tokenizer.json execution path first.
            // FunctionGemma/Gemma3 tokenizers are SentencePiece-style BPE and
            // need robust normalizer + pre-tokenizer handling from the HF parser.
            try {
                return HuggingFaceBpeTokenizer.load(hfConfig, new SentencePiecePreTokenizer(), false, true);
            } catch (Exception hfLoadFailure) {
                // For Gemma3/FunctionGemma this fallback causes severe token drift
                // and gibberish generations. Fail fast instead of silently degrading.
                String detectedModelType = detectModelType(modelDir);
                if (forceSentencePieceBpe || detectedModelType.startsWith("gemma3")
                        || detectedModelType.startsWith("gemma4")) {
                    throw new IOException("Failed to load SentencePiece-style tokenizer.json with strict HF parser for "
                            + detectedModelType + ". Refusing degraded fallback tokenizer path.", hfLoadFailure);
                }
                // Non-Gemma paths may still use the lightweight fallback.
                return loadSentencePieceStyleBpeTokenizer(hfConfig);
            }
        }
        // Default to GPT-2 pre-tokenizer for HuggingFace BPE models
        return HuggingFaceBpeTokenizer.load(hfConfig, new Gpt2PreTokenizer());
    }

    private static String detectTokenizerModelType(Path tokenizerJson) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(tokenizerJson.toFile());
            JsonNode model = root.path("model");
            if (model.isMissingNode()) {
                return null;
            }
            String type = model.path("type").asText();
            return type == null || type.isBlank() ? null : type;
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Load a specific type of tokenizer.
     */
    public static Tokenizer load(Path path, TokenizerType type, Path nativeLibPath) throws IOException {
        return switch (type) {
            case BPE -> HuggingFaceBpeTokenizer.load(path, new Gpt2PreTokenizer());
            case SENTENCE_PIECE -> throw new UnsupportedOperationException(
                    "Native SentencePiece fallback has been removed; use tokenizer.json with the pure-Java tokenizer runtime.");
            case WORD_PIECE -> WordPieceTokenizer.fromVocabFile(path);
            default -> throw new UnsupportedOperationException("Tokenizer type not yet implemented: " + type);
        };
    }

    private static boolean isSentencePieceStyle(Path tokenizerJson) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(tokenizerJson.toFile());
            JsonNode pre = root.path("pre_tokenizer");
            JsonNode norm = root.path("normalizer");
            boolean splitOnSpace = "Split".equals(pre.path("type").asText())
                    && " ".equals(pre.path("pattern").path("String").asText());
            boolean normToUnderline = "Replace".equals(norm.path("type").asText())
                    && " ".equals(norm.path("pattern").path("String").asText())
                    && "▁".equals(norm.path("content").asText());
            return splitOnSpace && normToUnderline;
        } catch (Exception e) {
            return false;
        }
    }

    private static boolean isMetaspaceNormalizedBpe(Path tokenizerJson) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(tokenizerJson.toFile());
            if (!"BPE".equalsIgnoreCase(root.path("model").path("type").asText(""))) {
                return false;
            }
            return containsSpaceToMetaspaceNormalizer(root.path("normalizer"));
        } catch (Exception e) {
            return false;
        }
    }

    private static boolean containsSpaceToMetaspaceNormalizer(JsonNode normalizer) {
        if (normalizer == null || normalizer.isMissingNode() || normalizer.isNull()) {
            return false;
        }
        if ("Replace".equals(normalizer.path("type").asText())
                && " ".equals(normalizer.path("pattern").path("String").asText())
                && "▁".equals(normalizer.path("content").asText())) {
            return true;
        }
        JsonNode normalizers = normalizer.path("normalizers");
        if (normalizers.isArray()) {
            for (JsonNode child : normalizers) {
                if (containsSpaceToMetaspaceNormalizer(child)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean isJavaOnlyTokenizerMode() {
        String prop = System.getProperty(JAVA_ONLY_PROP);
        if (prop != null && !prop.isBlank()) {
            return Boolean.parseBoolean(prop);
        }
        String env = System.getenv(JAVA_ONLY_ENV);
        if (env != null && !env.isBlank()) {
            return Boolean.parseBoolean(env);
        }
        return true;
    }

    private static String detectModelType(Path modelDir) {
        try {
            Path configPath = modelDir.resolve("config.json");
            if (!Files.exists(configPath)) {
                return "";
            }
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(configPath.toFile());
            String type = root.path("model_type").asText("");
            return type == null ? "" : type.trim().toLowerCase();
        } catch (Exception ignored) {
            return "";
        }
    }

    /**
     * Loads SentencePiece-style HF BPE tokenizer.json using the baseline pure-Java
     * BPE engine. This avoids native SentencePiece dependency while keeping
     * deterministic tokenization behavior for Gemma-style vocabularies.
     */
    private static BpeTokenizer loadSentencePieceStyleBpeTokenizer(Path tokenizerJson) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(tokenizerJson.toFile());
        JsonNode model = root.path("model");
        JsonNode vocab = model.path("vocab");
        JsonNode merges = model.path("merges");

        if (!vocab.isObject() || !merges.isArray()) {
            throw new IOException("Invalid SentencePiece-style tokenizer.json: missing model.vocab/model.merges");
        }

        Map<String, Integer> tokenToId = new HashMap<>();
        Map<Integer, String> idToToken = new HashMap<>();
        vocab.fields().forEachRemaining(e -> {
            int id = e.getValue().asInt();
            tokenToId.put(e.getKey(), id);
            idToToken.put(id, e.getKey());
        });

        Map<String, Integer> mergeRanks = new HashMap<>();
        int idx = 0;
        for (JsonNode merge : merges) {
            String left;
            String right;
            if (merge.isTextual()) {
                String[] parts = merge.asText().split(" ", 2);
                if (parts.length != 2) {
                    idx++;
                    continue;
                }
                left = parts[0];
                right = parts[1];
            } else if (merge.isArray() && merge.size() >= 2) {
                left = merge.get(0).asText();
                right = merge.get(1).asText();
            } else {
                idx++;
                continue;
            }
            mergeRanks.put(left + " " + right, idx++);
        }

        Map<String, Integer> specials = new HashMap<>();
        JsonNode addedTokens = root.path("added_tokens");
        if (addedTokens.isArray()) {
            for (JsonNode at : addedTokens) {
                if (!at.hasNonNull("id") || !at.hasNonNull("content")) {
                    continue;
                }
                int id = at.get("id").asInt();
                String content = at.get("content").asText();
                tokenToId.put(content, id);
                idToToken.put(id, content);
                specials.put(content, id);
            }
        }

        int bos = tokenToId.getOrDefault("<bos>", -1);
        int eos = tokenToId.getOrDefault("<eos>", -1);
        int pad = tokenToId.getOrDefault("<pad>", -1);
        int unk = tokenToId.getOrDefault("<unk>", -1);

        return new BpeTokenizer(
                tokenToId,
                idToToken,
                mergeRanks,
                true,
                unk,
                bos,
                eos,
                pad,
                specials);
    }
}

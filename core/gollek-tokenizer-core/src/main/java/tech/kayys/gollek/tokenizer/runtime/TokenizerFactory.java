package tech.kayys.gollek.tokenizer.runtime;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.tokenizer.impl.BpeTokenizer;
import tech.kayys.gollek.tokenizer.impl.Gpt2PreTokenizer;
import tech.kayys.gollek.tokenizer.impl.HuggingFaceBpeTokenizer;
import tech.kayys.gollek.tokenizer.impl.SentencePiecePreTokenizer;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.TokenizerType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

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

        boolean javaOnly = isJavaOnlyTokenizerMode();
        if (Files.exists(hfConfig)) {
            String modelType = detectTokenizerModelType(hfConfig);
            if (modelType != null && !modelType.equalsIgnoreCase("BPE")) {
                if (javaOnly) {
                    throw new IOException("Unsupported tokenizer.json model type for Java-only mode: " + modelType
                            + " (expected BPE). Disable Java-only mode to allow native SentencePiece fallback.");
                }
            } else {
                if (isSentencePieceStyle(hfConfig)) {
                    // Prefer full tokenizer.json execution path first.
                    // FunctionGemma/Gemma3 tokenizers are SentencePiece-style BPE and
                    // need robust normalizer + pre-tokenizer handling from the HF parser.
                    try {
                        return HuggingFaceBpeTokenizer.load(hfConfig, new SentencePiecePreTokenizer(), false, true);
                    } catch (Exception hfLoadFailure) {
                        // For Gemma3/FunctionGemma this fallback causes severe token drift
                        // and gibberish generations. Fail fast instead of silently degrading.
                        String detectedModelType = detectModelType(modelDir);
                        if (detectedModelType.startsWith("gemma3") || detectedModelType.startsWith("gemma4")) {
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

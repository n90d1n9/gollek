/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelConfig.java
 * ─────────────────
 * Model configuration from config.json.
 */
package tech.kayys.gollek.spi.model;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import org.jboss.logging.Logger;

/**
 * Parsed HuggingFace {@code config.json} for a transformer model checkpoint.
 *
 * <p>
 * Load via {@link #load(Path, ObjectMapper)} or
 * {@link #fromDirectory(Path, ObjectMapper)}.
 *
 * <p>
 * All fields default to sensible values if absent from the JSON — this
 * handles both slim community configs and full Transformers library exports.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig {

    private static final String DISABLE_GEMMA4_SHARED_KV_PROPERTY =
            "gollek.safetensor.disable_gemma4_shared_kv";
    public ModelConfig() {
        // Defaults applied via field initializers
    }

    private static final Logger log = Logger.getLogger(ModelConfig.class);

    // ── Architecture identity ─────────────────────────────────────────────────

    /**
     * Architecture type string — maps to a registered {@link ModelArchitecture}.
     * Examples: "LlamaForCausalLM", "MistralForCausalLM", "GemmaForCausalLM",
     * "Qwen2ForCausalLM", "PhiForCausalLM", "GPTNeoXForCausalLM".
     */
    @JsonProperty("architectures")
    private List<String> architectures;

    /** Short model-type key: "llama", "mistral", "gemma", "phi", "qwen2", etc. */
    @JsonProperty("model_type")
    private String modelType;

    // ── Core dimensions ───────────────────────────────────────────────────────

    /** Embedding / residual stream dimension. */
    @JsonProperty("hidden_size")
    private int hiddenSize = 4096;

    /** Number of transformer layers. */
    @JsonProperty("num_hidden_layers")
    private int numHiddenLayers = 32;

    /** Number of attention heads (query heads). */
    @JsonProperty("num_attention_heads")
    private int numAttentionHeads = 32;

    /**
     * Number of key-value heads.
     * Equal to {@code numAttentionHeads} for MHA.
     * Less than {@code numAttentionHeads} for GQA (LLaMA-2-70B, Mistral, Gemma2).
     * 1 for MQA.
     */
    @JsonProperty("num_key_value_heads")
    private Integer numKeyValueHeads; // null → defaults to numAttentionHeads

    /** FFN intermediate dimension. Typically 4× or 2.67× hidden_size. */
    @JsonProperty("intermediate_size")
    private int intermediateSize = 11008;

    /** Vocabulary size. */
    @JsonProperty("vocab_size")
    private int vocabSize = 32000;

    /** Maximum sequence length the model was trained on. */
    @JsonProperty("max_position_embeddings")
    private int maxPositionEmbeddings = 4096;

    /** Hidden size for per-layer input pathway (Gemma-4 style). */
    @JsonProperty("hidden_size_per_layer_input")
    private int hiddenSizePerLayerInput = 0;

    /** Vocab size for per-layer input pathway (Gemma-4 style). */
    @JsonProperty("vocab_size_per_layer_input")
    private int vocabSizePerLayerInput = 0;

    // ── Normalization ─────────────────────────────────────────────────────────

    /** RMS norm epsilon (LLaMA, Gemma, Mistral). */
    @JsonProperty("rms_norm_eps")
    private double rmsNormEps = 1e-5;

    /** Layer norm epsilon (GPT-2, BERT-style). */
    @JsonProperty("layer_norm_eps")
    private double layerNormEps = 1e-5;

    // ── Rotary position embedding ─────────────────────────────────────────────

    /** RoPE base frequency (default 10000 for most models, 500000 for LLaMA-3). */
    @JsonProperty("rope_theta")
    private double ropeTheta = 10000.0;

    /** Per-dimension RoPE scaling factor (LLaMA-2 long-context). */
    @JsonProperty("rope_scaling")
    private RopeScaling ropeScaling;

    // ── Gemma-4 layer-specific RoPE ─────────────────────────────────────────
    @JsonProperty("layer_types")
    private List<String> layerTypes;

    @JsonProperty("rope_local_base_freq")
    private Double ropeLocalBaseFreq;

    private Double ropeThetaFull;
    private Double ropeThetaSliding;
    private Double partialRotaryFactorFull;
    private Double partialRotaryFactorSliding;

    @JsonProperty("query_pre_attn_scalar")
    private Double queryPreAttnScalar;

    @JsonProperty("attn_logit_softcapping")
    private Double attnLogitSoftcapping;

    @JsonProperty("final_logit_softcapping")
    private Double finalLogitSoftcapping;

    // ── Activation function ───────────────────────────────────────────────────

    /** Activation in the FFN: "silu", "gelu", "relu", "gelu_new". */
    @JsonProperty("hidden_act")
    private String hiddenAct = "silu";

    // ── Special tokens ────────────────────────────────────────────────────────

    @JsonProperty("bos_token_id")
    private Integer bosTokenId;
    @JsonProperty("eos_token_id")
    private Integer eosTokenId;
    private List<Integer> eosTokenIds = new ArrayList<>();
    @JsonProperty("pad_token_id")
    private Integer padTokenId;

    // ── Dtype ─────────────────────────────────────────────────────────────────

    /** Storage dtype declared in config: "bfloat16", "float16", "float32". */
    @JsonProperty("torch_dtype")
    private String torchDtype = "bfloat16";

    // ── Tie embeddings ────────────────────────────────────────────────────────

    /**
     * Whether {@code lm_head.weight} is tied (shared) with
     * {@code model.embed_tokens.weight}.
     */
    @JsonProperty("tie_word_embeddings")
    private boolean tieWordEmbeddings = false;

    // ── Sliding window attention (Mistral) ────────────────────────────────────

    @JsonProperty("sliding_window")
    private Integer slidingWindow; // null = full attention

    @JsonProperty("use_sliding_window")
    private Boolean useSlidingWindow;

    // ── Head dimension override (Gemma2, some Phi variants) ──────────────────

    @JsonProperty("head_dim")
    private Integer headDim; // null = hiddenSize / numAttentionHeads

    @JsonProperty("global_head_dim")
    private Integer globalHeadDim;

    @JsonProperty("num_global_key_value_heads")
    private Integer numGlobalKeyValueHeads;

    @JsonProperty("attention_k_eq_v")
    private Boolean attentionKeyEqualsValue;

    @JsonProperty("num_kv_shared_layers")
    private Integer numKvSharedLayers;


    // ── MoE fields ──────────────────────────────────────────────────────────

    @JsonProperty("num_local_experts")
    private Integer numLocalExperts;

    @JsonProperty("num_experts_per_tok")
    private Integer numExpertsPerTok;

    @JsonProperty("decoder_sparse_step")
    private Integer decoderSparseStep;

    // ─────────────────────────────────────────────────────────────────────────
    // Factory / loading
    // ─────────────────────────────────────────────────────────────────────────

    /** Parse {@code config.json} from an explicit file path. */
    public static ModelConfig load(Path configPath, ObjectMapper mapper) throws IOException {
        if (!Files.exists(configPath)) {
            throw new IOException("config.json not found: " + configPath);
        }
        ModelConfig cfg = mapper.readValue(configPath.toFile(), ModelConfig.class);
        // Fallback for multi-modal configs where root fields may be lost or malformed.
        if ((cfg.modelType == null || cfg.modelType.isBlank())
                && (cfg.architectures == null || cfg.architectures.isEmpty())) {
            try {
                JsonNode root = mapper.readTree(configPath.toFile());
                String inferredType = textValue(root, "model_type");
                List<String> inferredArch = listValue(root, "architectures");

                if (inferredType == null || inferredType.isBlank()) {
                    inferredType = textValue(root.path("text_config"), "model_type");
                }
                if (inferredArch == null || inferredArch.isEmpty()) {
                    inferredArch = listValue(root.path("text_config"), "architectures");
                }
                if ((inferredType != null && !inferredType.isBlank())
                        || (inferredArch != null && !inferredArch.isEmpty())) {
                    cfg.modelType = inferredType;
                    cfg.architectures = inferredArch;
                    log.infof("ModelConfig: inferred fields from fallback: type=%s arch=%s",
                            cfg.modelType, cfg.primaryArchitecture());
                }
            } catch (Exception e) {
                log.debugf("ModelConfig: fallback inference failed: %s", e.getMessage());
            }
        }
        // Merge text_config values when root fields are missing (Gemma4-style configs).
        try {
            JsonNode root = mapper.readTree(configPath.toFile());
            JsonNode textCfg = root.path("text_config");
            if (textCfg != null && !textCfg.isMissingNode()) {
                if (!root.has("hidden_size")) {
                    Integer v = intValue(textCfg, "hidden_size");
                    if (v != null) cfg.hiddenSize = v;
                }
                if (!root.has("num_hidden_layers")) {
                    Integer v = intValue(textCfg, "num_hidden_layers");
                    if (v != null) cfg.numHiddenLayers = v;
                }
                if (!root.has("num_attention_heads")) {
                    Integer v = intValue(textCfg, "num_attention_heads");
                    if (v != null) cfg.numAttentionHeads = v;
                }
                if (!root.has("num_key_value_heads")) {
                    Integer v = intValue(textCfg, "num_key_value_heads");
                    if (v != null) cfg.numKeyValueHeads = v;
                }
                if (!root.has("intermediate_size")) {
                    Integer v = intValue(textCfg, "intermediate_size");
                    if (v != null) cfg.intermediateSize = v;
                }
                if (!root.has("vocab_size")) {
                    Integer v = intValue(textCfg, "vocab_size");
                    if (v != null) cfg.vocabSize = v;
                }
                if (!root.has("rms_norm_eps")) {
                    Double v = doubleValue(textCfg, "rms_norm_eps");
                    if (v != null) cfg.rmsNormEps = v;
                }
                if (!root.has("rope_theta")) {
                    Double v = doubleValue(textCfg, "rope_theta");
                    if (v != null) cfg.ropeTheta = v;
                }
                if (!root.has("head_dim")) {
                    Integer v = intValue(textCfg, "head_dim");
                    if (v != null) cfg.headDim = v;
                }
                if (!root.has("global_head_dim")) {
                    Integer v = intValue(textCfg, "global_head_dim");
                    if (v != null) cfg.globalHeadDim = v;
                }
                if (!root.has("num_global_key_value_heads")) {
                    Integer v = intValue(textCfg, "num_global_key_value_heads");
                    if (v != null) cfg.numGlobalKeyValueHeads = v;
                }
                if (!root.has("attention_k_eq_v")) {
                    Boolean v = boolValue(textCfg, "attention_k_eq_v");
                    if (v != null) cfg.attentionKeyEqualsValue = v;
                }
                if (!root.has("num_kv_shared_layers")) {
                    Integer v = intValue(textCfg, "num_kv_shared_layers");
                    if (v != null) cfg.numKvSharedLayers = v;
                }

                if (!root.has("query_pre_attn_scalar")) {
                    Double v = doubleValue(textCfg, "query_pre_attn_scalar");
                    if (v != null) cfg.queryPreAttnScalar = v;
                }
                if (!root.has("attn_logit_softcapping")) {
                    Double v = doubleValue(textCfg, "attn_logit_softcapping");
                    if (v != null) cfg.attnLogitSoftcapping = v;
                }
                if (!root.has("final_logit_softcapping")) {
                    Double v = doubleValue(textCfg, "final_logit_softcapping");
                    if (v != null) cfg.finalLogitSoftcapping = v;
                }
                if (!root.has("layer_types")) {
                    JsonNode types = textCfg.get("layer_types");
                    if (types != null && types.isArray()) {
                        List<String> values = new ArrayList<>();
                        types.forEach(n -> values.add(n.asText()));
                        cfg.layerTypes = values;
                    }
                }
                if (!root.has("rope_parameters")) {
                    JsonNode ropeParams = textCfg.get("rope_parameters");
                    if (ropeParams != null && ropeParams.isObject()) {
                        mergeRopeParameters(cfg, ropeParams);
                    }
                }
                if (!root.has("rope_local_base_freq")) {
                    Double v = doubleValue(textCfg, "rope_local_base_freq");
                    if (v != null) cfg.ropeLocalBaseFreq = v;
                }
                if (!root.has("hidden_size_per_layer_input")) {
                    Integer v = intValue(textCfg, "hidden_size_per_layer_input");
                    if (v != null) cfg.hiddenSizePerLayerInput = v;
                }
                if (!root.has("vocab_size_per_layer_input")) {
                    Integer v = intValue(textCfg, "vocab_size_per_layer_input");
                    if (v != null) cfg.vocabSizePerLayerInput = v;
                }
                if (!root.has("bos_token_id")) {
                    Integer v = intValue(textCfg, "bos_token_id");
                    if (v != null) cfg.bosTokenId = v;
                }
                if (!root.has("eos_token_id")) {
                    Integer v = intValue(textCfg, "eos_token_id");
                    if (v != null) cfg.eosTokenId = v;
                }
                if (!root.has("pad_token_id")) {
                    Integer v = intValue(textCfg, "pad_token_id");
                    if (v != null) cfg.padTokenId = v;
                }
                if (!root.has("sliding_window")) {
                    Integer v = intValue(textCfg, "sliding_window");
                    if (v != null) {
                        cfg.slidingWindow = v;
                    }
                }
                if (!root.has("use_sliding_window")) {
                    Boolean v = boolValue(textCfg, "use_sliding_window");
                    if (v != null) {
                        cfg.useSlidingWindow = v;
                    }
                }
                if (!root.has("max_position_embeddings")) {
                    Integer v = intValue(textCfg, "max_position_embeddings");
                    if (v != null) {
                        cfg.maxPositionEmbeddings = v;
                    }
                }
                if (!root.has("hidden_act")) {
                    String act = textValue(textCfg, "hidden_activation");
                    if (act == null || act.isBlank()) {
                        act = textValue(textCfg, "hidden_act");
                    }
                    if (act != null && !act.isBlank()) {
                        cfg.hiddenAct = act;
                    }
                }
            }
            JsonNode rootRopeParams = root.get("rope_parameters");
            if (rootRopeParams != null && rootRopeParams.isObject()) {
                mergeRopeParameters(cfg, rootRopeParams);
            }
            String rootActivation = textValue(root, "hidden_activation");
            if (rootActivation != null && !rootActivation.isBlank()) {
                cfg.hiddenAct = rootActivation;
            }
            String dtype = textValue(root, "dtype");
            if (dtype != null && !dtype.isBlank()) {
                cfg.torchDtype = dtype;
            }
            reconcileGemma3RopeDefaults(cfg);
        } catch (Exception e) {
            log.debugf("ModelConfig: text_config merge failed: %s", e.getMessage());
        }
        log.infof("Loaded model config: type=%s arch=%s layers=%d hidden=%d heads=%d kvHeads=%d",
                cfg.modelType, cfg.primaryArchitecture(),
                cfg.numHiddenLayers, cfg.hiddenSize,
                cfg.numAttentionHeads, cfg.resolvedNumKvHeads());
        return cfg;
    }

    /**
     * Load {@code config.json} from the model directory.
     *
     * @param modelDir directory containing config.json and the safetensors weights
     */
    public static ModelConfig fromDirectory(Path modelDir, ObjectMapper mapper) throws IOException {
        return load(modelDir.resolve("config.json"), mapper);
    }

    /**
     * Build a direct-engine config from GGUF metadata.
     *
     * <p>GGUF exports store the same architectural facts as HuggingFace
     * {@code config.json}, but under architecture-prefixed keys such as
     * {@code llama.embedding_length} or {@code gemma4.attention.head_count}.
     * This mapper keeps the Java-native GGUF path from using constructor
     * defaults that accidentally describe a different model.</p>
     */
    public static ModelConfig fromGgufMetadata(Map<String, Object> metadata) {
        Map<String, Object> meta = metadata != null ? metadata : Map.of();
        String arch = metadataString(meta, "general.architecture")
                .orElse("llama")
                .trim()
                .toLowerCase(Locale.ROOT);
        if (arch.isBlank()) {
            arch = "llama";
        }

        ModelConfig cfg = new ModelConfig();
        cfg.modelType = arch;
        cfg.architectures = List.of(ggufArchitectureClassName(arch));
        cfg.maxPositionEmbeddings = metadataInt(meta, cfg.maxPositionEmbeddings,
                arch + ".context_length",
                "general.context_length");
        cfg.hiddenSize = metadataInt(meta, cfg.hiddenSize,
                arch + ".embedding_length",
                "general.embedding_length");
        cfg.numHiddenLayers = metadataInt(meta, cfg.numHiddenLayers,
                arch + ".block_count",
                "general.block_count");
        cfg.intermediateSize = metadataMaxInt(meta, cfg.intermediateSize,
                arch + ".feed_forward_length",
                "general.feed_forward_length");
        cfg.numAttentionHeads = metadataInt(meta, cfg.numAttentionHeads,
                arch + ".attention.head_count",
                "general.attention.head_count");
        cfg.numKeyValueHeads = metadataInt(meta, cfg.numAttentionHeads,
                arch + ".attention.head_count_kv",
                "general.attention.head_count_kv");
        cfg.vocabSize = metadataInt(meta,
                metadataTokenCount(meta).orElse(cfg.vocabSize),
                arch + ".vocab_size",
                "general.vocab_size");
        cfg.rmsNormEps = metadataDouble(meta, cfg.rmsNormEps,
                arch + ".attention.layer_norm_rms_epsilon",
                "general.attention.layer_norm_rms_epsilon");
        cfg.ropeTheta = metadataDouble(meta, cfg.ropeTheta,
                arch + ".rope.freq_base",
                "general.rope.freq_base");
        cfg.ropeThetaFull = cfg.ropeTheta;

        Integer localHeadDim = metadataIntOptional(meta,
                arch + ".attention.head_dim",
                arch + ".attention.key_length_swa",
                arch + ".rope.dimension_count_swa",
                "general.attention.head_dim").orElse(null);
        Integer globalHeadDim = metadataIntOptional(meta,
                arch + ".attention.key_length",
                arch + ".rope.dimension_count").orElse(null);
        if (localHeadDim == null) {
            localHeadDim = globalHeadDim != null
                    ? globalHeadDim
                    : (cfg.numAttentionHeads > 0 ? cfg.hiddenSize / cfg.numAttentionHeads : 0);
        }
        cfg.headDim = localHeadDim != null && localHeadDim > 0 ? localHeadDim : null;
        if (globalHeadDim != null && globalHeadDim > 0 && !globalHeadDim.equals(cfg.headDim)) {
            cfg.globalHeadDim = globalHeadDim;
        }

        metadataIntOptional(meta, arch + ".attention.sliding_window")
                .ifPresent(value -> {
                    cfg.slidingWindow = value;
                    cfg.useSlidingWindow = value > 0;
                });
        metadataIntOptional(meta, arch + ".attention.shared_kv_layers")
                .ifPresent(value -> cfg.numKvSharedLayers = value);
        metadataIntOptional(meta, arch + ".embedding_length_per_layer_input")
                .ifPresent(value -> cfg.hiddenSizePerLayerInput = value);
        metadataDoubleOptional(meta, arch + ".rope.freq_base_swa")
                .ifPresent(value -> {
                    cfg.ropeLocalBaseFreq = value;
                    cfg.ropeThetaSliding = value;
                });
        metadataDoubleOptional(meta, arch + ".final_logit_softcapping")
                .ifPresent(value -> cfg.finalLogitSoftcapping = value);
        metadataIntOptional(meta, "tokenizer.ggml.bos_token_id")
                .ifPresent(value -> cfg.bosTokenId = value);
        metadataIntOptional(meta, "tokenizer.ggml.eos_token_id")
                .ifPresent(value -> {
                    cfg.eosTokenId = value;
                    cfg.eosTokenIds = List.of(value);
                });
        metadataIntOptional(meta, "tokenizer.ggml.padding_token_id")
                .or(() -> metadataIntOptional(meta, "tokenizer.ggml.pad_token_id"))
                .ifPresent(value -> cfg.padTokenId = value);

        List<String> layerTypes = metadataLayerTypes(meta, arch + ".attention.sliding_window_pattern");
        if (!layerTypes.isEmpty()) {
            cfg.layerTypes = layerTypes;
        }
        return cfg;
    }

    /**
     * Check whether a model directory has a config.json.
     */
    public static boolean exists(Path modelDir) {
        return Files.exists(modelDir.resolve("config.json"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Derived / computed accessors
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Primary architecture class name (first entry in the list, or model_type
     * fallback).
     */
    public String primaryArchitecture() {
        if (architectures != null && !architectures.isEmpty())
            return architectures.get(0);
        return modelType != null ? modelType : "unknown";
    }

    /**
     * Accepts either a JSON array or single string for "architectures".
     */
    @JsonSetter("architectures")
    public void setArchitectures(JsonNode node) {
        if (node == null || node.isNull()) {
            this.architectures = null;
            return;
        }
        if (node.isArray()) {
            List<String> out = new java.util.ArrayList<>();
            for (JsonNode n : node) {
                if (n != null && !n.isNull()) {
                    out.add(n.asText());
                }
            }
            this.architectures = out;
            return;
        }
        if (node.isTextual()) {
            this.architectures = List.of(node.asText());
            return;
        }
        // Unknown shape, keep null so we can fall back to model_type.
        this.architectures = null;
    }

    /**
     * Direct setter for runtime overrides.
     */
    public void setArchitectures(List<String> architectures) {
        this.architectures = architectures;
    }

    @JsonSetter("bos_token_id")
    public void setBosTokenId(JsonNode node) {
        this.bosTokenId = firstInt(node);
    }

    @JsonSetter("eos_token_id")
    public void setEosTokenId(JsonNode node) {
        this.eosTokenIds = allInts(node);
        this.eosTokenId = eosTokenIds.isEmpty() ? null : eosTokenIds.get(0);
    }

    @JsonSetter("pad_token_id")
    public void setPadTokenId(JsonNode node) {
        this.padTokenId = firstInt(node);
    }

    @JsonSetter("hidden_activation")
    public void setHiddenActivation(String hiddenActivation) {
        if (hiddenActivation != null && !hiddenActivation.isBlank()) {
            this.hiddenAct = hiddenActivation;
        }
    }

    @JsonSetter("dtype")
    public void setDtype(String dtype) {
        if (dtype != null && !dtype.isBlank()) {
            this.torchDtype = dtype;
        }
    }

    private static void mergeRopeParameters(ModelConfig cfg, JsonNode ropeParams) {
        if (cfg == null || ropeParams == null || !ropeParams.isObject()) {
            return;
        }
        JsonNode full = ropeParams.get("full_attention");
        if (full != null && full.isObject()) {
            Double v = doubleValue(full, "rope_theta");
            if (v != null) cfg.ropeThetaFull = v;
            Double p = doubleValue(full, "partial_rotary_factor");
            if (p != null) cfg.partialRotaryFactorFull = p;
        }
        JsonNode sliding = ropeParams.get("sliding_attention");
        if (sliding != null && sliding.isObject()) {
            Double v = doubleValue(sliding, "rope_theta");
            if (v != null) cfg.ropeThetaSliding = v;
            Double p = doubleValue(sliding, "partial_rotary_factor");
            if (p != null) cfg.partialRotaryFactorSliding = p;
        }
    }

    private static void reconcileGemma3RopeDefaults(ModelConfig cfg) {
        if (cfg == null || cfg.modelType == null || !cfg.modelType.toLowerCase().startsWith("gemma3")) {
            return;
        }
        if (cfg.ropeThetaFull == null) {
            cfg.ropeThetaFull = cfg.ropeTheta;
        }
        if (cfg.ropeThetaSliding == null) {
            cfg.ropeThetaSliding = cfg.ropeLocalBaseFreq != null ? cfg.ropeLocalBaseFreq : 10000.0;
        }
    }

    private static String textValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isTextual() ? v.asText() : v.toString();
    }

    private static List<String> listValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        if (v.isArray()) {
            List<String> out = new java.util.ArrayList<>();
            for (JsonNode n : v) {
                if (n != null && !n.isNull()) {
                    out.add(n.asText());
                }
            }
            return out;
        }
        if (v.isTextual()) {
            return List.of(v.asText());
        }
        return null;
    }

    private static Integer firstInt(JsonNode node) {
        if (node == null || node.isNull()) {
            return null;
        }
        if (node.isNumber()) {
            return node.asInt();
        }
        if (node.isArray() && node.size() > 0) {
            JsonNode first = node.get(0);
            return first != null && first.isNumber() ? first.asInt() : null;
        }
        return null;
    }

    private static Integer intValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asInt() : null;
    }

    private static Double doubleValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asDouble() : null;
    }

    private static Boolean boolValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isBoolean() ? v.asBoolean() : null;
    }

    private static List<Integer> allInts(JsonNode node) {
        if (node == null || node.isNull()) {
            return List.of();
        }
        List<Integer> values = new ArrayList<>();
        if (node.isArray()) {
            for (JsonNode item : node) {
                if (item != null && item.isNumber()) {
                    values.add(item.asInt());
                }
            }
            return values;
        }
        if (node.isNumber()) {
            values.add(node.asInt());
        }
        return values;
    }

    private static String ggufArchitectureClassName(String arch) {
        String normalized = arch == null ? "" : arch.toLowerCase(Locale.ROOT);
        if (normalized.contains("gemma4")) {
            return "Gemma4ForConditionalGeneration";
        }
        if (normalized.contains("gemma3")) {
            return "Gemma3ForCausalLM";
        }
        if (normalized.contains("gemma2")) {
            return "Gemma2ForCausalLM";
        }
        if (normalized.contains("gemma")) {
            return "GemmaForCausalLM";
        }
        if (normalized.contains("qwen3")) {
            return "Qwen3ForCausalLM";
        }
        if (normalized.contains("qwen")) {
            return "Qwen2ForCausalLM";
        }
        if (normalized.contains("mixtral")) {
            return "MixtralForCausalLM";
        }
        if (normalized.contains("mistral")) {
            return "MistralForCausalLM";
        }
        if (normalized.contains("phi3")) {
            return "Phi3ForCausalLM";
        }
        if (normalized.contains("phi")) {
            return "PhiForCausalLM";
        }
        return "LlamaForCausalLM";
    }

    private static Optional<String> metadataString(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return Optional.empty();
        }
        String text = value.toString().trim();
        return text.isEmpty() ? Optional.empty() : Optional.of(text);
    }

    private static int metadataInt(Map<String, Object> metadata, int fallback, String... keys) {
        return metadataIntOptional(metadata, keys).orElse(fallback);
    }

    private static Optional<Integer> metadataIntOptional(Map<String, Object> metadata, String... keys) {
        for (String key : keys) {
            Optional<Integer> value = firstNumericInt(metadata.get(key));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }

    private static int metadataMaxInt(Map<String, Object> metadata, int fallback, String... keys) {
        for (String key : keys) {
            Optional<Integer> value = maxNumericInt(metadata.get(key));
            if (value.isPresent()) {
                return value.get();
            }
        }
        return fallback;
    }

    private static double metadataDouble(Map<String, Object> metadata, double fallback, String... keys) {
        return metadataDoubleOptional(metadata, keys).orElse(fallback);
    }

    private static Optional<Double> metadataDoubleOptional(Map<String, Object> metadata, String... keys) {
        for (String key : keys) {
            Optional<Double> value = firstNumericDouble(metadata.get(key));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }

    private static Optional<Integer> metadataTokenCount(Map<String, Object> metadata) {
        Object tokens = metadata.get("tokenizer.ggml.tokens");
        if (tokens instanceof List<?> list) {
            return list.isEmpty() ? Optional.empty() : Optional.of(list.size());
        }
        if (tokens != null && tokens.getClass().isArray()) {
            return Optional.of(java.lang.reflect.Array.getLength(tokens));
        }
        return Optional.empty();
    }

    private static Optional<Integer> firstNumericInt(Object value) {
        if (value instanceof Number number) {
            return Optional.of(number.intValue());
        }
        if (value instanceof List<?> list) {
            for (Object item : list) {
                Optional<Integer> numeric = firstNumericInt(item);
                if (numeric.isPresent()) {
                    return numeric;
                }
            }
        }
        return Optional.empty();
    }

    private static Optional<Integer> maxNumericInt(Object value) {
        if (value instanceof Number number) {
            return Optional.of(number.intValue());
        }
        if (value instanceof List<?> list) {
            Integer max = null;
            for (Object item : list) {
                Optional<Integer> numeric = firstNumericInt(item);
                if (numeric.isPresent()) {
                    max = max == null ? numeric.get() : Math.max(max, numeric.get());
                }
            }
            return Optional.ofNullable(max);
        }
        return Optional.empty();
    }

    private static Optional<Double> firstNumericDouble(Object value) {
        if (value instanceof Number number) {
            return Optional.of(number.doubleValue());
        }
        if (value instanceof List<?> list) {
            for (Object item : list) {
                Optional<Double> numeric = firstNumericDouble(item);
                if (numeric.isPresent()) {
                    return numeric;
                }
            }
        }
        return Optional.empty();
    }

    private static List<String> metadataLayerTypes(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        if (!(value instanceof List<?> list) || list.isEmpty()) {
            return List.of();
        }
        List<String> out = new ArrayList<>(list.size());
        for (Object item : list) {
            if (item instanceof Boolean sliding) {
                out.add(sliding ? "sliding_attention" : "full_attention");
            }
        }
        return out.size() == list.size() ? List.copyOf(out) : List.of();
    }

    /** Resolved number of KV heads — defaults to numAttentionHeads if not set. */
    public int resolvedNumKvHeads() {
        return numKeyValueHeads != null ? numKeyValueHeads : numAttentionHeads;
    }

    public int resolvedNumGlobalKvHeads() {
        return numGlobalKeyValueHeads != null ? numGlobalKeyValueHeads : resolvedNumKvHeads();
    }

    public int resolvedNumKvHeadsForLayer(int layerIdx) {
        return usesAlternativeAttentionForLayer(layerIdx) ? resolvedNumGlobalKvHeads() : resolvedNumKvHeads();
    }

    public int resolvedMaxKvHeads() {
        return Math.max(resolvedNumKvHeads(), resolvedNumGlobalKvHeads());
    }

    /** Per-head dimension: {@code hiddenSize / numAttentionHeads}. */
    public int resolvedHeadDim() {
        return headDim != null ? headDim : (numAttentionHeads > 0 ? hiddenSize / numAttentionHeads : 0);
    }

    /** Maximum head dimension across all layers (for heterogeneous architectures like Gemma 4). */
    public int resolvedMaxHeadDim() {
        int base = resolvedHeadDim();
        return globalHeadDim != null ? Math.max(base, globalHeadDim) : base;
    }

    public int resolvedNumKvSharedLayers() {
        return numKvSharedLayers != null ? Math.max(0, numKvSharedLayers) : 0;
    }

    public boolean usesSharedKvCache(int layerIdx) {
        String modelTypeLower = modelType != null ? modelType.toLowerCase() : "";
        if (modelTypeLower.startsWith("gemma4")
                && Boolean.getBoolean(DISABLE_GEMMA4_SHARED_KV_PROPERTY)) {
            return false;
        }
        int sharedLayers = resolvedNumKvSharedLayers();
        if (sharedLayers <= 0 || layerIdx < 0 || layerIdx >= numHiddenLayers) {
            return false;
        }
        return layerIdx >= Math.max(0, numHiddenLayers - sharedLayers);
    }

    public int sharedKvSourceLayer(int layerIdx) {
        if (!usesSharedKvCache(layerIdx)) {
            return layerIdx;
        }
        int sharedStart = Math.max(0, numHiddenLayers - resolvedNumKvSharedLayers());
        String type = layerType(layerIdx);
        for (int prev = sharedStart - 1; prev >= 0; prev--) {
            if (type == null || type.equals(layerType(prev))) {
                return prev;
            }
        }
        return Math.max(0, sharedStart - 1);
    }

    public boolean isSharedKvSourceLayer(int layerIdx) {
        if (layerIdx < 0 || layerIdx >= numHiddenLayers) {
            return false;
        }
        int sharedStart = Math.max(0, numHiddenLayers - resolvedNumKvSharedLayers());
        if (sharedStart <= 0 || layerIdx >= sharedStart) {
            return false;
        }
        String type = layerType(layerIdx);
        for (int sharedLayer = sharedStart; sharedLayer < numHiddenLayers; sharedLayer++) {
            if ((type == null || type.equals(layerType(sharedLayer)))
                    && sharedKvSourceLayer(sharedLayer) == layerIdx) {
                return true;
            }
        }
        return false;
    }

    /**
     * GQA group size: number of query heads sharing one KV head.
     * 1 = MHA, N = GQA/MQA.
     */
    public int kvGroupSize() {
        return numAttentionHeads / resolvedNumKvHeads();
    }

    /** Whether this model uses Grouped-Query Attention (GQA or MQA). */
    public boolean isGroupedQueryAttention() {
        return resolvedNumKvHeads() < numAttentionHeads;
    }

    public boolean attentionKeyEqualsValue() {
        return Boolean.TRUE.equals(attentionKeyEqualsValue);
    }

    public boolean usesAlternativeAttentionForLayer(int layerIdx) {
        return attentionKeyEqualsValue() && !isSlidingAttentionLayer(layerIdx);
    }

    public Optional<Integer> bosTokenId() {
        return Optional.ofNullable(bosTokenId);
    }

    public Optional<Integer> eosTokenId() {
        return Optional.ofNullable(eosTokenId);
    }

    public List<Integer> eosTokenIds() {
        if (eosTokenIds == null || eosTokenIds.isEmpty()) {
            return eosTokenId != null ? List.of(eosTokenId) : List.of();
        }
        return List.copyOf(eosTokenIds);
    }

    public Optional<Integer> padTokenId() {
        return Optional.ofNullable(padTokenId);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Overrides (runtime reconciliation)
    // ─────────────────────────────────────────────────────────────────────────

    /** Override the number of attention heads when config.json is inconsistent with weights. */
    public void overrideNumAttentionHeads(int heads) {
        if (heads > 0) {
            this.numAttentionHeads = heads;
        }
    }

    /** Override the number of key-value heads when config.json is inconsistent with weights. */
    public void overrideNumKeyValueHeads(Integer heads) {
        if (heads != null && heads > 0) {
            this.numKeyValueHeads = heads;
        }
    }

    /** Override head dimension when config.json is inconsistent with weights. */
    public void overrideHeadDim(Integer dim) {
        if (dim != null && dim > 0) {
            this.headDim = dim;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Plain getters (used by JSON deserialisation and compute kernels)
    // ─────────────────────────────────────────────────────────────────────────

    public String modelType() {
        return modelType;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public int numHiddenLayers() {
        return numHiddenLayers;
    }

    public int numAttentionHeads() {
        return numAttentionHeads;
    }

    public int intermediateSize() {
        return intermediateSize;
    }

    public int vocabSize() {
        return vocabSize;
    }

    public int maxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }

    public int hiddenSizePerLayerInput() {
        return hiddenSizePerLayerInput;
    }

    public int vocabSizePerLayerInput() {
        return vocabSizePerLayerInput;
    }

    public String layerType(int layerIdx) {
        if (layerTypes == null || layerIdx < 0 || layerIdx >= layerTypes.size()) {
            return null;
        }
        return layerTypes.get(layerIdx);
    }

    public double ropeThetaForLayer(int layerIdx) {
        String type = layerType(layerIdx);
        if ("full_attention".equals(type) && ropeThetaFull != null) {
            return ropeThetaFull;
        }
        if ("sliding_attention".equals(type) && ropeThetaSliding != null) {
            return ropeThetaSliding;
        }
        return ropeTheta;
    }

    public double partialRotaryFactorForLayer(int layerIdx) {
        String type = layerType(layerIdx);
        if ("full_attention".equals(type) && partialRotaryFactorFull != null) {
            return partialRotaryFactorFull;
        }
        if ("sliding_attention".equals(type) && partialRotaryFactorSliding != null) {
            return partialRotaryFactorSliding;
        }
        return 1.0;
    }

    public boolean isSlidingAttentionLayer(int layerIdx) {
        return "sliding_attention".equals(layerType(layerIdx));
    }

    public Double finalLogitSoftcapping() {
        return finalLogitSoftcapping;
    }

    public Double attnLogitSoftcapping() {
        return attnLogitSoftcapping;
    }

    public double queryPreAttnScalar() {
        return queryPreAttnScalar != null && queryPreAttnScalar > 0
                ? queryPreAttnScalar
                : Math.max(1, resolvedHeadDim());
    }

    public double rmsNormEps() {
        return rmsNormEps;
    }

    public double layerNormEps() {
        return layerNormEps;
    }

    public double ropeTheta() {
        return ropeTheta;
    }

    public RopeScaling ropeScaling() {
        return ropeScaling;
    }

    public String hiddenAct() {
        return hiddenAct;
    }

    public String torchDtype() {
        return torchDtype;
    }

    public boolean tieWordEmbeddings() {
        return tieWordEmbeddings;
    }

    public List<String> architectures() {
        return architectures;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Nested types
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * RoPE scaling configuration (used for long-context variants).
     * <p>
     * Types: "linear", "dynamic", "yarn".
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class RopeScaling {
        @JsonProperty("type")
        public String type;
        @JsonProperty("factor")
        public double factor = 1.0;
        @JsonProperty("low_freq_factor")
        public double lowFreqFactor = 1.0;
        @JsonProperty("high_freq_factor")
        public double highFreqFactor = 4.0;
        @JsonProperty("original_max_position_embeddings")
        public int originalMaxPositionEmbeddings = 8192;
    }

    @Override
    public String toString() {
        return "ModelConfig{type=" + modelType
                + ", arch=" + primaryArchitecture()
                + ", layers=" + numHiddenLayers
                + ", hidden=" + hiddenSize
                + ", heads=" + numAttentionHeads + "/" + resolvedNumKvHeads()
                + ", vocab=" + vocabSize
                + '}';
    }

    /**
     * Check if model has sliding window attention.
     */
    public boolean hasSlidingWindow() {
        return !Boolean.FALSE.equals(useSlidingWindow)
                && slidingWindow != null
                && slidingWindow > 0;
    }

    /**
     * Get sliding window size.
     */
    public int slidingWindowSize() {
        return slidingWindow != null ? slidingWindow : Integer.MAX_VALUE;
    }

    /**
     * Check if model is MoE (Mixture of Experts).
     */
    public boolean isMoe() {
        return numLocalExperts != null && numLocalExperts > 1;
    }

    /**
     * Get number of local experts (for MoE models).
     */
    public int numLocalExperts() {
        return numLocalExperts != null ? numLocalExperts : 0;
    }

    /**
     * Get number of experts per token (for MoE models).
     */
    public int numExpertsPerTok() {
        return numExpertsPerTok != null ? numExpertsPerTok : 0;
    }

    /**
     * Check if layer is MoE layer (for interleaved architectures).
     */
    public boolean isMoeLayer(int layerIdx) {
        if (!isMoe())
            return false;
        int step = decoderSparseStep != null ? decoderSparseStep : 1;
        return (layerIdx + 1) % step == 0;
    }

    /**
     * Get number of key-value heads (GQA).
     */
    public int numKeyValueHeads() {
        return numKeyValueHeads;
    }
}

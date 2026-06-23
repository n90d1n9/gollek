package tech.kayys.gollek.spi.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.ArrayList;
import java.util.List;

/**
 * Parsed HuggingFace {@code config.json} for a transformer model checkpoint.
 * This is a pure data class with no business logic.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig {

    // ── Architecture identity ─────────────────────────────────────────────────
    @JsonProperty("architectures")
    private List<String> architectures;

    @JsonProperty("model_type")
    private String modelType;

    // ── Core dimensions ───────────────────────────────────────────────────────
    @JsonProperty("hidden_size")
    private int hiddenSize = 4096;

    @JsonProperty("num_hidden_layers")
    private int numHiddenLayers = 32;

    @JsonProperty("num_attention_heads")
    private int numAttentionHeads = 32;

    @JsonProperty("num_key_value_heads")
    private Integer numKeyValueHeads;

    @JsonProperty("intermediate_size")
    private int intermediateSize = 11008;

    @JsonProperty("vocab_size")
    private int vocabSize = 32000;

    @JsonProperty("max_position_embeddings")
    private int maxPositionEmbeddings = 4096;

    @JsonProperty("hidden_size_per_layer_input")
    private int hiddenSizePerLayerInput = 0;

    @JsonProperty("vocab_size_per_layer_input")
    private int vocabSizePerLayerInput = 0;

    // ── Normalization ─────────────────────────────────────────────────────────
    @JsonProperty("rms_norm_eps")
    private double rmsNormEps = 1e-5;

    @JsonProperty("layer_norm_eps")
    private double layerNormEps = 1e-5;

    // ── Rotary position embedding ─────────────────────────────────────────────
    @JsonProperty("rope_theta")
    private double ropeTheta = 10000.0;

    @JsonProperty("rope_scaling")
    private RopeScaling ropeScaling;

    @JsonProperty("layer_types")
    private List<String> layerTypes;

    @JsonProperty("rope_local_base_freq")
    private Double ropeLocalBaseFreq;

    private Double ropeThetaFull;
    private Double ropeThetaSliding;

    @JsonProperty("partial_rotary_factor")
    private Double partialRotaryFactor;

    private Double partialRotaryFactorFull;
    private Double partialRotaryFactorSliding;

    @JsonProperty("query_pre_attn_scalar")
    private Double queryPreAttnScalar;

    @JsonProperty("attn_logit_softcapping")
    private Double attnLogitSoftcapping;

    @JsonProperty("final_logit_softcapping")
    private Double finalLogitSoftcapping;

    // ── Activation function ───────────────────────────────────────────────────
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
    @JsonProperty("torch_dtype")
    private String torchDtype = "bfloat16";

    // ── Tie embeddings ────────────────────────────────────────────────────────
    @JsonProperty("tie_word_embeddings")
    private boolean tieWordEmbeddings = false;

    // ── Sliding window attention ──────────────────────────────────────────────
    @JsonProperty("sliding_window")
    private Integer slidingWindow;

    @JsonProperty("use_sliding_window")
    private Boolean useSlidingWindow;

    // ── Head dimension override ──────────────────────────────────────────────
    @JsonProperty("head_dim")
    private Integer headDim;

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

    @JsonProperty("enable_moe_block")
    private Boolean enableMoeBlock;

    @JsonProperty("moe_intermediate_size")
    private Integer moeIntermediateSize;

    @JsonProperty("use_double_wide_mlp")
    private Boolean useDoubleWideMlp;

    // ── Vision ────────────────────────────────────────────────────────────────
    @JsonProperty("vision_config")
    private VisionConfig visionConfig;

    @JsonProperty("vision_soft_tokens_per_image")
    private Integer visionSoftTokensPerImage;

    // ── Constructors ──────────────────────────────────────────────────────────
    public ModelConfig() {
        // Defaults applied via field initializers
    }

    // ── Custom setters with @JsonSetter for flexible deserialization ──────

    @JsonSetter("architectures")
    public void setArchitectures(JsonNode node) {
        if (node == null || node.isNull()) {
            this.architectures = null;
            return;
        }
        if (node.isArray()) {
            List<String> out = new ArrayList<>();
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
        this.architectures = null;
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

    @JsonSetter("activation")
    public void setActivation(String activation) {
        if (activation != null && !activation.isBlank()) {
            this.hiddenAct = activation;
        }
    }

    @JsonSetter("num_kv_heads")
    public void setNumKvHeads(Integer numKvHeads) {
        if (numKvHeads != null) {
            this.numKeyValueHeads = numKvHeads;
        }
    }

    @JsonSetter("ffn_hidden_size")
    public void setFfnHiddenSize(Integer ffnHiddenSize) {
        if (ffnHiddenSize != null) {
            this.intermediateSize = ffnHiddenSize;
        }
    }

    @JsonSetter("num_experts")
    public void setNumExperts(Integer numExperts) {
        if (numExperts != null) {
            this.numLocalExperts = numExperts;
        }
    }

    @JsonSetter("top_k_experts")
    public void setTopKExperts(Integer topKExperts) {
        if (topKExperts != null) {
            this.numExpertsPerTok = topKExperts;
        }
    }

    @JsonSetter("layer_norm_epsilon")
    public void setLayerNormEpsilon(Double layerNormEpsilon) {
        if (layerNormEpsilon != null) {
            this.layerNormEps = layerNormEpsilon;
        }
    }

    @JsonSetter("dtype")
    public void setDtype(String dtype) {
        if (dtype != null && !dtype.isBlank()) {
            this.torchDtype = dtype;
        }
    }

    // ── Helper methods for JSON deserialization ─────────────────────────────

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

    // ── Getters and Setters ──────────────────────────────────────────────────

    public List<String> getArchitectures() {
        return architectures;
    }

    public void setArchitectures(List<String> architectures) {
        this.architectures = architectures;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }

    public int getNumHiddenLayers() {
        return numHiddenLayers;
    }

    public void setNumHiddenLayers(int numHiddenLayers) {
        this.numHiddenLayers = numHiddenLayers;
    }

    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }

    public void setNumAttentionHeads(int numAttentionHeads) {
        this.numAttentionHeads = numAttentionHeads;
    }

    public Integer getNumKeyValueHeads() {
        return numKeyValueHeads;
    }

    public void setNumKeyValueHeads(Integer numKeyValueHeads) {
        this.numKeyValueHeads = numKeyValueHeads;
    }

    public int getIntermediateSize() {
        return intermediateSize;
    }

    public void setIntermediateSize(int intermediateSize) {
        this.intermediateSize = intermediateSize;
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    public int getMaxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }

    public void setMaxPositionEmbeddings(int maxPositionEmbeddings) {
        this.maxPositionEmbeddings = maxPositionEmbeddings;
    }

    public int getHiddenSizePerLayerInput() {
        return hiddenSizePerLayerInput;
    }

    public void setHiddenSizePerLayerInput(int hiddenSizePerLayerInput) {
        this.hiddenSizePerLayerInput = hiddenSizePerLayerInput;
    }

    public int getVocabSizePerLayerInput() {
        return vocabSizePerLayerInput;
    }

    public void setVocabSizePerLayerInput(int vocabSizePerLayerInput) {
        this.vocabSizePerLayerInput = vocabSizePerLayerInput;
    }

    public double getRmsNormEps() {
        return rmsNormEps;
    }

    public void setRmsNormEps(double rmsNormEps) {
        this.rmsNormEps = rmsNormEps;
    }

    public double getLayerNormEps() {
        return layerNormEps;
    }

    public void setLayerNormEps(double layerNormEps) {
        this.layerNormEps = layerNormEps;
    }

    public double getRopeTheta() {
        return ropeTheta;
    }

    public void setRopeTheta(double ropeTheta) {
        this.ropeTheta = ropeTheta;
    }

    public RopeScaling getRopeScaling() {
        return ropeScaling;
    }

    public void setRopeScaling(RopeScaling ropeScaling) {
        this.ropeScaling = ropeScaling;
    }

    public List<String> getLayerTypes() {
        return layerTypes;
    }

    public void setLayerTypes(List<String> layerTypes) {
        this.layerTypes = layerTypes;
    }

    public Double getRopeLocalBaseFreq() {
        return ropeLocalBaseFreq;
    }

    public void setRopeLocalBaseFreq(Double ropeLocalBaseFreq) {
        this.ropeLocalBaseFreq = ropeLocalBaseFreq;
    }

    public Double getRopeThetaFull() {
        return ropeThetaFull;
    }

    public void setRopeThetaFull(Double ropeThetaFull) {
        this.ropeThetaFull = ropeThetaFull;
    }

    public Double getRopeThetaSliding() {
        return ropeThetaSliding;
    }

    public void setRopeThetaSliding(Double ropeThetaSliding) {
        this.ropeThetaSliding = ropeThetaSliding;
    }

    public Double getPartialRotaryFactor() {
        return partialRotaryFactor;
    }

    public void setPartialRotaryFactor(Double partialRotaryFactor) {
        this.partialRotaryFactor = partialRotaryFactor;
    }

    public Double getPartialRotaryFactorFull() {
        return partialRotaryFactorFull;
    }

    public void setPartialRotaryFactorFull(Double partialRotaryFactorFull) {
        this.partialRotaryFactorFull = partialRotaryFactorFull;
    }

    public Double getPartialRotaryFactorSliding() {
        return partialRotaryFactorSliding;
    }

    public void setPartialRotaryFactorSliding(Double partialRotaryFactorSliding) {
        this.partialRotaryFactorSliding = partialRotaryFactorSliding;
    }

    public Double getQueryPreAttnScalar() {
        return queryPreAttnScalar;
    }

    public void setQueryPreAttnScalar(Double queryPreAttnScalar) {
        this.queryPreAttnScalar = queryPreAttnScalar;
    }

    public Double getAttnLogitSoftcapping() {
        return attnLogitSoftcapping;
    }

    public void setAttnLogitSoftcapping(Double attnLogitSoftcapping) {
        this.attnLogitSoftcapping = attnLogitSoftcapping;
    }

    public Double getFinalLogitSoftcapping() {
        return finalLogitSoftcapping;
    }

    public void setFinalLogitSoftcapping(Double finalLogitSoftcapping) {
        this.finalLogitSoftcapping = finalLogitSoftcapping;
    }

    public String getHiddenAct() {
        return hiddenAct;
    }

    public void setHiddenAct(String hiddenAct) {
        this.hiddenAct = hiddenAct;
    }

    public Integer getBosTokenId() {
        return bosTokenId;
    }

    public void setBosTokenId(Integer bosTokenId) {
        this.bosTokenId = bosTokenId;
    }

    public Integer getEosTokenId() {
        return eosTokenId;
    }

    public void setEosTokenId(Integer eosTokenId) {
        this.eosTokenId = eosTokenId;
    }

    public List<Integer> getEosTokenIds() {
        return eosTokenIds;
    }

    public void setEosTokenIds(List<Integer> eosTokenIds) {
        this.eosTokenIds = eosTokenIds;
    }

    public Integer getPadTokenId() {
        return padTokenId;
    }

    public void setPadTokenId(Integer padTokenId) {
        this.padTokenId = padTokenId;
    }

    public String getTorchDtype() {
        return torchDtype;
    }

    public void setTorchDtype(String torchDtype) {
        this.torchDtype = torchDtype;
    }

    public boolean isTieWordEmbeddings() {
        return tieWordEmbeddings;
    }

    public void setTieWordEmbeddings(boolean tieWordEmbeddings) {
        this.tieWordEmbeddings = tieWordEmbeddings;
    }

    public Integer getSlidingWindow() {
        return slidingWindow;
    }

    public void setSlidingWindow(Integer slidingWindow) {
        this.slidingWindow = slidingWindow;
    }

    public Boolean getUseSlidingWindow() {
        return useSlidingWindow;
    }

    public void setUseSlidingWindow(Boolean useSlidingWindow) {
        this.useSlidingWindow = useSlidingWindow;
    }

    public Integer getHeadDim() {
        return headDim;
    }

    public void setHeadDim(Integer headDim) {
        this.headDim = headDim;
    }

    public Integer getGlobalHeadDim() {
        return globalHeadDim;
    }

    public void setGlobalHeadDim(Integer globalHeadDim) {
        this.globalHeadDim = globalHeadDim;
    }

    public Integer getNumGlobalKeyValueHeads() {
        return numGlobalKeyValueHeads;
    }

    public void setNumGlobalKeyValueHeads(Integer numGlobalKeyValueHeads) {
        this.numGlobalKeyValueHeads = numGlobalKeyValueHeads;
    }

    public Boolean getAttentionKeyEqualsValue() {
        return attentionKeyEqualsValue;
    }

    public void setAttentionKeyEqualsValue(Boolean attentionKeyEqualsValue) {
        this.attentionKeyEqualsValue = attentionKeyEqualsValue;
    }

    public Integer getNumKvSharedLayers() {
        return numKvSharedLayers;
    }

    public void setNumKvSharedLayers(Integer numKvSharedLayers) {
        this.numKvSharedLayers = numKvSharedLayers;
    }

    public Integer getNumLocalExperts() {
        return numLocalExperts;
    }

    public void setNumLocalExperts(Integer numLocalExperts) {
        this.numLocalExperts = numLocalExperts;
    }

    public Integer getNumExpertsPerTok() {
        return numExpertsPerTok;
    }

    public void setNumExpertsPerTok(Integer numExpertsPerTok) {
        this.numExpertsPerTok = numExpertsPerTok;
    }

    public Integer getDecoderSparseStep() {
        return decoderSparseStep;
    }

    public void setDecoderSparseStep(Integer decoderSparseStep) {
        this.decoderSparseStep = decoderSparseStep;
    }

    public Boolean getEnableMoeBlock() {
        return enableMoeBlock;
    }

    public void setEnableMoeBlock(Boolean enableMoeBlock) {
        this.enableMoeBlock = enableMoeBlock;
    }

    public Integer getMoeIntermediateSize() {
        return moeIntermediateSize;
    }

    public void setMoeIntermediateSize(Integer moeIntermediateSize) {
        this.moeIntermediateSize = moeIntermediateSize;
    }

    public Boolean getUseDoubleWideMlp() {
        return useDoubleWideMlp;
    }

    public void setUseDoubleWideMlp(Boolean useDoubleWideMlp) {
        this.useDoubleWideMlp = useDoubleWideMlp;
    }

    public VisionConfig getVisionConfig() {
        return visionConfig;
    }

    public void setVisionConfig(VisionConfig visionConfig) {
        this.visionConfig = visionConfig;
    }

    public Integer getVisionSoftTokensPerImage() {
        return visionSoftTokensPerImage;
    }

    public void setVisionSoftTokensPerImage(Integer visionSoftTokensPerImage) {
        this.visionSoftTokensPerImage = visionSoftTokensPerImage;
    }

    // ── Computed property methods ────────────────────────────────────────────

    public String getPrimaryArchitecture() {
        if (architectures != null && !architectures.isEmpty()) {
            return architectures.get(0);
        }
        return modelType != null ? modelType : "unknown";
    }

    public int getResolvedNumKvHeads() {
        return numKeyValueHeads != null ? numKeyValueHeads : numAttentionHeads;
    }

    public int getResolvedNumGlobalKvHeads() {
        return numGlobalKeyValueHeads != null ? numGlobalKeyValueHeads : getResolvedNumKvHeads();
    }

    public int getResolvedHeadDim() {
        return headDim != null ? headDim : 
               (numAttentionHeads > 0 ? hiddenSize / numAttentionHeads : 0);
    }

    public int getResolvedMaxHeadDim() {
        int base = getResolvedHeadDim();
        return globalHeadDim != null ? Math.max(base, globalHeadDim) : base;
    }

    public int getResolvedNumKvSharedLayers() {
        return numKvSharedLayers != null ? Math.max(0, numKvSharedLayers) : 0;
    }

    public int getKvGroupSize() {
        return numAttentionHeads / getResolvedNumKvHeads();
    }

    public boolean isGroupedQueryAttention() {
        return getResolvedNumKvHeads() < numAttentionHeads;
    }

    public boolean attentionKeyEqualsValue() {
        return Boolean.TRUE.equals(attentionKeyEqualsValue);
    }

    public boolean hasSlidingWindow() {
        return !Boolean.FALSE.equals(useSlidingWindow)
                && slidingWindow != null
                && slidingWindow > 0;
    }

    public int getSlidingWindowSize() {
        return slidingWindow != null ? slidingWindow : Integer.MAX_VALUE;
    }

    public boolean isMoe() {
        return numLocalExperts != null && numLocalExperts > 1;
    }

    public int getNumLocalExperts() {
        return numLocalExperts != null ? numLocalExperts : 0;
    }

    public int getNumExpertsPerTok() {
        return numExpertsPerTok != null ? numExpertsPerTok : 0;
    }

    public boolean enableMoeBlock() {
        return Boolean.TRUE.equals(enableMoeBlock);
    }

    public int getMoeIntermediateSize() {
        return moeIntermediateSize != null ? moeIntermediateSize : 0;
    }

    public boolean usesDoubleWideMlp() {
        return Boolean.TRUE.equals(useDoubleWideMlp);
    }

    public boolean isGemma4PackedMoe() {
        String type = modelType == null ? "" : modelType.toLowerCase();
        String arch = getPrimaryArchitecture() == null ? "" : getPrimaryArchitecture().toLowerCase();
        boolean gemma4 = type.startsWith("gemma4") || arch.contains("gemma4");
        return gemma4
                && (enableMoeBlock()
                        || getMoeIntermediateSize() > 0
                        || (getNumLocalExperts() > 1 && getNumExpertsPerTok() > 0));
    }

    public boolean requiresGemma4PackedMoeRuntime() {
        return isGemma4PackedMoe();
    }

    public String getLayerType(int layerIdx) {
        if (layerTypes == null || layerIdx < 0 || layerIdx >= layerTypes.size()) {
            return null;
        }
        return layerTypes.get(layerIdx);
    }

    public double getRopeThetaForLayer(int layerIdx) {
        String type = getLayerType(layerIdx);
        if ("full_attention".equals(type) && ropeThetaFull != null) {
            return ropeThetaFull;
        }
        if ("sliding_attention".equals(type) && ropeThetaSliding != null) {
            return ropeThetaSliding;
        }
        return ropeTheta;
    }

    public double getPartialRotaryFactorForLayer(int layerIdx) {
        String type = getLayerType(layerIdx);
        if ("full_attention".equals(type) && partialRotaryFactorFull != null) {
            return partialRotaryFactorFull;
        }
        if ("sliding_attention".equals(type) && partialRotaryFactorSliding != null) {
            return partialRotaryFactorSliding;
        }
        if (partialRotaryFactor != null && partialRotaryFactor > 0.0) {
            return partialRotaryFactor;
        }
        return 1.0;
    }

    public boolean isSlidingAttentionLayer(int layerIdx) {
        return "sliding_attention".equals(getLayerType(layerIdx));
    }

    public boolean isMoeLayer(int layerIdx) {
        if (!isMoe()) return false;
        int step = decoderSparseStep != null ? decoderSparseStep : 1;
        return (layerIdx + 1) % step == 0;
    }

    public int getResolvedNumKvHeadsForLayer(int layerIdx) {
        return usesAlternativeAttentionForLayer(layerIdx) 
            ? getResolvedNumGlobalKvHeads() 
            : getResolvedNumKvHeads();
    }

    public int getResolvedMaxKvHeads() {
        return Math.max(getResolvedNumKvHeads(), getResolvedNumGlobalKvHeads());
    }

    public int getResolvedHeadDimForLayer(int layerIdx) {
        return usesAlternativeAttentionForLayer(layerIdx) && globalHeadDim != null
                ? globalHeadDim
                : getResolvedHeadDim();
    }

    public boolean usesAlternativeAttentionForLayer(int layerIdx) {
        return attentionKeyEqualsValue() && !isSlidingAttentionLayer(layerIdx);
    }

    public boolean usesSharedKvCache(int layerIdx) {
        String modelTypeLower = modelType != null ? modelType.toLowerCase() : "";
        if (modelTypeLower.startsWith("gemma4")
                && Boolean.getBoolean("gollek.safetensor.disable_gemma4_shared_kv")) {
            return false;
        }
        int sharedLayers = getResolvedNumKvSharedLayers();
        if (sharedLayers <= 0 || layerIdx < 0 || layerIdx >= numHiddenLayers) {
            return false;
        }
        return layerIdx >= Math.max(0, numHiddenLayers - sharedLayers);
    }

    public int getSharedKvSourceLayer(int layerIdx) {
        if (!usesSharedKvCache(layerIdx)) {
            return layerIdx;
        }
        int sharedStart = Math.max(0, numHiddenLayers - getResolvedNumKvSharedLayers());
        String type = getLayerType(layerIdx);
        for (int prev = sharedStart - 1; prev >= 0; prev--) {
            if (type == null || type.equals(getLayerType(prev))) {
                return prev;
            }
        }
        return Math.max(0, sharedStart - 1);
    }

    public boolean isSharedKvSourceLayer(int layerIdx) {
        if (layerIdx < 0 || layerIdx >= numHiddenLayers) {
            return false;
        }
        int sharedStart = Math.max(0, numHiddenLayers - getResolvedNumKvSharedLayers());
        if (sharedStart <= 0 || layerIdx >= sharedStart) {
            return false;
        }
        String type = getLayerType(layerIdx);
        for (int sharedLayer = sharedStart; sharedLayer < numHiddenLayers; sharedLayer++) {
            if ((type == null || type.equals(getLayerType(sharedLayer)))
                    && getSharedKvSourceLayer(sharedLayer) == layerIdx) {
                return true;
            }
        }
        return false;
    }

    public double getQueryPreAttnScalar() {
        return queryPreAttnScalar != null && queryPreAttnScalar > 0
                ? queryPreAttnScalar
                : Math.max(1, getResolvedHeadDim());
    }

    public int getVisionSoftTokensPerImage() {
        return visionSoftTokensPerImage != null ? visionSoftTokensPerImage : 280;
    }

    // ── Nested classes ────────────────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class RopeScaling {
        @JsonProperty("type")
        private String type;
        
        @JsonProperty("factor")
        private double factor = 1.0;
        
        @JsonProperty("low_freq_factor")
        private double lowFreqFactor = 1.0;
        
        @JsonProperty("high_freq_factor")
        private double highFreqFactor = 4.0;
        
        @JsonProperty("original_max_position_embeddings")
        private int originalMaxPositionEmbeddings = 8192;

        public String getType() {
            return type;
        }

        public void setType(String type) {
            this.type = type;
        }

        public double getFactor() {
            return factor;
        }

        public void setFactor(double factor) {
            this.factor = factor;
        }

        public double getLowFreqFactor() {
            return lowFreqFactor;
        }

        public void setLowFreqFactor(double lowFreqFactor) {
            this.lowFreqFactor = lowFreqFactor;
        }

        public double getHighFreqFactor() {
            return highFreqFactor;
        }

        public void setHighFreqFactor(double highFreqFactor) {
            this.highFreqFactor = highFreqFactor;
        }

        public int getOriginalMaxPositionEmbeddings() {
            return originalMaxPositionEmbeddings;
        }

        public void setOriginalMaxPositionEmbeddings(int originalMaxPositionEmbeddings) {
            this.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class VisionConfig {
        @JsonProperty("hidden_size")
        private int hiddenSize = 768;
        
        @JsonProperty("intermediate_size")
        private int intermediateSize = 3072;
        
        @JsonProperty("num_attention_heads")
        private int numAttentionHeads = 12;
        
        @JsonProperty("num_hidden_layers")
        private int numHiddenLayers = 16;
        
        @JsonProperty("patch_size")
        private int patchSize = 16;
        
        @JsonProperty("head_dim")
        private int headDim = 64;

        public int getHiddenSize() {
            return hiddenSize;
        }

        public void setHiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
        }

        public int getIntermediateSize() {
            return intermediateSize;
        }

        public void setIntermediateSize(int intermediateSize) {
            this.intermediateSize = intermediateSize;
        }

        public int getNumAttentionHeads() {
            return numAttentionHeads;
        }

        public void setNumAttentionHeads(int numAttentionHeads) {
            this.numAttentionHeads = numAttentionHeads;
        }

        public int getNumHiddenLayers() {
            return numHiddenLayers;
        }

        public void setNumHiddenLayers(int numHiddenLayers) {
            this.numHiddenLayers = numHiddenLayers;
        }

        public int getPatchSize() {
            return patchSize;
        }

        public void setPatchSize(int patchSize) {
            this.patchSize = patchSize;
        }

        public int getHeadDim() {
            return headDim;
        }

        public void setHeadDim(int headDim) {
            this.headDim = headDim;
        }
    }

    // ── Override methods ─────────────────────────────────────────────────────

    /**
     * Override the number of attention heads when config.json is inconsistent with weights.
     */
    public void overrideNumAttentionHeads(int heads) {
        if (heads > 0) {
            this.numAttentionHeads = heads;
        }
    }

    /**
     * Override hidden size when config.json is unavailable or reconciled from weights.
     */
    public void overrideHiddenSize(int size) {
        if (size > 0) {
            this.hiddenSize = size;
        }
    }

    /**
     * Override layer count when config.json is unavailable or reconciled from weights.
     */
    public void overrideNumHiddenLayers(int layers) {
        if (layers > 0) {
            this.numHiddenLayers = layers;
        }
    }

    /**
     * Override FFN intermediate size when config.json is unavailable or reconciled from weights.
     */
    public void overrideIntermediateSize(int size) {
        if (size > 0) {
            this.intermediateSize = size;
        }
    }

    /**
     * Override the number of key-value heads when config.json is inconsistent with weights.
     */
    public void overrideNumKeyValueHeads(Integer heads) {
        if (heads != null && heads > 0) {
            this.numKeyValueHeads = heads;
        }
    }

    /**
     * Override head dimension when config.json is inconsistent with weights.
     */
    public void overrideHeadDim(Integer dim) {
        if (dim != null && dim > 0) {
            this.headDim = dim;
        }
    }

    @Override
    public String toString() {
        return "ModelConfig{type=" + modelType
                + ", arch=" + getPrimaryArchitecture()
                + ", layers=" + numHiddenLayers
                + ", hidden=" + hiddenSize
                + ", heads=" + numAttentionHeads + "/" + getResolvedNumKvHeads()
                + ", vocab=" + vocabSize
                + '}';
    }
}
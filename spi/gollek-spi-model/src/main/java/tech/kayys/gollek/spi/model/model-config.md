
## 1. Core Data Class (with all Jackson annotations)
```java
// ModelConfig.java
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
```

## 2. Architecture Mapper (with all 800+ mappings)
```java
// ArchitectureMapper.java
package tech.kayys.gollek.spi.model.mapper;

import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Maps GGUF architecture strings to HuggingFace class names.
 * This extracts the 800+ line mapping from the original god class.
 */
public class ArchitectureMapper {
    
    /**
     * Maps GGUF architecture string to HuggingFace class name.
     */
    public String mapGgufToHfClassName(String arch) {
        String normalized = arch == null ? "" : arch.toLowerCase(Locale.ROOT);
        
        // Gemma family
        if (normalized.contains("shieldgemma2") || normalized.contains("shield_gemma2")
                || normalized.contains("shield-gemma2")) {
            return "ShieldGemma2ForImageClassification";
        }
        if (normalized.contains("t5gemma2") || normalized.contains("t5_gemma2")
                || normalized.contains("t5-gemma2")) {
            return "T5Gemma2ForConditionalGeneration";
        }
        if (normalized.contains("t5gemma") || normalized.contains("t5_gemma")
                || normalized.contains("t5-gemma")) {
            return "T5GemmaForConditionalGeneration";
        }
        if (normalized.contains("vaultgemma") || normalized.contains("vault_gemma")
                || normalized.contains("vault-gemma")) {
            return "VaultGemmaForCausalLM";
        }
        if (normalized.contains("gemma4_unified") || normalized.contains("gemma4-unified")
                || normalized.contains("gemma-4-unified")) {
            return "Gemma4ForMultimodalLM";
        }
        if (normalized.contains("gemma4_audio") || normalized.contains("gemma4-audio")
                || normalized.contains("gemma-4-audio")) {
            return "Gemma4AudioModel";
        }
        if (normalized.contains("gemma4_vision") || normalized.contains("gemma4-vision")
                || normalized.contains("gemma-4-vision")) {
            return "Gemma4VisionModel";
        }
        if (normalized.contains("gemma4_text") || normalized.contains("gemma4-text")
                || normalized.contains("gemma-4-text")) {
            return "Gemma4ForCausalLM";
        }
        if (normalized.contains("gemma4") || normalized.contains("gemma-4")) {
            return "Gemma4ForConditionalGeneration";
        }
        if (normalized.contains("gemma3n")) {
            return "Gemma3nForConditionalGeneration";
        }
        if (normalized.contains("paligemma") || normalized.contains("pali_gemma")) {
            return "PaliGemmaForConditionalGeneration";
        }
        if (normalized.contains("recurrent_gemma") || normalized.contains("recurrent-gemma")) {
            return "RecurrentGemmaForCausalLM";
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
        
        // Qwen family
        if (normalized.contains("colqwen2")) {
            return "ColQwen2ForRetrieval";
        }
        if (normalized.contains("qwen3_omni_moe") || normalized.contains("qwen3-omni-moe")) {
            return "Qwen3OmniMoeForConditionalGeneration";
        }
        if (normalized.contains("qwen3_5_moe") || normalized.contains("qwen3.5_moe")
                || normalized.contains("qwen3-5-moe") || normalized.contains("qwen3.5-moe")) {
            return "Qwen3_5MoeForConditionalGeneration";
        }
        if (normalized.contains("qwen3_5") || normalized.contains("qwen3.5")
                || normalized.contains("qwen3-5")) {
            return "Qwen3_5ForConditionalGeneration";
        }
        if (normalized.contains("qwen3_vl_moe") || normalized.contains("qwen3-vl-moe")) {
            return "Qwen3VLMoeForConditionalGeneration";
        }
        if (normalized.contains("qwen3_vl") || normalized.contains("qwen3-vl")) {
            return "Qwen3VLForConditionalGeneration";
        }
        if (normalized.contains("qwen3_moe") || normalized.contains("qwen3-moe")) {
            return "Qwen3MoeForCausalLM";
        }
        if (normalized.contains("qwen3_next") || normalized.contains("qwen3-next")) {
            return "Qwen3NextForCausalLM";
        }
        if (normalized.contains("qwen3")) {
            return "Qwen3ForCausalLM";
        }
        if (normalized.contains("qwen2_5_omni") || normalized.contains("qwen2.5_omni")
                || normalized.contains("qwen2-5-omni") || normalized.contains("qwen2.5-omni")) {
            return "Qwen2_5OmniForConditionalGeneration";
        }
        if (normalized.contains("qwen2_5_vl") || normalized.contains("qwen2.5_vl")
                || normalized.contains("qwen2-5-vl") || normalized.contains("qwen2.5-vl")) {
            return "Qwen2_5_VLForConditionalGeneration";
        }
        if (normalized.contains("qwen2_vl") || normalized.contains("qwen2-vl")) {
            return "Qwen2VLForConditionalGeneration";
        }
        if (normalized.contains("qwen2_audio") || normalized.contains("qwen2-audio")) {
            return "Qwen2AudioForConditionalGeneration";
        }
        if (normalized.contains("qwen2_moe") || normalized.contains("qwen2-moe")) {
            return "Qwen2MoeForCausalLM";
        }
        if (normalized.contains("qwen")) {
            return "Qwen2ForCausalLM";
        }
        
        // Mistral family
        if (normalized.contains("ministral3") || normalized.contains("ministral_3")
                || normalized.contains("ministral-3")) {
            return "Ministral3ForCausalLM";
        }
        if (normalized.contains("ministral")) {
            return "MinistralForCausalLM";
        }
        if (normalized.contains("mixtral")) {
            return "MixtralForCausalLM";
        }
        if (normalized.contains("mistral4") || normalized.contains("mistral_4")
                || normalized.contains("mistral-4")) {
            return "Mistral4ForCausalLM";
        }
        if (normalized.contains("mistral3") || normalized.contains("mistral_3")
                || normalized.contains("mistral-3")) {
            return "Mistral3ForConditionalGeneration";
        }
        if (normalized.contains("mistral")) {
            return "MistralForCausalLM";
        }
        
        // Phi family
        if (normalized.contains("phi3")) {
            return "Phi3ForCausalLM";
        }
        if (normalized.contains("phi4_multimodal") || normalized.contains("phi4-multimodal")
                || normalized.contains("phi_4_multimodal") || normalized.contains("phi-4-multimodal")) {
            return "Phi4MultimodalForCausalLM";
        }
        if (normalized.contains("phimoe") || normalized.contains("phi_moe")
                || normalized.contains("phi-moe")) {
            return "PhimoeForCausalLM";
        }
        if (normalized.contains("phi")) {
            return "PhiForCausalLM";
        }
        
        // Falcon family
        if (normalized.contains("falcon_h1") || normalized.contains("falcon-h1")) {
            return "FalconH1ForCausalLM";
        }
        if (normalized.contains("falcon_mamba") || normalized.contains("falcon-mamba")) {
            return "FalconMambaForCausalLM";
        }
        if (normalized.contains("falcon")) {
            return "FalconForCausalLM";
        }
        
        // Llama family
        if (normalized.contains("llama4") || normalized.contains("llama-4")) {
            return "Llama4ForConditionalGeneration";
        }
        if (normalized.contains("mllama")) {
            return "MllamaForConditionalGeneration";
        }
        if (normalized.contains("diffllama") || normalized.contains("diff_llama")
                || normalized.contains("diff-llama")) {
            return "DiffLlamaForCausalLM";
        }
        if (normalized.contains("code_llama") || normalized.contains("code-llama")
                || normalized.contains("codellama")) {
            return "LlamaForCausalLM";
        }
        if (normalized.contains("llama")) {
            return "LlamaForCausalLM";
        }
        
        // DeepSeek family
        if (normalized.contains("deepseek_vl_hybrid") || normalized.contains("deepseek-vl-hybrid")) {
            return "DeepseekVLHybridForConditionalGeneration";
        }
        if (normalized.contains("deepseek_vl") || normalized.contains("deepseek-vl")) {
            return "DeepseekVLForConditionalGeneration";
        }
        if (normalized.contains("deepseek_v2") || normalized.contains("deepseek-v2")) {
            return "DeepseekV2ForCausalLM";
        }
        if (normalized.contains("deepseek_v3") || normalized.contains("deepseek-v3")
                || normalized.contains("deepseek_moe") || normalized.contains("deepseek-moe")
                || normalized.contains("deepseek_r1") || normalized.contains("deepseek-r1")
                || normalized.contains("deepseek")) {
            return "DeepseekV3ForCausalLM";
        }
        
        // Other popular models
        if (normalized.contains("cohere")) {
            return "Cohere2ForCausalLM";
        }
        if (normalized.contains("kimi")) {
            return "KimiForCausalLM";
        }
        if (normalized.equals("yi") || normalized.contains("yi_")) {
            return "YiForCausalLM";
        }
        if (normalized.contains("bloom")) {
            return "BloomForCausalLM";
        }
        if (normalized.contains("bitnet")) {
            return "BitNetForCausalLM";
        }
        if (normalized.contains("dbrx")) {
            return "DbrxForCausalLM";
        }
        if (normalized.contains("exaone_moe") || normalized.contains("exaone-moe")) {
            return "ExaoneMoeForCausalLM";
        }
        if (normalized.contains("exaone4")) {
            return "Exaone4ForCausalLM";
        }
        if (normalized.contains("bamba")) {
            return "BambaForCausalLM";
        }
        if (normalized.contains("zamba2")) {
            return "Zamba2ForCausalLM";
        }
        if (normalized.contains("zamba")) {
            return "ZambaForCausalLM";
        }
        if (normalized.contains("arcee")) {
            return "ArceeForCausalLM";
        }
        
        // Granite family
        if (normalized.contains("granitemoehybrid")) {
            return "GraniteMoeHybridForCausalLM";
        }
        if (normalized.contains("granitemoeshared")) {
            return "GraniteMoeSharedForCausalLM";
        }
        if (normalized.contains("granitemoe")) {
            return "GraniteMoeForCausalLM";
        }
        if (normalized.contains("granite")) {
            return "GraniteForCausalLM";
        }
        
        if (normalized.contains("nemotron")) {
            return "NemotronForCausalLM";
        }
        if (normalized.contains("stablelm") || normalized.contains("stable_lm")) {
            return "StableLmForCausalLM";
        }
        if (normalized.contains("persimmon")) {
            return "PersimmonForCausalLM";
        }
        if (normalized.contains("smollm3") || normalized.contains("smol_lm3")
                || normalized.contains("smol-lm3")) {
            return "SmolLM3ForCausalLM";
        }
        if (normalized.contains("xglm")) {
            return "XGLMForCausalLM";
        }
        
        // GLM family
        if (normalized.contains("glm_ocr_vision") || normalized.contains("glm-ocr-vision")) {
            return "GlmOcrVisionModel";
        }
        if (normalized.contains("glm_ocr_text") || normalized.contains("glm-ocr-text")) {
            return "GlmOcrTextModel";
        }
        if (normalized.contains("glm_ocr") || normalized.contains("glm-ocr")) {
            return "GlmOcrForConditionalGeneration";
        }
        if (normalized.equals("glm") || normalized.contains("glm_") || normalized.contains("glm-")) {
            return "GlmForCausalLM";
        }
        
        // Code models
        if (normalized.contains("starcoder2")) {
            return "Starcoder2ForCausalLM";
        }
        if (normalized.contains("gpt_bigcode")) {
            return "GPTBigCodeForCausalLM";
        }
        if (normalized.contains("gpt_oss") || normalized.contains("gpt-oss")) {
            return "GptOssForCausalLM";
        }
        if (normalized.contains("codegen")) {
            return "CodeGenForCausalLM";
        }
        
        // GPT family
        if (normalized.contains("gpt_neox_japanese") || normalized.contains("gpt-neox-japanese")
                || normalized.contains("gptneox_japanese")) {
            return "GPTNeoXJapaneseForCausalLM";
        }
        if (normalized.contains("gpt_neox") || normalized.contains("gptneox")) {
            return "GPTNeoXForCausalLM";
        }
        if (normalized.contains("gptj") || normalized.contains("gpt_j")) {
            return "GPTJForCausalLM";
        }
        if (normalized.contains("gpt_neo") || normalized.contains("gpt-neo")
                || normalized.contains("gptneo")) {
            return "GPTNeoForCausalLM";
        }
        if (normalized.contains("openai_gpt") || normalized.contains("openai-gpt")) {
            return "OpenAIGPTLMHeadModel";
        }
        if (normalized.contains("dialogpt") || normalized.contains("megatron_gpt2")
                || normalized.contains("megatron-gpt2") || normalized.contains("gpt2")) {
            return "GPT2LMHeadModel";
        }
        if (normalized.contains("mpt")) {
            return "MptForCausalLM";
        }
        
        // Mamba family
        if (normalized.contains("mamba2")) {
            return "Mamba2ForCausalLM";
        }
        if (normalized.contains("mamba")) {
            return "MambaForCausalLM";
        }
        if (normalized.contains("rwkv")) {
            return "RwkvForCausalLM";
        }
        if (normalized.contains("jamba")) {
            return "JambaForCausalLM";
        }
        
        // Olmo family
        if (normalized.contains("olmo_hybrid") || normalized.contains("olmo-hybrid")) {
            return "OlmoHybridForCausalLM";
        }
        if (normalized.contains("olmoe") || normalized.contains("olmo_moe")
                || normalized.contains("olmo-moe")) {
            return "OlmoeForCausalLM";
        }
        if (normalized.contains("olmo3")) {
            return "Olmo3ForCausalLM";
        }
        if (normalized.contains("olmo2")) {
            return "Olmo2ForCausalLM";
        }
        if (normalized.contains("olmo")) {
            return "OlmoForCausalLM";
        }
        
        // T5/BART family
        if (normalized.contains("plbart")) {
            return "PLBartForConditionalGeneration";
        }
        if (normalized.contains("mbart")) {
            return "MBartForConditionalGeneration";
        }
        if (normalized.contains("barthez") || normalized.contains("bartpho")) {
            return "BartForConditionalGeneration";
        }
        if (normalized.contains("blenderbot_small") || normalized.contains("blenderbot-small")) {
            return "BlenderbotSmallForConditionalGeneration";
        }
        if (normalized.contains("blenderbot")) {
            return "BlenderbotForConditionalGeneration";
        }
        if (normalized.contains("bart")) {
            return "BartForConditionalGeneration";
        }
        if (normalized.contains("marian")) {
            return "MarianMTModel";
        }
        if (normalized.contains("fsmt")) {
            return "FSMTForConditionalGeneration";
        }
        if (normalized.contains("m2m_100") || normalized.contains("m2m100")) {
            return "M2M100ForConditionalGeneration";
        }
        if (normalized.contains("nllb-moe") || normalized.contains("nllb_moe")) {
            return "NllbMoeForConditionalGeneration";
        }
        if (normalized.contains("nllb")) {
            return "M2M100ForConditionalGeneration";
        }
        if (normalized.contains("bigbird_pegasus") || normalized.contains("bigbird-pegasus")) {
            return "BigBirdPegasusForConditionalGeneration";
        }
        if (normalized.contains("big_bird") || normalized.contains("bigbird")) {
            return "BigBirdModel";
        }
        if (normalized.contains("pegasus_x") || normalized.contains("pegasus-x")) {
            return "PegasusXForConditionalGeneration";
        }
        if (normalized.contains("pegasus")) {
            return "PegasusForConditionalGeneration";
        }
        if (normalized.contains("prophetnet") || normalized.contains("prophet_net")) {
            return "ProphetNetForConditionalGeneration";
        }
        if (normalized.contains("switch_transformers") || normalized.contains("switch-transformers")) {
            return "SwitchTransformersForConditionalGeneration";
        }
        if (normalized.contains("longt5")) {
            return "LongT5ForConditionalGeneration";
        }
        if (normalized.contains("umt5")) {
            return "UMT5ForConditionalGeneration";
        }
        if (normalized.contains("mt5")) {
            return "MT5ForConditionalGeneration";
        }
        if (normalized.contains("byt5") || normalized.contains("t5")) {
            return "T5ForConditionalGeneration";
        }
        
        // Vision models
        if (normalized.contains("pixtral")) {
            return "PixtralVisionModel";
        }
        if (normalized.contains("fuyu")) {
            return "FuyuForCausalLM";
        }
        if (normalized.contains("kosmos2_5") || normalized.contains("kosmos2.5")
                || normalized.contains("kosmos-2.5") || normalized.contains("kosmos_2_5")) {
            return "Kosmos2_5ForConditionalGeneration";
        }
        if (normalized.contains("kosmos-2") || normalized.contains("kosmos2")
                || normalized.contains("kosmos_2")) {
            return "Kosmos2ForConditionalGeneration";
        }
        if (normalized.contains("instructblipvideo")) {
            return "InstructBlipVideoForConditionalGeneration";
        }
        if (normalized.contains("instructblip")) {
            return "InstructBlipForConditionalGeneration";
        }
        if (normalized.contains("video_llava") || normalized.contains("video-llava")) {
            return "VideoLlavaForConditionalGeneration";
        }
        if (normalized.contains("vipllava") || normalized.contains("vip_llava")) {
            return "VipLlavaForConditionalGeneration";
        }
        if (normalized.contains("llava_next_video") || normalized.contains("llava-next-video")) {
            return "LlavaNextVideoForConditionalGeneration";
        }
        if (normalized.contains("llava_next") || normalized.contains("llava-next")) {
            return "LlavaNextForConditionalGeneration";
        }
        if (normalized.contains("llava_onevision") || normalized.contains("llava-onevision")) {
            return "LlavaOnevisionForConditionalGeneration";
        }
        if (normalized.contains("llava")) {
            return "LlavaForConditionalGeneration";
        }
        
        // CLIP and vision encoders
        if (normalized.contains("florence2") || normalized.contains("florence_vision")) {
            return "Florence2ForConditionalGeneration";
        }
        if (normalized.contains("bridgetower")) {
            return "BridgeTowerModel";
        }
        if (normalized.contains("align_vision_model") || normalized.contains("align-vision-model")) {
            return "AlignVisionModel";
        }
        if (normalized.contains("align_text_model") || normalized.contains("align-text-model")) {
            return "AlignTextModel";
        }
        if (normalized.equals("align") || normalized.contains("align_")
                || normalized.contains("align-")) {
            return "AlignModel";
        }
        if (normalized.contains("visual_bert") || normalized.contains("visual-bert")) {
            return "VisualBertModel";
        }
        if (normalized.contains("lxmert")) {
            return "LxmertModel";
        }
        if (normalized.contains("vilt")) {
            return "ViltModel";
        }
        if (normalized.contains("siglip2")) {
            return "Siglip2Model";
        }
        if (normalized.contains("siglip")) {
            return "SiglipModel";
        }
        if (normalized.contains("clip")) {
            return "CLIPModel";
        }
        
        // Layout models
        if (normalized.contains("layoutxlm")) {
            return "LayoutLMv2Model";
        }
        if (normalized.contains("layoutlmv3")) {
            return "LayoutLMv3Model";
        }
        if (normalized.contains("layoutlmv2")) {
            return "LayoutLMv2Model";
        }
        if (normalized.contains("layoutlm")) {
            return "LayoutLMModel";
        }
        if (normalized.contains("markuplm") || normalized.contains("markup_lm")) {
            return "MarkupLMModel";
        }
        if (normalized.equals("bros") || normalized.contains("bros_") || normalized.contains("bros-")) {
            return "BrosModel";
        }
        if (normalized.equals("lilt") || normalized.contains("lilt_") || normalized.contains("lilt-")) {
            return "LiltModel";
        }
        
        // OCR models
        if (normalized.contains("mgp_str") || normalized.contains("mgp-str")) {
            return "MgpstrForSceneTextRecognition";
        }
        if (normalized.contains("trocr")) {
            return "TrOCRForCausalLM";
        }
        if (normalized.contains("pix2struct")) {
            return "Pix2StructForConditionalGeneration";
        }
        if (normalized.contains("nougat")) {
            return "VisionEncoderDecoderModel";
        }
        if (normalized.contains("lighton_ocr") || normalized.contains("lighton-ocr")) {
            return "LightOnOcrForConditionalGeneration";
        }
        if (normalized.contains("got_ocr2") || normalized.contains("got-ocr2")) {
            return "GotOcr2ForConditionalGeneration";
        }
        
        // Speech models
        if (normalized.contains("speecht5") || normalized.contains("speech_t5")) {
            return "SpeechT5Model";
        }
        if (normalized.contains("bark")) {
            return "BarkModel";
        }
        if (normalized.contains("musicgen_decoder")) {
            return "MusicgenForCausalLM";
        }
        if (normalized.contains("musicgen")) {
            return "MusicgenForConditionalGeneration";
        }
        if (normalized.contains("seamless_m4t_v2") || normalized.contains("seamless-m4t-v2")) {
            return "SeamlessM4Tv2Model";
        }
        if (normalized.contains("seamless_m4t") || normalized.contains("seamless-m4t")) {
            return "SeamlessM4TModel";
        }
        if (normalized.contains("wav2vec2_conformer") || normalized.contains("wav2vec2-conformer")) {
            return "Wav2Vec2ConformerModel";
        }
        if (normalized.contains("wav2vec2")) {
            return "Wav2Vec2Model";
        }
        if (normalized.contains("hubert")) {
            return "HubertModel";
        }
        if (normalized.contains("wavlm")) {
            return "WavLMModel";
        }
        if (normalized.contains("encodec")) {
            return "EncodecModel";
        }
        if (normalized.contains("clap")) {
            return "ClapModel";
        }
        if (normalized.contains("whisper")) {
            return "WhisperForConditionalGeneration";
        }
        
        // RAG and DPR
        if (normalized.contains("rag")) {
            return "RagModel";
        }
        if (normalized.contains("dpr")) {
            return "DPRQuestionEncoder";
        }
        
        // Vision-language models
        if (normalized.contains("video_llama_3") || normalized.contains("video-llama-3")) {
            return "VideoLlama3ForConditionalGeneration";
        }
        if (normalized.contains("internvl")) {
            return "InternVLForConditionalGeneration";
        }
        if (normalized.contains("smolvlm") || normalized.contains("smol_vlm")) {
            return "SmolVLMForConditionalGeneration";
        }
        if (normalized.contains("fast_vlm") || normalized.contains("fast-vlm")) {
            return "FastVlmForConditionalGeneration";
        }
        if (normalized.contains("idefics3")) {
            return "Idefics3ForConditionalGeneration";
        }
        if (normalized.contains("idefics2")) {
            return "Idefics2ForConditionalGeneration";
        }
        if (normalized.contains("idefics")) {
            return "IdeficsForVisionText2Text";
        }
        if (normalized.contains("chameleon")) {
            return "ChameleonForConditionalGeneration";
        }
        
        // Retrieval models
        if (normalized.contains("colmodernvbert")) {
            return "ColModernVBertForRetrieval";
        }
        if (normalized.contains("colpali")) {
            return "ColPaliForRetrieval";
        }
        
        // BLIP family
        if (normalized.contains("blip_2") || normalized.contains("blip-2") || normalized.contains("blip2")) {
            return "Blip2ForConditionalGeneration";
        }
        if (normalized.contains("blip")) {
            return "BlipForConditionalGeneration";
        }
        
        // SAM models
        if (normalized.contains("sam_hq") || normalized.contains("sam-hq")) {
            return "SamHQModel";
        }
        if (normalized.contains("sam2")) {
            return "Sam2Model";
        }
        if (normalized.equals("sam") || normalized.contains("segment_anything")) {
            return "SamModel";
        }
        if (normalized.contains("clipseg")) {
            return "CLIPSegForImageSegmentation";
        }
        if (normalized.contains("metaclip_2") || normalized.contains("metaclip-2")) {
            return "MetaClip2Model";
        }
        
        // Group and multi-modal
        if (normalized.contains("vision_text_dual_encoder")
                || normalized.contains("vision-text-dual-encoder")) {
            return "VisionTextDualEncoderModel";
        }
        if (normalized.contains("x_clip") || normalized.contains("x-clip")
                || normalized.contains("xclip")) {
            return "XCLIPModel";
        }
        
        // Chinese CLIP
        if (normalized.contains("altclip_vision_model") || normalized.contains("altclip-vision-model")) {
            return "AltCLIPVisionModel";
        }
        if (normalized.contains("altclip_text_model") || normalized.contains("altclip-text-model")) {
            return "AltCLIPTextModel";
        }
        if (normalized.contains("altclip")) {
            return "AltCLIPModel";
        }
        if (normalized.contains("chinese_clip_vision_model")
                || normalized.contains("chinese-clip-vision-model")) {
            return "ChineseCLIPVisionModel";
        }
        if (normalized.contains("chinese_clip_text_model")
                || normalized.contains("chinese-clip-text-model")) {
            return "ChineseCLIPTextModel";
        }
        if (normalized.contains("chinese_clip") || normalized.contains("chinese-clip")) {
            return "ChineseCLIPModel";
        }
        
        // GroupViT
        if (normalized.contains("groupvit_vision_model")
                || normalized.contains("groupvit-vision-model")) {
            return "GroupViTVisionModel";
        }
        if (normalized.contains("groupvit_text_model") || normalized.contains("groupvit-text-model")) {
            return "GroupViTTextModel";
        }
        if (normalized.contains("groupvit")) {
            return "GroupViTModel";
        }
        
        // Object detection
        if (normalized.contains("owlv2") || normalized.contains("owl_v2")
                || normalized.contains("owl-v2")) {
            return "Owlv2ForObjectDetection";
        }
        if (normalized.contains("owlvit") || normalized.contains("owl_vit")
                || normalized.contains("owl-vit")) {
            return "OwlViTForObjectDetection";
        }
        if (normalized.contains("mm_grounding_dino") || normalized.contains("mm-grounding-dino")) {
            return "MMGroundingDinoForObjectDetection";
        }
        if (normalized.contains("grounding_dino") || normalized.contains("grounding-dino")) {
            return "GroundingDinoForObjectDetection";
        }
        if (normalized.contains("deformable_detr") || normalized.contains("deformable-detr")) {
            return "DeformableDetrModel";
        }
        if (normalized.contains("conditional_detr") || normalized.contains("conditional-detr")) {
            return "ConditionalDetrModel";
        }
        if (normalized.contains("rt_detr") || normalized.contains("rt-detr")) {
            return "RTDetrModel";
        }
        if (normalized.contains("detr")) {
            return "DetrModel";
        }
        if (normalized.contains("yolos")) {
            return "YolosForObjectDetection";
        }
        
        // Segmentation
        if (normalized.contains("mask2former") || normalized.contains("mask2-former")) {
            return "Mask2FormerForUniversalSegmentation";
        }
        if (normalized.contains("maskformer")) {
            return "MaskFormerForInstanceSegmentation";
        }
        if (normalized.contains("oneformer")) {
            return "OneFormerForUniversalSegmentation";
        }
        if (normalized.contains("upernet")) {
            return "UperNetForSemanticSegmentation";
        }
        
        // Depth estimation
        if (normalized.contains("depth_anything") || normalized.contains("depth-anything")) {
            return "DepthAnythingForDepthEstimation";
        }
        if (normalized.contains("depth_pro") || normalized.contains("depth-pro")) {
            return "DepthProForDepthEstimation";
        }
        if (normalized.contains("zoedepth") || normalized.contains("zoe_depth")) {
            return "ZoeDepthForDepthEstimation";
        }
        if (normalized.equals("dpt") || normalized.contains("dpt_") || normalized.contains("dpt-")) {
            return "DPTForDepthEstimation";
        }
        
        // Vision transformers
        if (normalized.contains("segformer")) {
            return "SegformerModel";
        }
        if (normalized.contains("beit")) {
            return "BeitModel";
        }
        if (normalized.contains("dinov2_with_registers")
                || normalized.contains("dinov2-with-registers")) {
            return "Dinov2WithRegistersModel";
        }
        if (normalized.contains("dinov2")) {
            return "Dinov2Model";
        }
        if (normalized.contains("deit")) {
            return "DeiTModel";
        }
        if (normalized.contains("focalnet")) {
            return "FocalNetModel";
        }
        if (normalized.equals("cvt") || normalized.contains("cvt_") || normalized.contains("cvt-")) {
            return "CvtModel";
        }
        if (normalized.contains("levit")) {
            return "LevitModel";
        }
        if (normalized.contains("mobilevitv2") || normalized.contains("mobilevit_v2")
                || normalized.contains("mobilevit-v2")) {
            return "MobileViTV2Model";
        }
        if (normalized.contains("mobilevit")) {
            return "MobileViTModel";
        }
        if (normalized.contains("poolformer")) {
            return "PoolFormerModel";
        }
        if (normalized.contains("pvt_v2") || normalized.contains("pvt-v2")) {
            return "PvtV2Model";
        }
        if (normalized.equals("pvt") || normalized.contains("pvt_") || normalized.contains("pvt-")) {
            return "PvtModel";
        }
        if (normalized.contains("vit")) {
            return "ViTModel";
        }
        if (normalized.contains("donut")) {
            return "DonutSwinModel";
        }
        if (normalized.contains("swinv2") || normalized.contains("swin_v2")) {
            return "Swinv2Model";
        }
        if (normalized.contains("swin")) {
            return "SwinModel";
        }
        
        // CNN models
        if (normalized.contains("convnextv2") || normalized.contains("convnext_v2")) {
            return "ConvNextV2Model";
        }
        if (normalized.contains("convnext")) {
            return "ConvNextModel";
        }
        if (normalized.contains("efficientnet")) {
            return "EfficientNetModel";
        }
        if (normalized.contains("mobilenet_v2") || normalized.contains("mobilenet-v2")) {
            return "MobileNetV2Model";
        }
        if (normalized.contains("mobilenet_v1") || normalized.contains("mobilenet-v1")) {
            return "MobileNetV1Model";
        }
        if (normalized.contains("resnet")) {
            return "ResNetModel";
        }
        if (normalized.contains("regnet")) {
            return "RegNetModel";
        }
        if (normalized.contains("vits")) {
            return "VitsModel";
        }
        
        // Data2Vec
        if (normalized.contains("data2vec_audio") || normalized.contains("data2vec-audio")) {
            return "Data2VecAudioModel";
        }
        if (normalized.contains("data2vec_vision") || normalized.contains("data2vec-vision")) {
            return "Data2VecVisionModel";
        }
        if (normalized.contains("data2vec_text") || normalized.contains("data2vec-text")) {
            return "Data2VecTextModel";
        }
        
        // Long context
        if (normalized.equals("led") || normalized.contains("led_") || normalized.contains("led-")) {
            return "LEDForConditionalGeneration";
        }
        if (normalized.contains("longformer")) {
            return "LongformerModel";
        }
        if (normalized.contains("reformer")) {
            return "ReformerModelWithLMHead";
        }
        
        // BERT variants
        if (normalized.contains("modernbert_decoder")
                || normalized.contains("modernbert-decoder")
                || normalized.contains("modern_bert_decoder")) {
            return "ModernBertDecoderForCausalLM";
        }
        if (normalized.contains("modernbert") || normalized.contains("modern_bert")) {
            return "ModernBertModel";
        }
        if (normalized.contains("modernvbert") || normalized.contains("modern_vbert")
                || normalized.contains("modern-vbert")) {
            return "ModernVBertForMaskedLM";
        }
        if (normalized.contains("nomic_bert") || normalized.contains("nomic-bert")) {
            return "NomicBertModel";
        }
        if (normalized.contains("eurobert") || normalized.contains("euro_bert")) {
            return "EuroBertModel";
        }
        if (normalized.contains("distilbert") || normalized.contains("distil_bert")) {
            return "DistilBertModel";
        }
        if (normalized.contains("rembert") || normalized.contains("rem_bert")) {
            return "RemBertModel";
        }
        if (normalized.contains("mobilebert") || normalized.contains("mobile_bert")) {
            return "MobileBertModel";
        }
        if (normalized.contains("megatron_bert") || normalized.contains("megatron-bert")) {
            return "MegatronBertModel";
        }
        if (normalized.contains("squeezebert") || normalized.contains("squeeze_bert")) {
            return "SqueezeBertModel";
        }
        if (normalized.contains("bert_japanese") || normalized.contains("bert-japanese")) {
            return "BertModel";
        }
        if (normalized.contains("bert_generation") || normalized.contains("bert-generation")) {
            return "BertGenerationDecoder";
        }
        if (normalized.contains("convbert") || normalized.contains("conv_bert")) {
            return "ConvBertModel";
        }
        if (normalized.contains("roc_bert") || normalized.contains("roc-bert")) {
            return "RoCBertForMaskedLM";
        }
        if (normalized.contains("wav2vec2_bert") || normalized.contains("wav2vec2-bert")) {
            return "Wav2Vec2BertForCTC";
        }
        if (normalized.contains("bert")) {
            return "BertModel";
        }
        
        // RoBERTa variants
        if (normalized.contains("xlm_roberta_xl") || normalized.contains("xlm-roberta-xl")) {
            return "XLMRobertaXLModel";
        }
        if (normalized.contains("xlm_roberta") || normalized.contains("xlm-roberta")) {
            return "XLMRobertaModel";
        }
        if (normalized.contains("jina_embeddings_v3") || normalized.contains("jina-embeddings-v3")) {
            return "JinaEmbeddingsV3Model";
        }
        if (normalized.contains("camembert") || normalized.contains("bertweet")
                || normalized.contains("herbert") || normalized.contains("phobert")) {
            return "RobertaModel";
        }
        if (normalized.contains("roberta_prelayernorm")
                || normalized.contains("roberta-prelayernorm")
                || normalized.contains("roberta_pre_layer_norm")
                || normalized.contains("roberta-pre-layer-norm")) {
            return "RobertaPreLayerNormModel";
        }
        if (normalized.contains("roberta")) {
            return "RobertaModel";
        }
        
        // Other models
        if (normalized.contains("luke")) {
            return "LukeModel";
        }
        if (normalized.contains("funnel")) {
            return "FunnelModel";
        }
        if (normalized.contains("fnet")) {
            return "FNetModel";
        }
        if (normalized.contains("flaubert")) {
            return "FlaubertModel";
        }
        if (normalized.contains("mpnet")) {
            return "MPNetModel";
        }
        if (normalized.contains("xlnet")) {
            return "XLNetModel";
        }
        if (normalized.equals("xmod") || normalized.contains("xmod_") || normalized.contains("xmod-")) {
            return "XmodModel";
        }
        if (normalized.equals("xlm") || normalized.contains("xlm_") || normalized.contains("xlm-")) {
            return "XLMModel";
        }
        if (normalized.contains("electra")) {
            return "ElectraModel";
        }
        if (normalized.contains("albert")) {
            return "AlbertModel";
        }
        if (normalized.contains("roformer")) {
            return "RoFormerModel";
        }
        if (normalized.contains("deberta_v2") || normalized.contains("deberta-v2")) {
            return "DebertaV2Model";
        }
        if (normalized.contains("deberta")) {
            return "DebertaModel";
        }
        if (normalized.contains("esmfold")) {
            return "EsmForProteinFolding";
        }
        if (normalized.equals("esm") || normalized.contains("esm_") || normalized.contains("esm-")) {
            return "EsmModel";
        }
        if (normalized.contains("biogpt") || normalized.contains("bio_gpt")) {
            return "BioGptForCausalLM";
        }
        if (normalized.equals("opt") || normalized.contains("facebook_opt")) {
            return "OPTForCausalLM";
        }
        if (normalized.contains("ctrl")) {
            return "CTRLLMHeadModel";
        }
        if (normalized.contains("cpmant") || normalized.contains("cpm-ant")) {
            return "CpmAntForCausalLM";
        }
        if (normalized.equals("cpm") || normalized.contains("cpm_") || normalized.contains("cpm-")) {
            return "CpmForCausalLM";
        }
        if (normalized.contains("ernie")) {
            return "ErnieModel";
        }
        if (normalized.contains("canine")) {
            return "CanineModel";
        }
        if (normalized.contains("flava")) {
            return "FlavaModel";
        }
        
        // Default fallback
        return "LlamaForCausalLM";
    }
    
    public Optional<String> detectModelType(Map<String, Object> metadata) {
        Object value = metadata.get("general.architecture");
        if (value == null) {
            value = metadata.get("model_type");
        }
        if (value == null) {
            return Optional.empty();
        }
        String text = value.toString().trim();
        return text.isEmpty() ? Optional.empty() : Optional.of(text);
    }
}
```

## 3. JSON Config Merger
```java
// JsonConfigMerger.java
package tech.kayys.gollek.spi.model.loader;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Merges text_config and other nested JSON structures into the root config.
 */
public class JsonConfigMerger {
    
    private static final Logger log = LoggerFactory.getLogger(JsonConfigMerger.class);
    
    private final ObjectMapper objectMapper;
    private final RopeParameterMerger ropeParameterMerger;
    
    public JsonConfigMerger(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
        this.ropeParameterMerger = new RopeParameterMerger();
    }
    
    /**
     * Merge text_config fields into root config.
     */
    public void mergeTextConfig(ModelConfig cfg, Path configPath) throws IOException {
        JsonNode root = objectMapper.readTree(configPath.toFile());
        JsonNode textCfg = root.path("text_config");
        
        if (textCfg == null || textCfg.isMissingNode()) {
            return;
        }
        
        // Merge all fields from text_config into config
        mergeFieldIfMissing(root, textCfg, cfg, "hidden_size", 
            (c, v) -> c.setHiddenSize(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "num_hidden_layers",
            (c, v) -> c.setNumHiddenLayers(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "num_attention_heads",
            (c, v) -> c.setNumAttentionHeads(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "num_key_value_heads",
            (c, v) -> c.setNumKeyValueHeads(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "intermediate_size",
            (c, v) -> c.setIntermediateSize(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "vocab_size",
            (c, v) -> c.setVocabSize(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "rms_norm_eps",
            (c, v) -> c.setRmsNormEps(v.asDouble()));
        mergeFieldIfMissing(root, textCfg, cfg, "rope_theta",
            (c, v) -> c.setRopeTheta(v.asDouble()));
        mergeFieldIfMissing(root, textCfg, cfg, "head_dim",
            (c, v) -> c.setHeadDim(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "global_head_dim",
            (c, v) -> c.setGlobalHeadDim(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "num_global_key_value_heads",
            (c, v) -> c.setNumGlobalKeyValueHeads(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "attention_k_eq_v",
            (c, v) -> c.setAttentionKeyEqualsValue(v.asBoolean()));
        mergeFieldIfMissing(root, textCfg, cfg, "num_kv_shared_layers",
            (c, v) -> c.setNumKvSharedLayers(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "enable_moe_block",
            (c, v) -> c.setEnableMoeBlock(v.asBoolean()));
        mergeFieldIfMissing(root, textCfg, cfg, "moe_intermediate_size",
            (c, v) -> c.setMoeIntermediateSize(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "use_double_wide_mlp",
            (c, v) -> c.setUseDoubleWideMlp(v.asBoolean()));
        mergeFieldIfMissing(root, textCfg, cfg, "query_pre_attn_scalar",
            (c, v) -> c.setQueryPreAttnScalar(v.asDouble()));
        mergeFieldIfMissing(root, textCfg, cfg, "attn_logit_softcapping",
            (c, v) -> c.setAttnLogitSoftcapping(v.asDouble()));
        mergeFieldIfMissing(root, textCfg, cfg, "final_logit_softcapping",
            (c, v) -> c.setFinalLogitSoftcapping(v.asDouble()));
        mergeFieldIfMissing(root, textCfg, cfg, "hidden_size_per_layer_input",
            (c, v) -> c.setHiddenSizePerLayerInput(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "vocab_size_per_layer_input",
            (c, v) -> c.setVocabSizePerLayerInput(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "bos_token_id",
            (c, v) -> c.setBosTokenId(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "eos_token_id",
            (c, v) -> c.setEosTokenId(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "pad_token_id",
            (c, v) -> c.setPadTokenId(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "sliding_window",
            (c, v) -> c.setSlidingWindow(v.asInt()));
        mergeFieldIfMissing(root, textCfg, cfg, "use_sliding_window",
            (c, v) -> c.setUseSlidingWindow(v.asBoolean()));
        mergeFieldIfMissing(root, textCfg, cfg, "max_position_embeddings",
            (c, v) -> c.setMaxPositionEmbeddings(v.asInt()));
        
        // Special handling for num_local_experts
        if (!root.has("num_local_experts") && !root.has("num_experts")) {
            Integer v = intValue(textCfg, "num_local_experts");
            if (v == null) {
                v = intValue(textCfg, "num_experts");
            }
            if (v != null) {
                cfg.setNumLocalExperts(v);
            }
        }
        
        // Special handling for num_experts_per_tok
        if (!root.has("num_experts_per_tok") && !root.has("top_k_experts")) {
            Integer v = intValue(textCfg, "num_experts_per_tok");
            if (v == null) {
                v = intValue(textCfg, "top_k_experts");
            }
            if (v != null) {
                cfg.setNumExpertsPerTok(v);
            }
        }
        
        // Handle hidden_activation
        if (!root.has("hidden_act")) {
            String act = textValue(textCfg, "hidden_activation");
            if (act == null || act.isBlank()) {
                act = textValue(textCfg, "hidden_act");
            }
            if (act != null && !act.isBlank()) {
                cfg.setHiddenAct(act);
            }
        }
        
        // Handle layer_types
        if (!root.has("layer_types")) {
            JsonNode types = textCfg.get("layer_types");
            if (types != null && types.isArray()) {
                List<String> values = new ArrayList<>();
                types.forEach(n -> values.add(n.asText()));
                cfg.setLayerTypes(values);
            }
        }
        
        // Handle rope_parameters
        JsonNode ropeParams = textCfg.get("rope_parameters");
        if (ropeParams != null && ropeParams.isObject()) {
            ropeParameterMerger.merge(cfg, ropeParams);
        }
        
        // Handle partial_rotary_factor
        if (!root.has("partial_rotary_factor")) {
            Double v = doubleValue(textCfg, "partial_rotary_factor");
            if (v != null) {
                cfg.setPartialRotaryFactor(v);
            }
        }
        
        if (!root.has("rope_local_base_freq")) {
            Double v = doubleValue(textCfg, "rope_local_base_freq");
            if (v != null) {
                cfg.setRopeLocalBaseFreq(v);
            }
        }
    }
    
    @FunctionalInterface
    private interface FieldMerger {
        void merge(ModelConfig config, JsonNode value);
    }
    
    private void mergeFieldIfMissing(JsonNode root, JsonNode textConfig, 
                                     ModelConfig cfg, String field, FieldMerger merger) {
        if (!root.has(field)) {
            JsonNode value = textConfig.get(field);
            if (value != null && !value.isNull()) {
                merger.merge(cfg, value);
            }
        }
    }
    
    private Integer intValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asInt() : null;
    }
    
    private Double doubleValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asDouble() : null;
    }
    
    private String textValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isTextual() ? v.asText() : v.toString();
    }
}
```

## 4. Rope Parameter Merger
```java
// RopeParameterMerger.java
package tech.kayys.gollek.spi.model.loader;

import com.fasterxml.jackson.databind.JsonNode;
import tech.kayys.gollek.spi.model.ModelConfig;

/**
 * Merges rope_parameters from JSON into ModelConfig.
 */
public class RopeParameterMerger {
    
    public void merge(ModelConfig cfg, JsonNode ropeParams) {
        if (ropeParams == null || !ropeParams.isObject()) {
            return;
        }
        
        // Root params
        Double rootTheta = doubleValue(ropeParams, "rope_theta");
        if (rootTheta != null) {
            cfg.setRopeTheta(rootTheta);
        }
        
        Double rootPartial = doubleValue(ropeParams, "partial_rotary_factor");
        if (rootPartial != null) {
            cfg.setPartialRotaryFactor(rootPartial);
        }
        
        String rootType = textValue(ropeParams, "rope_type");
        if (rootType != null) {
            ModelConfig.RopeScaling scaling = cfg.getRopeScaling();
            if (scaling == null) {
                scaling = new ModelConfig.RopeScaling();
                cfg.setRopeScaling(scaling);
            }
            scaling.setType(rootType);
        }
        
        // Full attention params
        JsonNode full = ropeParams.get("full_attention");
        if (full != null && full.isObject()) {
            Double v = doubleValue(full, "rope_theta");
            if (v != null) {
                cfg.setRopeThetaFull(v);
            }
            Double p = doubleValue(full, "partial_rotary_factor");
            if (p != null) {
                cfg.setPartialRotaryFactorFull(p);
            }
        }
        
        // Sliding attention params
        JsonNode sliding = ropeParams.get("sliding_attention");
        if (sliding != null && sliding.isObject()) {
            Double v = doubleValue(sliding, "rope_theta");
            if (v != null) {
                cfg.setRopeThetaSliding(v);
            }
            Double p = doubleValue(sliding, "partial_rotary_factor");
            if (p != null) {
                cfg.setPartialRotaryFactorSliding(p);
            }
        }
    }
    
    private Double doubleValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asDouble() : null;
    }
    
    private String textValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isTextual() ? v.asText() : v.toString();
    }
}
```

## 5. ModelConfigLoader (with factory methods)
```java
// ModelConfigLoader.java
package tech.kayys.gollek.spi.model.loader;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads ModelConfig from JSON files.
 */
public class ModelConfigLoader {
    
    private static final Logger log = LoggerFactory.getLogger(ModelConfigLoader.class);
    
    private final ObjectMapper objectMapper;
    private final JsonConfigMerger jsonMerger;
    
    public ModelConfigLoader(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
        this.jsonMerger = new JsonConfigMerger(objectMapper);
    }
    
    /**
     * Parse {@code config.json} from an explicit file path.
     */
    public ModelConfig load(Path configPath) throws IOException {
        if (!Files.exists(configPath)) {
            throw new IOException("config.json not found: " + configPath);
        }
        
        ModelConfig cfg = objectMapper.readValue(configPath.toFile(), ModelConfig.class);
        
        // Apply fallback inference for missing fields
        inferMissingFields(cfg, configPath);
        
        // Merge text_config fields
        jsonMerger.mergeTextConfig(cfg, configPath);
        
        // Apply post-load reconciliation
        postProcess(cfg);
        
        log.info("Loaded model config: type={} arch={} layers={} hidden={} heads={} kvHeads={}",
                cfg.getModelType(), cfg.getPrimaryArchitecture(),
                cfg.getNumHiddenLayers(), cfg.getHiddenSize(),
                cfg.getNumAttentionHeads(), cfg.getResolvedNumKvHeads());
        
        return cfg;
    }
    
    /**
     * Load {@code config.json} from the model directory.
     */
    public ModelConfig loadFromDirectory(Path modelDir) throws IOException {
        return load(modelDir.resolve("config.json"));
    }
    
    private void inferMissingFields(ModelConfig cfg, Path configPath) throws IOException {
        if ((cfg.getModelType() == null || cfg.getModelType().isBlank())
                && (cfg.getArchitectures() == null || cfg.getArchitectures().isEmpty())) {
            
            JsonNode root = objectMapper.readTree(configPath.toFile());
            String inferredType = extractTextValue(root, "model_type");
            List<String> inferredArch = extractListValue(root, "architectures");
            
            if (inferredType == null || inferredType.isBlank()) {
                inferredType = extractTextValue(root.path("text_config"), "model_type");
            }
            if (inferredArch == null || inferredArch.isEmpty()) {
                inferredArch = extractListValue(root.path("text_config"), "architectures");
            }
            
            if ((inferredType != null && !inferredType.isBlank())
                    || (inferredArch != null && !inferredArch.isEmpty())) {
                cfg.setModelType(inferredType);
                cfg.setArchitectures(inferredArch);
                log.info("Inferred fields from fallback: type={} arch={}",
                        cfg.getModelType(), cfg.getPrimaryArchitecture());
            }
        }
    }
    
    private void postProcess(ModelConfig cfg) {
        reconcileGemma3RopeDefaults(cfg);
        reconcileGemma4Defaults(cfg);
    }
    
    private void reconcileGemma3RopeDefaults(ModelConfig cfg) {
        if (cfg.getModelType() == null) return;
        if (!cfg.getModelType().toLowerCase().startsWith("gemma3")) return;
        
        if (cfg.getRopeThetaFull() == null) {
            cfg.setRopeThetaFull(cfg.getRopeTheta());
        }
        if (cfg.getRopeThetaSliding() == null) {
            Double localBase = cfg.getRopeLocalBaseFreq();
            cfg.setRopeThetaSliding(localBase != null ? localBase : 10000.0);
        }
    }
    
    private void reconcileGemma4Defaults(ModelConfig cfg) {
        if (cfg.getModelType() == null) return;
        if (!cfg.getModelType().toLowerCase().startsWith("gemma4")) return;
        
        // Gemma4 specific defaults
        if (cfg.getQueryPreAttnScalar() == null) {
            cfg.setQueryPreAttnScalar((double) cfg.getResolvedHeadDim());
        }
    }
    
    private String extractTextValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode()) return null;
        JsonNode v = node.get(field);
        return (v != null && v.isTextual()) ? v.asText() : null;
    }
    
    private List<String> extractListValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode()) return null;
        JsonNode v = node.get(field);
        if (v == null || !v.isArray()) return null;
        
        List<String> out = new ArrayList<>();
        v.forEach(n -> {
            if (n != null && !n.isNull()) out.add(n.asText());
        });
        return out;
    }
}
```

## 6. GGUF Metadata Mapper
```java
// GgufMetadataMapper.java
package tech.kayys.gollek.spi.model.mapper;

import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.*;

/**
 * Maps GGUF metadata to ModelConfig.
 */
public class GgufMetadataMapper {
    
    private final ArchitectureMapper architectureMapper;
    
    public GgufMetadataMapper() {
        this.architectureMapper = new ArchitectureMapper();
    }
    
    public GgufMetadataMapper(ArchitectureMapper architectureMapper) {
        this.architectureMapper = architectureMapper;
    }
    
    /**
     * Build a model config from GGUF metadata.
     */
    public ModelConfig fromGgufMetadata(Map<String, Object> metadata) {
        Map<String, Object> meta = metadata != null ? metadata : Map.of();
        
        String arch = architectureMapper.detectModelType(meta)
                .orElse("llama")
                .trim()
                .toLowerCase(Locale.ROOT);
        if (arch.isBlank()) {
            arch = "llama";
        }
        
        ModelConfig cfg = new ModelConfig();
        cfg.setModelType(arch);
        cfg.setArchitectures(List.of(architectureMapper.mapGgufToHfClassName(arch)));
        cfg.setMaxPositionEmbeddings(extractInt(meta, cfg.getMaxPositionEmbeddings(),
                arch + ".context_length", "general.context_length"));
        cfg.setHiddenSize(extractInt(meta, cfg.getHiddenSize(),
                arch + ".embedding_length", "general.embedding_length"));
        cfg.setNumHiddenLayers(extractInt(meta, cfg.getNumHiddenLayers(),
                arch + ".block_count", "general.block_count"));
        cfg.setIntermediateSize(extractMaxInt(meta, cfg.getIntermediateSize(),
                arch + ".feed_forward_length", "general.feed_forward_length"));
        cfg.setNumAttentionHeads(extractInt(meta, cfg.getNumAttentionHeads(),
                arch + ".attention.head_count", "general.attention.head_count"));
        cfg.setNumKeyValueHeads(extractInt(meta, cfg.getNumAttentionHeads(),
                arch + ".attention.head_count_kv", "general.attention.head_count_kv"));
        cfg.setVocabSize(extractInt(meta,
                extractTokenCount(meta).orElse(cfg.getVocabSize()),
                arch + ".vocab_size", "general.vocab_size"));
        cfg.setRmsNormEps(extractDouble(meta, cfg.getRmsNormEps(),
                arch + ".attention.layer_norm_rms_epsilon",
                "general.attention.layer_norm_rms_epsilon"));
        cfg.setRopeTheta(extractDouble(meta, cfg.getRopeTheta(),
                arch + ".rope.freq_base", "general.rope.freq_base"));
        cfg.setRopeThetaFull(cfg.getRopeTheta());
        
        // Head dimensions
        extractHeadDimensions(cfg, meta, arch);
        
        // Other fields
        extractOptionalFields(cfg, meta, arch);
        
        // Special tokens
        extractSpecialTokens(cfg, meta);
        
        // Layer types
        extractLayerTypes(cfg, meta, arch);
        
        return cfg;
    }
    
    private void extractHeadDimensions(ModelConfig cfg, Map<String, Object> meta, String arch) {
        Optional<Integer> localHeadDim = extractIntOptional(meta,
                arch + ".attention.head_dim",
                arch + ".attention.key_length_swa",
                arch + ".rope.dimension_count_swa",
                "general.attention.head_dim");
        
        Optional<Integer> globalHeadDim = extractIntOptional(meta,
                arch + ".attention.key_length",
                arch + ".rope.dimension_count");
        
        if (localHeadDim.isEmpty() && globalHeadDim.isPresent()) {
            localHeadDim = globalHeadDim;
        }
        
        if (localHeadDim.isEmpty()) {
            int computed = cfg.getNumAttentionHeads() > 0 
                    ? cfg.getHiddenSize() / cfg.getNumAttentionHeads() 
                    : 0;
            if (computed > 0) {
                cfg.setHeadDim(computed);
            }
        } else {
            cfg.setHeadDim(localHeadDim.get());
        }
        
        globalHeadDim.ifPresent(dim -> {
            if (!dim.equals(cfg.getHeadDim())) {
                cfg.setGlobalHeadDim(dim);
            }
        });
    }
    
    private void extractOptionalFields(ModelConfig cfg, Map<String, Object> meta, String arch) {
        extractIntOptional(meta, arch + ".attention.sliding_window")
                .ifPresent(value -> {
                    cfg.setSlidingWindow(value);
                    cfg.setUseSlidingWindow(value > 0);
                });
        
        extractIntOptional(meta, arch + ".attention.shared_kv_layers")
                .ifPresent(cfg::setNumKvSharedLayers);
        
        extractIntOptional(meta, arch + ".embedding_length_per_layer_input")
                .ifPresent(cfg::setHiddenSizePerLayerInput);
        
        extractIntOptional(meta, arch + ".vocab_size_per_layer_input")
                .ifPresent(cfg::setVocabSizePerLayerInput);
        
        extractDoubleOptional(meta, arch + ".rope.freq_base_swa")
                .ifPresent(value -> {
                    cfg.setRopeLocalBaseFreq(value);
                    cfg.setRopeThetaSliding(value);
                });
        
        extractDoubleOptional(meta, arch + ".final_logit_softcapping")
                .ifPresent(cfg::setFinalLogitSoftcapping);
        
        // MoE fields
        extractIntOptional(meta, arch + ".num_local_experts")
                .ifPresent(cfg::setNumLocalExperts);
        extractIntOptional(meta, arch + ".num_experts_per_tok")
                .ifPresent(cfg::setNumExpertsPerTok);
        extractIntOptional(meta, arch + ".decoder_sparse_step")
                .ifPresent(cfg::setDecoderSparseStep);
        extractBooleanOptional(meta, arch + ".enable_moe_block")
                .ifPresent(cfg::setEnableMoeBlock);
        extractIntOptional(meta, arch + ".moe_intermediate_size")
                .ifPresent(cfg::setMoeIntermediateSize);
        extractBooleanOptional(meta, arch + ".use_double_wide_mlp")
                .ifPresent(cfg::setUseDoubleWideMlp);
        
        // Head dimensions
        extractIntOptional(meta, arch + ".num_global_key_value_heads")
                .ifPresent(cfg::setNumGlobalKeyValueHeads);
        extractBooleanOptional(meta, arch + ".attention_k_eq_v")
                .ifPresent(cfg::setAttentionKeyEqualsValue);
    }
    
    private void extractSpecialTokens(ModelConfig cfg, Map<String, Object> meta) {
        extractIntOptional(meta, "tokenizer.ggml.bos_token_id")
                .ifPresent(cfg::setBosTokenId);
        
        extractIntOptional(meta, "tokenizer.ggml.eos_token_id")
                .ifPresent(value -> {
                    cfg.setEosTokenId(value);
                    cfg.setEosTokenIds(List.of(value));
                });
        
        extractIntOptional(meta, "tokenizer.ggml.padding_token_id")
                .or(() -> extractIntOptional(meta, "tokenizer.ggml.pad_token_id"))
                .ifPresent(cfg::setPadTokenId);
    }
    
    private void extractLayerTypes(ModelConfig cfg, Map<String, Object> meta, String arch) {
        Object value = meta.get(arch + ".attention.sliding_window_pattern");
        if (value instanceof List<?> list && !list.isEmpty()) {
            List<String> types = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Boolean sliding) {
                    types.add(sliding ? "sliding_attention" : "full_attention");
                }
            }
            if (!types.isEmpty()) {
                cfg.setLayerTypes(types);
            }
        }
    }
    
    private Optional<Integer> extractTokenCount(Map<String, Object> metadata) {
        Object tokens = metadata.get("tokenizer.ggml.tokens");
        if (tokens instanceof List<?> list) {
            return list.isEmpty() ? Optional.empty() : Optional.of(list.size());
        }
        if (tokens != null && tokens.getClass().isArray()) {
            return Optional.of(java.lang.reflect.Array.getLength(tokens));
        }
        return Optional.empty();
    }
    
    private int extractInt(Map<String, Object> meta, int fallback, String... keys) {
        return extractIntOptional(meta, keys).orElse(fallback);
    }
    
    private Optional<Integer> extractIntOptional(Map<String, Object> meta, String... keys) {
        for (String key : keys) {
            Optional<Integer> value = firstNumericInt(meta.get(key));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }
    
    private int extractMaxInt(Map<String, Object> meta, int fallback, String... keys) {
        for (String key : keys) {
            Optional<Integer> value = maxNumericInt(meta.get(key));
            if (value.isPresent()) {
                return value.get();
            }
        }
        return fallback;
    }
    
    private double extractDouble(Map<String, Object> meta, double fallback, String... keys) {
        return extractDoubleOptional(meta, keys).orElse(fallback);
    }
    
    private Optional<Double> extractDoubleOptional(Map<String, Object> meta, String... keys) {
        for (String key : keys) {
            Optional<Double> value = firstNumericDouble(meta.get(key));
            if (value.isPresent()) {
                return value;
            }
        }
        return Optional.empty();
    }
    
    private Optional<Boolean> extractBooleanOptional(Map<String, Object> meta, String key) {
        Object value = meta.get(key);
        if (value instanceof Boolean) {
            return Optional.of((Boolean) value);
        }
        return Optional.empty();
    }
    
    private Optional<Integer> firstNumericInt(Object value) {
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
    
    private Optional<Integer> maxNumericInt(Object value) {
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
    
    private Optional<Double> firstNumericDouble(Object value) {
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
}
```

## 7. Usage Example
```java
// Usage example
public class ConfigUsageExample {
    public static void main(String[] args) throws IOException {
        // Setup
        ObjectMapper mapper = new ObjectMapper();
        ModelConfigLoader loader = new ModelConfigLoader(mapper);
        GgufMetadataMapper ggufMapper = new GgufMetadataMapper();
        
        // Load from JSON
        Path configPath = Path.of("/path/to/model/config.json");
        ModelConfig config = loader.load(configPath);
        
        // Load from GGUF metadata
        Map<String, Object> metadata = Map.of(
            "general.architecture", "llama",
            "llama.embedding_length", 4096,
            "llama.block_count", 32,
            "llama.attention.head_count", 32
        );
        ModelConfig ggufConfig = ggufMapper.fromGgufMetadata(metadata);
        
        // Use config properties
        System.out.println("Model: " + config.getModelType());
        System.out.println("Hidden size: " + config.getHiddenSize());
        System.out.println("Num heads: " + config.getNumAttentionHeads());
        System.out.println("KV heads: " + config.getResolvedNumKvHeads());
        System.out.println("Is MoE: " + config.isMoe());
        System.out.println("Has sliding window: " + config.hasSlidingWindow());
    }
}
```

This refactoring achieves:

1. **Separation of Concerns**: Each class has a single responsibility
2. **Preserved Jackson annotations**: All `@JsonProperty` and `@JsonSetter` annotations are kept
3. **Maintained all original functionality**: All computed properties and helper methods are preserved
4. **Extracted the 800+ line architecture mapping** into its own class
5. **No Lombok**: Pure Java with manual getters/setters
6. **Modular**: Easy to test and maintain each component independently

The refactored structure is much more maintainable while preserving all the original behavior and Jackson deserialization capabilities.
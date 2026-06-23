// JsonConfigMerger.java
package tech.kayys.gollek.spi.model.loader;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.logging.Logger;

import tech.kayys.gollek.spi.model.ModelConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Merges text_config and other nested JSON structures into the root config.
 */
public class JsonConfigMerger {
    
    private static final Logger log = Logger.getLogger(JsonConfigMerger.class.getName());
    
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

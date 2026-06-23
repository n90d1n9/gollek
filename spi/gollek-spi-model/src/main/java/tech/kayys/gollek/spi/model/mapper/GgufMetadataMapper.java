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

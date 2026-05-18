package tech.kayys.gollek.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelConfigGgufMetadataTest {

    @Test
    void mapsLlamaGgufMetadataToDirectEngineConfig() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "llama");
        metadata.put("llama.context_length", 8192);
        metadata.put("llama.embedding_length", 4096);
        metadata.put("llama.block_count", 32);
        metadata.put("llama.feed_forward_length", 11008);
        metadata.put("llama.attention.head_count", 32);
        metadata.put("llama.attention.head_count_kv", 8);
        metadata.put("llama.attention.layer_norm_rms_epsilon", 0.00001d);
        metadata.put("llama.rope.freq_base", 500000.0d);
        metadata.put("tokenizer.ggml.tokens", List.of("a", "b", "c"));
        metadata.put("tokenizer.ggml.bos_token_id", 1);
        metadata.put("tokenizer.ggml.eos_token_id", 2);

        ModelConfig config = ModelConfig.fromGgufMetadata(metadata);

        assertEquals("llama", config.modelType());
        assertEquals("LlamaForCausalLM", config.primaryArchitecture());
        assertEquals(8192, config.maxPositionEmbeddings());
        assertEquals(4096, config.hiddenSize());
        assertEquals(32, config.numHiddenLayers());
        assertEquals(11008, config.intermediateSize());
        assertEquals(32, config.numAttentionHeads());
        assertEquals(8, config.resolvedNumKvHeads());
        assertEquals(128, config.resolvedHeadDim());
        assertEquals(3, config.vocabSize());
        assertEquals(500000.0d, config.ropeTheta(), 0.0001d);
        assertEquals(1, config.bosTokenId().orElseThrow());
        assertEquals(2, config.eosTokenId().orElseThrow());
    }

    @Test
    void mapsGemma4GgufMetadataIncludingHeterogeneousAttentionHints() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "gemma4");
        metadata.put("gemma4.context_length", 32768);
        metadata.put("gemma4.embedding_length", 2304);
        metadata.put("gemma4.block_count", 5);
        metadata.put("gemma4.feed_forward_length", List.of(6144, 6144, 12288));
        metadata.put("gemma4.attention.head_count", 8);
        metadata.put("gemma4.attention.head_count_kv", 4);
        metadata.put("gemma4.attention.key_length", 256);
        metadata.put("gemma4.attention.key_length_swa", 128);
        metadata.put("gemma4.rope.freq_base", 1_000_000.0d);
        metadata.put("gemma4.rope.freq_base_swa", 10_000.0d);
        metadata.put("gemma4.attention.shared_kv_layers", 2);
        metadata.put("gemma4.attention.sliding_window", 512);
        metadata.put("gemma4.attention.sliding_window_pattern", List.of(true, true, false, true, false));
        metadata.put("gemma4.embedding_length_per_layer_input", 256);
        metadata.put("gemma4.vocab_size", 262144);
        metadata.put("gemma4.final_logit_softcapping", 30.0d);

        ModelConfig config = ModelConfig.fromGgufMetadata(metadata);

        assertEquals("gemma4", config.modelType());
        assertEquals("Gemma4ForConditionalGeneration", config.primaryArchitecture());
        assertEquals(32768, config.maxPositionEmbeddings());
        assertEquals(2304, config.hiddenSize());
        assertEquals(5, config.numHiddenLayers());
        assertEquals(12288, config.intermediateSize());
        assertEquals(8, config.numAttentionHeads());
        assertEquals(4, config.resolvedNumKvHeads());
        assertEquals(128, config.resolvedHeadDim());
        assertEquals(256, config.resolvedMaxHeadDim());
        assertEquals(2, config.resolvedNumKvSharedLayers());
        assertEquals(256, config.hiddenSizePerLayerInput());
        assertEquals(262144, config.vocabSize());
        assertTrue(config.hasSlidingWindow());
        assertEquals(512, config.slidingWindowSize());
        assertEquals("sliding_attention", config.layerType(0));
        assertEquals("full_attention", config.layerType(2));
        assertEquals(1_000_000.0d, config.ropeThetaForLayer(2), 0.0001d);
        assertEquals(10_000.0d, config.ropeThetaForLayer(0), 0.0001d);
        assertEquals(30.0d, config.finalLogitSoftcapping(), 0.0001d);
    }
}

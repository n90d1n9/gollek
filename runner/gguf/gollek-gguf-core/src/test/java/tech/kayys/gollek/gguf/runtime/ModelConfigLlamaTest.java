package tech.kayys.gollek.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ModelConfigLlamaTest {
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
}

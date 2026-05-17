package tech.kayys.gollek.spi.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelConfigGemma4TextMergeTest {

    @Test
    void mergesSlidingWindowMaxPositionsAndActivationFromTextConfig() throws IOException {
        String json = """
                {
                  "model_type": "gemma4",
                  "architectures": ["Gemma4ForConditionalGeneration"],
                  "dtype": "bfloat16",
                  "eos_token_id": [1, 106],
                  "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 2560,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "intermediate_size": 10240,
                    "vocab_size": 262144,
                    "sliding_window": 512,
                    "max_position_embeddings": 131072,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "head_dim": 256,
                    "layer_types": ["sliding_attention", "full_attention"]
                  }
                }
                """;
        Path dir = Files.createTempDirectory("gollek-modelconfig-test");
        Path cfgPath = dir.resolve("config.json");
        Files.writeString(cfgPath, json, StandardCharsets.UTF_8);
        try {
            ModelConfig cfg = ModelConfig.load(cfgPath, new ObjectMapper());
            assertEquals(512, cfg.slidingWindowSize());
            assertEquals(131072, cfg.maxPositionEmbeddings());
            assertEquals("gelu_pytorch_tanh", cfg.hiddenAct());
            assertEquals(2560, cfg.hiddenSize());
            assertEquals(2, cfg.numHiddenLayers());
            assertTrue(cfg.isSlidingAttentionLayer(0));
            assertEquals("sliding_attention", cfg.layerType(0));
            assertEquals("full_attention", cfg.layerType(1));
        } finally {
            Files.deleteIfExists(cfgPath);
            Files.deleteIfExists(dir);
        }
    }
}

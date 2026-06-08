/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionPackedQkvTest {

    @Test
    void resolvesPhiMiniPackedQkvHeadLayout() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "architectures": ["Phi3ForCausalLM"],
                  "hidden_size": 3072,
                  "num_attention_heads": 24,
                  "num_key_value_heads": 8,
                  "partial_rotary_factor": 0.75
                }
                """, ModelConfig.class);
        AccelTensor packedQkvWeight = AccelTensor.view(java.lang.foreign.MemorySegment.NULL,
                new long[] { 5120, 3072 });
        AttentionInput input = new AttentionInput(
                AccelTensor.view(java.lang.foreign.MemorySegment.NULL, new long[] { 1, 9, 3072 }),
                packedQkvWeight, packedQkvWeight, packedQkvWeight, null,
                null, null, null, null,
                new PackedQkvArchitecture(), config, null, 0, 0, true,
                null, null, null, null);

        FlashAttentionHeadLayout layout = FlashAttentionHeadLayout.resolve(input, config, 0);

        assertEquals(24, layout.numQueryHeads());
        assertEquals(8, layout.numKeyValueHeads());
        assertEquals(128, layout.headDim());
        assertTrue(layout.packedQkvProjection());
        assertEquals(3072, layout.queryProjectionDim());
        assertEquals(1024, layout.keyValueProjectionDim());
        assertEquals(5120, layout.packedQkvProjectionDim());
        assertEquals(0.75, config.partialRotaryFactorForLayer(0), 0.0001);
    }

    @Test
    void infersPackedQkvFromSharedProjectionWeightShape() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "architectures": ["Phi3ForCausalLM"],
                  "hidden_size": 3072,
                  "num_attention_heads": 24,
                  "num_key_value_heads": 8
                }
                """, ModelConfig.class);
        AccelTensor sharedPackedQkvWeight = AccelTensor.view(java.lang.foreign.MemorySegment.NULL,
                new long[] { 5120, 3072 });
        AttentionInput input = new AttentionInput(
                AccelTensor.view(java.lang.foreign.MemorySegment.NULL, new long[] { 1, 9, 3072 }),
                sharedPackedQkvWeight, sharedPackedQkvWeight, sharedPackedQkvWeight, null,
                null, null, null, null,
                new NonPackedArchitecture(), config, null, 0, 0, true,
                null, null, null, null);

        FlashAttentionHeadLayout layout = FlashAttentionHeadLayout.resolve(input, config, 0);

        assertTrue(layout.packedQkvProjection());
        assertEquals(128, layout.headDim());
        assertEquals(3072, layout.queryProjectionDim());
        assertEquals(5120, layout.packedQkvProjectionDim());
    }

    @Test
    void rejectsSeparateProjectionLayoutWhenQueryRowsCannotFormHeads() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "architectures": ["Phi3ForCausalLM"],
                  "hidden_size": 5120,
                  "num_attention_heads": 24,
                  "num_key_value_heads": 8
                }
                """, ModelConfig.class);
        AccelTensor qWeight = AccelTensor.view(MemorySegment.NULL, new long[] { 5120, 5120 });
        AccelTensor kWeight = AccelTensor.view(MemorySegment.NULL, new long[] { 1024, 5120 });
        AccelTensor vWeight = AccelTensor.view(MemorySegment.NULL, new long[] { 1024, 5120 });
        AttentionInput input = new AttentionInput(
                AccelTensor.view(MemorySegment.NULL, new long[] { 1, 9, 5120 }),
                qWeight, kWeight, vWeight, null,
                null, null, null, null,
                new NonPackedArchitecture(), config, null, 0, 0, true,
                null, null, null, null);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> FlashAttentionHeadLayout.resolve(input, config, 0));

        String message = error.getMessage();
        assertTrue(message.contains("query projection layout mismatch"));
        assertTrue(message.contains("rows=5120"));
        assertTrue(message.contains("numQueryHeads=24"));
        assertTrue(message.contains("configuredHeadDim=213"));
        assertTrue(message.contains("modelType=phi3"));
        assertTrue(message.contains("model-family runtime traits"));
    }

    @Test
    void splitsPackedQkvProjectionIntoContiguousQueryKeyValueTensors() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "architectures": ["Phi3ForCausalLM"],
                  "hidden_size": 4,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1
                }
                """, ModelConfig.class);
        AccelTensor x = AccelTensor.zeros(1, 2, 4);
        AccelTensor packedQkvWeight = AccelTensor.zeros(8, 4);
        AttentionInput input = new AttentionInput(
                x, packedQkvWeight, packedQkvWeight, packedQkvWeight, null,
                null, null, null, null,
                new PackedQkvArchitecture(), config, null, 0, 0, true,
                null, null, null, null);

        FlashAttentionHeadLayout layout = FlashAttentionHeadLayout.resolve(input, config, 0);
        FlashAttentionProjector projector = new FlashAttentionProjector(null, () -> false);
        FlashAttentionProjector.LinearTriple qkv = projector.projectPackedQkv(input, config,
                FlashAttentionModelPolicy.resolve(input.arch, config), layout);

        assertEquals(4, qkv.first().size(-1));
        assertEquals(2, qkv.second().size(-1));
        assertEquals(2, qkv.third().size(-1));
        assertTrue(qkv.first().isContiguous());
        assertTrue(qkv.second().isContiguous());
        assertTrue(qkv.third().isContiguous());
    }

    @Test
    void providesReusableWorkspaceBuffersForSeparateQueryKeyValueProjections() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "architectures": ["Phi3ForCausalLM"],
                  "hidden_size": 4,
                  "intermediate_size": 16,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1
                }
                """, ModelConfig.class);
        KVCacheManager.KVCacheSession session = new KVCacheManager.KVCacheSession(16, new BlockManager());
        session.allocate(config, GenerationConfig.defaults());
        AccelTensor x = AccelTensor.zeros(1, 2, 4);
        AccelTensor qWeight = AccelTensor.zeros(4, 4);
        AccelTensor kWeight = AccelTensor.zeros(2, 4);
        AccelTensor vWeight = AccelTensor.zeros(2, 4);
        AttentionInput input = new AttentionInput(
                x, qWeight, kWeight, vWeight, null,
                null, null, null, null,
                new NonPackedArchitecture(), config, session, 0, 0, true,
                null, null, null, null);

        FlashAttentionProjector projector = new FlashAttentionProjector(null, () -> false);
        FlashAttentionProjector.ProjectionBuffers buffers = projector.attentionProjectionBuffers(input, true);

        try {
            assertNotNull(buffers);
            assertTrue(buffers.q().hasShape(new long[] { 1, 2, 4 }));
            assertTrue(buffers.k().hasShape(new long[] { 1, 2, 2 }));
            assertTrue(buffers.v().hasShape(new long[] { 1, 2, 2 }));

            long qBytes = buffers.q().numel() * Float.BYTES;
            long kBytes = buffers.k().numel() * Float.BYTES;
            MemorySegment scratch = session.getWorkspace().getCombinedSeg();
            buffers.q().dataPtr().set(ValueLayout.JAVA_FLOAT, 0, 1.25f);
            buffers.k().dataPtr().set(ValueLayout.JAVA_FLOAT, 0, 2.5f);
            buffers.v().dataPtr().set(ValueLayout.JAVA_FLOAT, 0, 3.75f);

            assertEquals(1.25f, scratch.get(ValueLayout.JAVA_FLOAT, 0), 0.0001f);
            assertEquals(2.5f, scratch.get(ValueLayout.JAVA_FLOAT, qBytes), 0.0001f);
            assertEquals(3.75f, scratch.get(ValueLayout.JAVA_FLOAT, qBytes + kBytes), 0.0001f);
        } finally {
            if (buffers != null) {
                buffers.q().close();
                buffers.k().close();
                buffers.v().close();
            }
            x.close();
            qWeight.close();
            kWeight.close();
            vWeight.close();
            session.close();
        }
    }

    private static final class PackedQkvArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "phi";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("Phi3ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("phi3");
        }

        @Override
        public String embedTokensWeight() {
            return "model.embed_tokens.weight";
        }

        @Override
        public String finalNormWeight() {
            return "model.final_layernorm.weight";
        }

        @Override
        public boolean hasFusedQKV() {
            return true;
        }

        @Override
        public String layerQueryWeight(int i) {
            return "model.layers.%d.self_attn.qkv_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return layerQueryWeight(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return layerQueryWeight(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "model.layers.%d.self_attn.o_proj.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "model.layers.%d.input_layernorm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "model.layers.%d.mlp.gate_up_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return layerFfnGateWeight(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "model.layers.%d.mlp.down_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
        }
    }

    private static final class NonPackedArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "generic";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("GenericForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("generic");
        }

        @Override
        public String embedTokensWeight() {
            return "model.embed_tokens.weight";
        }

        @Override
        public String finalNormWeight() {
            return "model.norm.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "model.layers.%d.self_attn.q_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "model.layers.%d.self_attn.k_proj.weight".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "model.layers.%d.self_attn.v_proj.weight".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "model.layers.%d.self_attn.o_proj.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "model.layers.%d.input_layernorm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "model.layers.%d.mlp.gate_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "model.layers.%d.mlp.up_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "model.layers.%d.mlp.down_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
        }
    }
}
